from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        quantize_adapter: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.quantize_adapter = quantize_adapter

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def adapter_weight(self, name: str) -> torch.Tensor:
        weight = getattr(self, name)
        if self.quantize_adapter:
            return fake_int4_dequantize(weight)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        dropped = self.dropout(x).to(torch.float32)
        lora = F.linear(F.linear(dropped, self.adapter_weight("lora_A")), self.adapter_weight("lora_B"))
        return base_out + lora.to(base_out.dtype) * self.scaling


def fake_int4_dequantize(weight: torch.Tensor) -> torch.Tensor:
    max_abs = weight.detach().abs().amax()
    if max_abs == 0:
        return weight
    scale = max_abs / 7.0
    quantized = torch.clamp(torch.round(weight / scale), -8, 7)
    return quantized * scale


def find_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_model(
    model: nn.Module,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    quantize_adapter: bool = False,
) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue
        parent, child_name = find_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout, quantize_adapter))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No linear modules matched LoRA targets: {target_modules}")
    return replaced


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a not in state or key_b not in state:
            missing.append(name)
            continue
        module.lora_A.data.copy_(state[key_a].to(module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state[key_b].to(module.lora_B.device, dtype=module.lora_B.dtype))
    if missing:
        raise ValueError(f"Missing LoRA weights for modules: {', '.join(missing[:5])}")


def save_lora_adapter(model: nn.Module, output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter_model.pt")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_lora_adapter(model: nn.Module, adapter_dir: Path, quantize_adapter: bool = False) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    state_path = adapter_dir / "adapter_model.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Missing adapter files in {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    apply_lora_to_model(
        model,
        tuple(config["target_modules"]),
        int(config["rank"]),
        float(config["alpha"]),
        float(config.get("dropout", 0.0)),
        quantize_adapter=quantize_adapter,
    )
    state = torch.load(state_path, map_location="cpu")
    load_lora_state(model, state)
    return config
