from __future__ import annotations

import json
import math
from contextlib import contextmanager
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
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled:
            return base_out
        dropped = self.dropout(x).to(torch.float32)
        lora = F.linear(F.linear(dropped, self.lora_A), self.lora_B)
        return base_out + lora.to(base_out.dtype) * self.scaling


class ResidualLoRALinear(nn.Module):
    """Add a trainable LoRA branch while preserving a frozen parent LoRA path."""

    def __init__(
        self,
        parent_layer: LoRALinear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = parent_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.base_layer.dropout = nn.Identity()

        original = parent_layer.base_layer
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled:
            return base_out
        dropped = self.dropout(x).to(torch.float32)
        lora = F.linear(F.linear(dropped, self.lora_A), self.lora_B)
        return base_out + lora.to(base_out.dtype) * self.scaling


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
) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue
        parent, child_name = find_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No linear modules matched LoRA targets: {target_modules}")
    return replaced


def apply_residual_lora_to_model(
    model: nn.Module,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> tuple[list[str], list[str]]:
    """Add a residual branch, retaining existing LoRA modules as frozen parents."""
    replaced: list[str] = []
    parent_backed: list[str] = []
    for name, module in list(model.named_modules()):
        if not any(name.endswith(target) for target in target_modules):
            continue
        parent, child_name = find_parent_module(model, name)
        if isinstance(module, LoRALinear):
            setattr(parent, child_name, ResidualLoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
            parent_backed.append(name)
        elif isinstance(module, nn.Linear):
            setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    if not replaced:
        raise ValueError(f"No linear or LoRA modules matched residual targets: {target_modules}")
    return replaced, parent_backed


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def outer_lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear | ResidualLoRALinear]]:
    selected: list[tuple[str, LoRALinear | ResidualLoRALinear]] = []
    selected_names: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(f"{parent_name}.") for parent_name in selected_names):
            continue
        if isinstance(module, (LoRALinear, ResidualLoRALinear)):
            selected.append((name, module))
            selected_names.append(name)
    return selected


@contextmanager
def disable_outer_lora(model: nn.Module):
    modules = [module for _, module in outer_lora_modules(model)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in outer_lora_modules(model):
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return state


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing: list[str] = []
    for name, module in outer_lora_modules(model):
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a not in state or key_b not in state:
            missing.append(name)
            continue
        module.lora_A.data.copy_(state[key_a].to(module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state[key_b].to(module.lora_B.device, dtype=module.lora_B.dtype))
    if missing:
        raise ValueError(f"Missing LoRA weights for modules: {', '.join(missing[:5])}")


def merge_lora_modules(model: nn.Module) -> tuple[int, list[str]]:
    merged_names: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base_layer
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA base layer is not nn.Linear: {name}")
        delta = (module.lora_B.detach().to(torch.float32) @ module.lora_A.detach().to(torch.float32))
        delta = (delta * module.scaling).to(device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            base.weight.add_(delta)
        parent, child_name = find_parent_module(model, name)
        setattr(parent, child_name, base)
        merged_names.append(name)
    return len(merged_names), merged_names


def load_lora_adapter_files(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    config_path = adapter_dir / "adapter_config.json"
    state_path = adapter_dir / "adapter_model.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Missing adapter files in {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Adapter state is empty or invalid: {adapter_dir}")
    return config, state


def _adapter_module_names(state: dict[str, torch.Tensor]) -> set[str]:
    names = {key[: -len(".lora_A")] for key in state if key.endswith(".lora_A")}
    for name in names:
        if f"{name}.lora_B" not in state:
            raise ValueError(f"Missing LoRA B tensor for module: {name}")
    extra_b = {key[: -len(".lora_B")] for key in state if key.endswith(".lora_B")} - names
    if extra_b:
        raise ValueError(f"Missing LoRA A tensor for module: {sorted(extra_b)[0]}")
    return names


def compose_lora_state_dicts(
    first_state: dict[str, torch.Tensor],
    first_config: dict[str, Any],
    second_state: dict[str, torch.Tensor],
    second_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], int, list[str]]:
    first_rank = int(first_config["rank"])
    second_rank = int(second_config["rank"])
    if first_rank <= 0 or second_rank <= 0:
        raise ValueError("Adapter ranks must be positive")
    first_scale = float(first_config["alpha"]) / first_rank
    second_scale = float(second_config["alpha"]) / second_rank
    first_names = _adapter_module_names(first_state)
    second_names = _adapter_module_names(second_state)
    module_names = sorted(first_names | second_names)
    if not module_names:
        raise ValueError("No LoRA modules found to compose")

    composed: dict[str, torch.Tensor] = {}
    for name in module_names:
        first_a = first_state.get(f"{name}.lora_A")
        first_b = first_state.get(f"{name}.lora_B")
        second_a = second_state.get(f"{name}.lora_A")
        second_b = second_state.get(f"{name}.lora_B")
        reference_a = first_a if first_a is not None else second_a
        reference_b = first_b if first_b is not None else second_b
        if reference_a is None or reference_b is None:
            raise ValueError(f"Incomplete LoRA tensors for module: {name}")
        in_features = int(reference_a.shape[1])
        out_features = int(reference_b.shape[0])

        if first_a is None:
            first_a = torch.zeros(first_rank, in_features, dtype=torch.float32)
            first_b = torch.zeros(out_features, first_rank, dtype=torch.float32)
        if second_a is None:
            second_a = torch.zeros(second_rank, in_features, dtype=torch.float32)
            second_b = torch.zeros(out_features, second_rank, dtype=torch.float32)
        if first_a.shape != (first_rank, in_features) or first_b is None or first_b.shape != (out_features, first_rank):
            raise ValueError(f"Unexpected first adapter shape for module: {name}")
        if second_a.shape != (second_rank, in_features) or second_b is None or second_b.shape != (out_features, second_rank):
            raise ValueError(f"Unexpected second adapter shape for module: {name}")

        composed[f"{name}.lora_A"] = torch.cat(
            [first_a.to(torch.float32), second_a.to(torch.float32)], dim=0
        )
        composed[f"{name}.lora_B"] = torch.cat(
            [first_b.to(torch.float32) * first_scale, second_b.to(torch.float32) * second_scale], dim=1
        )
    return composed, first_rank + second_rank, module_names


def save_lora_adapter(model: nn.Module, output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter_model.pt")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_lora_adapter(model: nn.Module, adapter_dir: Path) -> dict[str, Any]:
    config, state = load_lora_adapter_files(adapter_dir)
    apply_lora_to_model(
        model,
        tuple(config["target_modules"]),
        int(config["rank"]),
        float(config["alpha"]),
        float(config.get("dropout", 0.0)),
    )
    load_lora_state(model, state)
    return config
