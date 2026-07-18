from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_comma_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def infer_model_id(model_dir: Path) -> str:
    name = model_dir.name
    if "--" in name:
        namespace, model_name = name.split("--", 1)
        if namespace and model_name:
            return f"{namespace}/{model_name}"
    return name


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_local_student(
    model_dir: Path,
    adapter_path: Path | None,
    device: str,
    dtype_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from .lora_utils import load_lora_adapter
    except ImportError:
        from lora_utils import load_lora_adapter

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=resolve_torch_dtype(dtype_name),
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapter_config: dict[str, Any] = {}
    if adapter_path is not None:
        adapter_config = load_lora_adapter(model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model, adapter_config
