from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_utils import load_lora_adapter, merge_lora_modules


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_ADAPTER = ROOT / "models" / "adapters" / "p0a2_deepseek_recovery"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "checkpoints" / "p0a2-deepseek-recovery-merged"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a2_deepseek_merged_hf.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a manual LoRA adapter into a Hugging Face base model.")
    parser.add_argument("--local-model-dir", "--local_model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--adapter-path", "--adapter_path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = resolve_path(args.local_model_dir)
    adapter_dir = resolve_path(args.adapter_path)
    output_dir = resolve_path(args.output_dir)
    audit_path = resolve_path(args.audit)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapter_config = load_lora_adapter(model, adapter_dir)
    merged_count, merged_names = merge_lora_modules(model)
    if merged_count <= 0:
        print("No LoRA modules were merged.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(output_dir)
    export_meta = {
        "created_by": "model_compression/export_merged_hf.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "base_model_dir": display_path(model_dir),
        "adapter_path": display_path(adapter_dir),
        "adapter_config": adapter_config,
        "merged_lora_module_count": merged_count,
        "merged_lora_modules_hash": sha256_text("\n".join(merged_names) + "\n"),
        "dtype": args.dtype,
    }
    write_json(output_dir / "MERGED_ADAPTER.json", export_meta)

    audit = {
        "gate": "G-KD-TRACE-merged-hf-export",
        "check_version": "1.0",
        "created_by": "model_compression/export_merged_hf.py",
        "created_ts": export_meta["created_ts"],
        "status": "passed",
        "base_model_dir": display_path(model_dir),
        "base_model_hash": sha256_dir(model_dir),
        "adapter_path": display_path(adapter_dir),
        "adapter_hash": sha256_dir(adapter_dir),
        "output_dir": display_path(output_dir),
        "output_dir_hash": sha256_dir(output_dir),
        "merged_lora_module_count": merged_count,
        "merged_lora_modules_hash": export_meta["merged_lora_modules_hash"],
        "dtype": args.dtype,
        "safe_serialization": bool(args.safe_serialization),
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_dir)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"output_dir_hash={audit['output_dir_hash']}")
    print("Merged HF export passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
