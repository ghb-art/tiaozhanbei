#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the shared P0-A4 Student LoRA into its dense base.")
    parser.add_argument("--base-model", default="models/pretrained/Qwen--Qwen3-1.7B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="models/checkpoints/p0a4/student-shared-merged")
    parser.add_argument("--audit", default="reports/audit/gate_p0a4_student_shared_merge.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Missing P0-A4 training dependency: {exc}", file=sys.stderr)
        return 1
    base = resolve_path(args.base_model)
    adapter = resolve_path(args.adapter)
    output = resolve_path(args.output)
    audit = resolve_path(args.audit)
    if not base.is_dir() or not adapter.is_dir():
        print("Base model or adapter directory is missing.", file=sys.stderr)
        return 1
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    report = {
        "gate": "P0-A4-STUDENT-SHARED-MERGE",
        "check_version": "1.0",
        "created_by": "model_compression/merge_p0a4_adapter.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "base_model": base.relative_to(ROOT).as_posix(),
        "base_hash": hash_directory(base),
        "adapter": adapter.relative_to(ROOT).as_posix(),
        "adapter_hash": hash_directory(adapter),
        "output": output.relative_to(ROOT).as_posix(),
        "output_hash": hash_directory(output),
    }
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged Student written to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
