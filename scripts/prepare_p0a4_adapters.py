#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "humaneval", "cmmlu")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert P0-A4 task LoRAs to GGUF and freeze Top-1 routing.")
    parser.add_argument("--adapter", action="append", required=True, help="task=PEFT_adapter_directory")
    parser.add_argument("--llama-cpp-dir", default="external/llama.cpp")
    parser.add_argument("--base", default="models/checkpoints/p0a4/student-shared-merged")
    parser.add_argument("--base-audit", default="reports/audit/gate_p0a4_student_shared_merge.json")
    parser.add_argument("--output-dir", default="models/adapters/p0a4")
    parser.add_argument("--manifest", default="models/adapters/p0a4/router_manifest.json")
    parser.add_argument("--audit", default="reports/audit/gate_p0a4_adapter_router_prepare.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = {}
    for item in args.adapter:
        if "=" not in item:
            print(f"Invalid --adapter value: {item}", file=sys.stderr)
            return 2
        task, path = item.split("=", 1)
        if task not in TASKS or task in mapping:
            print(f"Invalid or duplicate task adapter: {task}", file=sys.stderr)
            return 2
        mapping[task] = resolve_path(path)
    if set(mapping) != set(TASKS):
        print(f"All task adapters are required: {TASKS}", file=sys.stderr)
        return 2
    converter = resolve_path(args.llama_cpp_dir) / "convert_lora_to_gguf.py"
    base = resolve_path(args.base)
    base_audit_path = resolve_path(args.base_audit)
    if not converter.is_file():
        print(f"Missing llama.cpp LoRA converter: {converter}", file=sys.stderr)
        return 1
    if not base.is_dir():
        print(f"Missing merged Student config for adapter conversion: {base}", file=sys.stderr)
        return 1
    if not base_audit_path.is_file():
        print(f"Missing merged Student audit: {base_audit_path}", file=sys.stderr)
        return 1
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    if base_audit.get("status") != "passed" or resolve_path(str(base_audit.get("output", ""))) != base:
        print("Merged Student audit does not match the Adapter base", file=sys.stderr)
        return 1
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    adapters = {}
    for adapter_id, task in enumerate(TASKS):
        source = mapping[task]
        if not source.is_dir():
            print(f"Missing adapter: {source}", file=sys.stderr)
            return 1
        output = output_dir / f"{task}-rank-adapter-f16.gguf"
        command = [
            sys.executable,
            str(converter),
            str(source),
            "--base",
            str(base),
            "--outfile",
            str(output),
            "--outtype",
            "f16",
        ]
        commands.append(command)
        if not args.dry_run:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0 or not output.is_file():
                print(f"Adapter conversion failed: {task}", file=sys.stderr)
                return 1
        adapters[task] = {
            "id": adapter_id,
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output) if output.is_file() else "",
            "request_lora": [
                {"id": index, "scale": 1.0 if index == adapter_id else 0.0}
                for index in range(len(TASKS))
            ],
        }
    manifest = {
        "router_version": "p0a4-top1-adapter-1.0",
        "created_by": "scripts/prepare_p0a4_adapters.py",
        "routing": "explicit dataset_key; exactly one adapter has scale=1",
        "base_model": base.relative_to(ROOT).as_posix(),
        "base_model_hash": base_audit.get("output_hash", ""),
        "base_model_audit": base_audit_path.relative_to(ROOT).as_posix(),
        "base_model_audit_hash": sha256_file(base_audit_path),
        "unknown_task": {"route": "shared", "request_lora": []},
        "task_adapters": adapters,
        "server_flags": ["--lora", "<comma-separated adapters>", "--lora-init-without-apply"],
    }
    manifest_path = resolve_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit = {
        "gate": "P0-A4-ADAPTER-ROUTER-PREPARE",
        "check_version": "1.0",
        "created_by": "scripts/prepare_p0a4_adapters.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run_passed" if args.dry_run else "passed",
        "top1": True,
        "unknown_task_fallback": "shared",
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_hash": sha256_file(manifest_path),
        "base_model": base.relative_to(ROOT).as_posix(),
        "base_model_hash": base_audit.get("output_hash", ""),
        "base_model_audit": base_audit_path.relative_to(ROOT).as_posix(),
        "base_model_audit_hash": sha256_file(base_audit_path),
        "commands": commands,
    }
    audit_path = resolve_path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
