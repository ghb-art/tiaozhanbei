#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "humaneval", "cmmlu")


class MergeError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MergeError(f"Missing shard trace: {display_path(path)}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MergeError(f"Non-object row {display_path(path)}:{line_number}")
                rows.append(value)
    return rows


def expected_ids(split_dir: Path) -> list[str]:
    ids = []
    for task in TASKS:
        path = split_dir / f"{task}_test.txt"
        if not path.is_file():
            raise MergeError(f"Missing official split: {display_path(path)}")
        ids.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and seal deterministic P0-A4 official-full shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--split-dir", default="data/splits/p0a4_official_full")
    parser.add_argument("--role", choices=("baseline", "student"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--allow-replace", action="store_true", help="Infrastructure recovery only; never use for a new model result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        split_dir = resolve_path(args.split_dir)
        inputs = [resolve_path(value) for value in args.input]
        output = resolve_path(args.output)
        audit_path = resolve_path(args.audit)
        if output.exists() and not args.allow_replace:
            raise MergeError(f"Refusing to replace sealed trace: {display_path(output)}")
        expected = expected_ids(split_dir)
        expected_set = set(expected)
        if len(expected) != len(expected_set):
            raise MergeError("Official-full split contains duplicate IDs")
        indexed: dict[str, dict[str, Any]] = {}
        duplicate_count = 0
        for path in inputs:
            for row in load_jsonl(path):
                sample_id = str(row.get("sample_id", ""))
                if sample_id in indexed:
                    duplicate_count += 1
                else:
                    indexed[sample_id] = row
        missing = sorted(expected_set - set(indexed))
        extra = sorted(set(indexed) - expected_set)
        if duplicate_count or missing or extra:
            raise MergeError(
                f"Shard identity failure duplicate={duplicate_count} missing={len(missing)} extra={len(extra)}"
            )
        rows = [indexed[sample_id] for sample_id in expected]
        counts = Counter(str(row.get("dataset_key", "")) for row in rows)
        correct = Counter(
            str(row.get("dataset_key", "")) for row in rows if row.get("correct") is True
        )
        errors = sum(bool(row.get("generation_error")) for row in rows)
        accuracy = {
            task: correct[task] / counts[task] if counts[task] else 0.0 for task in TASKS
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.chmod(0o440)
        manifest_path = split_dir / "manifest.json"
        report = {
            "gate": f"P0-A4-{args.role.upper()}-OFFICIAL-FULL",
            "check_version": "1.0",
            "created_by": "scripts/p0a4_merge_shards.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if errors == 0 else "failed",
            "role": args.role,
            "model_name": args.model_name,
            "split_manifest": display_path(manifest_path),
            "split_manifest_hash": sha256_file(manifest_path),
            "input_shards": [
                {"path": display_path(path), "sha256": sha256_file(path)} for path in inputs
            ],
            "sealed_trace": display_path(output),
            "sealed_trace_hash": sha256_file(output),
            "sample_count": len(rows),
            "dataset_counts": dict(counts),
            "correct_counts": dict(correct),
            "accuracy_by_dataset": accuracy,
            "macro_accuracy": sum(accuracy.values()) / len(accuracy),
            "generation_error_count": errors,
            "duplicate_sample_count": 0,
            "missing_sample_count": 0,
            "extra_sample_count": 0,
            "item_level_feedback_allowed_for_training": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, MergeError, json.JSONDecodeError) as exc:
        print(f"P0-A4 shard merge failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote and sealed {display_path(output)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"status={report['status']} accuracy={report['accuracy_by_dataset']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
