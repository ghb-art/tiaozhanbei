#!/usr/bin/env python3
"""Freeze the 86 P0-A6 Code rows not used by the quick-100 gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/p0a6/train.jsonl"
QUICK = ROOT / "data/p0a6/quick_validation.jsonl"
FULL = ROOT / "data/p0a6/full_validation.jsonl"
OUTPUT = ROOT / "data/p0a8/code_internal_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a8_code_validation.json"


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BuildError(f"Non-object row in {path.relative_to(ROOT)}")
                rows.append(value)
    return rows


def main() -> int:
    train = read_jsonl(TRAIN)
    quick = read_jsonl(QUICK)
    full = read_jsonl(FULL)
    train_ids = {str(row.get("sample_id")) for row in train}
    quick_code_ids = {
        str(row.get("sample_id")) for row in quick if row.get("domain") == "code"
    }
    full_code = [row for row in full if row.get("domain") == "code"]
    selected: list[dict[str, Any]] = []
    for row in full_code:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in quick_code_ids:
            continue
        if sample_id in train_ids:
            raise BuildError(f"Training/internal-validation overlap: {sample_id}")
        if row.get("dataset_key") != "opencodeinstruct":
            raise BuildError(f"Unexpected Code dataset: {sample_id}")
        tests = row.get("unit_tests")
        if not isinstance(tests, list) or not tests:
            raise BuildError(f"Missing executable tests: {sample_id}")
        copied = dict(row)
        copied["split_role"] = "p0a8_code_internal_validation"
        selected.append(copied)
    selected.sort(key=lambda row: str(row["sample_id"]))
    if len(full_code) != 186 or len(quick_code_ids) != 100 or len(selected) != 86:
        raise BuildError(
            f"Unexpected counts full={len(full_code)} quick={len(quick_code_ids)} "
            f"selected={len(selected)}"
        )
    if len({str(row["sample_id"]) for row in selected}) != 86:
        raise BuildError("Duplicate P0-A8 validation id")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    audit = {
        "gate": "P0-A8-CODE-TRAIN-ONLY-VALIDATION-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a8_code_validation.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "train": "data/p0a6/train.jsonl",
        "train_hash": sha256_file(TRAIN),
        "quick": "data/p0a6/quick_validation.jsonl",
        "quick_hash": sha256_file(QUICK),
        "full": "data/p0a6/full_validation.jsonl",
        "full_hash": sha256_file(FULL),
        "output": "data/p0a8/code_internal_validation.jsonl",
        "output_hash": sha256_file(OUTPUT),
        "full_code_rows": 186,
        "quick_code_rows_excluded": 100,
        "internal_validation_rows": 86,
        "train_validation_overlap": 0,
        "quick_validation_overlap": 0,
        "formal_test_loaded": False,
        "humaneval_loaded": False,
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = AUDIT.with_suffix(AUDIT.suffix + ".tmp")
    audit_tmp.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit_tmp.replace(AUDIT)
    print(f"Wrote data/p0a8/code_internal_validation.jsonl rows={len(selected)}")
    print("Wrote reports/audit/gate_p0a8_code_validation.json status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"P0-A8 Code validation build failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
