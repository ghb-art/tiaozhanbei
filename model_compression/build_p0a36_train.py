#!/usr/bin/env python3
"""Build P0-A36 train data from balanced generated MCQs and C-Eval replay."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data/p0a36/nlp_train.jsonl"
REPLAY = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
VALIDATION = ROOT / "data/p0a36/nlp_validation.jsonl"
TEACHER_AUDIT = ROOT / "reports/audit/gate_p0a36_teacher_data.json"
OUTPUT = ROOT / "data/p0a36/train.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a36_train_data.json"


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    teacher = json.loads(TEACHER_AUDIT.read_text(encoding="utf-8"))
    if teacher.get("status") != "passed" or teacher.get("train_rows") != 1024:
        raise BuildError("P0-A36 Teacher audit is not promotable")
    new = read_jsonl(NEW)
    replay = read_jsonl(REPLAY)
    validation = read_jsonl(VALIDATION)
    if len(new) != 1024 or len(replay) != 1335 or len(validation) != 256:
        raise BuildError(
            f"Unexpected inputs: new={len(new)} replay={len(replay)} validation={len(validation)}"
        )
    train: list[dict[str, Any]] = []
    for row in new:
        copied = dict(row)
        copied["training_weight"] = 2.0
        copied["answer_token_weight"] = max(2.0, float(copied.get("answer_token_weight", 1.0)))
        copied["kl_weight"] = 0.10
        train.append(copied)
    for row in replay:
        copied = dict(row)
        copied["domain"] = "nlp"
        copied["task_id"] = "nlp"
        copied["split_role"] = "train"
        copied["training_weight"] = 1.0
        copied["answer_token_weight"] = max(2.0, float(copied.get("answer_token_weight", 1.0)))
        copied["kl_weight"] = 0.20
        train.append(copied)
    train.sort(key=lambda row: str(row["sample_id"]))
    ids = [str(row["sample_id"]) for row in train]
    validation_ids = {str(row["sample_id"]) for row in validation}
    if len(ids) != len(set(ids)):
        raise BuildError("Duplicate P0-A36 training ids")
    if set(ids) & validation_ids:
        raise BuildError("P0-A36 train-validation overlap")
    write_jsonl(OUTPUT, train)
    report = {
        "gate": "P0-A36-BALANCED-MCQ-TRAIN-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a36_train.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "train_rows": len(train),
        "train_dataset_counts": dict(Counter(str(row.get("dataset_key")) for row in train)),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "policy": {
            "initial_adapter": "P0-A10 NLP step136",
            "new_mcq_weight": 2.0,
            "ceval_replay_weight": 1.0,
            "p0a34_validation_reused": False,
            "formal_cmmlu_test_opened": False,
        },
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (NEW, REPLAY, VALIDATION, TEACHER_AUDIT)
        },
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_hash": sha256_file(OUTPUT),
        "errors": [],
    }
    write_json(AUDIT, report)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(train)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A36 train data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
