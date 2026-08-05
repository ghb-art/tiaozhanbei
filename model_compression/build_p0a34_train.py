#!/usr/bin/env python3
"""Build the P0-A34 Chinese-exam continuation corpus."""

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
NEW = ROOT / "data/p0a34/coig_verified_train.jsonl"
CEVAL = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
MMLU = ROOT / "data/p0a7/nlp_mmlu_aux_train.jsonl"
VALIDATION = ROOT / "data/p0a34/nlp_validation.jsonl"
TEACHER_AUDIT = ROOT / "reports/audit/gate_p0a34_teacher_data.json"
OUTPUT = ROOT / "data/p0a34/train.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a34_train_data.json"
SEED = 20260802


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


def replay(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["domain"] = "nlp"
    copied["task_id"] = "nlp"
    copied["split_role"] = "train"
    copied["training_weight"] = 1.0
    copied["answer_token_weight"] = max(2.0, float(copied.get("answer_token_weight", 1.0)))
    copied["kl_weight"] = 0.20 if copied.get("dataset_key") == "ceval_rationale_train" else 0.10
    return copied


def main() -> int:
    teacher = json.loads(TEACHER_AUDIT.read_text(encoding="utf-8"))
    if teacher.get("status") != "passed" or int(teacher.get("selected_count", 0)) != 820:
        raise BuildError("P0-A34 Teacher data audit has not passed with 820 rows")
    new = read_jsonl(NEW)
    ceval = read_jsonl(CEVAL)
    mmlu_all = read_jsonl(MMLU)
    validation = read_jsonl(VALIDATION)
    if (len(new), len(ceval), len(mmlu_all), len(validation)) != (820, 1335, 3000, 260):
        raise BuildError("Unexpected P0-A34 source counts")
    mmlu = sorted(
        mmlu_all,
        key=lambda row: sha256_text(f"{SEED}:{row['sample_id']}"),
    )[:1000]
    train = [dict(row) for row in new]
    train.extend(replay(row) for row in ceval)
    train.extend(replay(row) for row in mmlu)
    train.sort(key=lambda row: sha256_text(f"{SEED}:train:{row['sample_id']}"))
    ids = [str(row["sample_id"]) for row in train]
    validation_ids = {str(row["sample_id"]) for row in validation}
    if len(ids) != len(set(ids)):
        raise BuildError("Duplicate P0-A34 training sample ids")
    if set(ids) & validation_ids:
        raise BuildError("P0-A34 train-validation overlap")
    write_jsonl(OUTPUT, train)
    report = {
        "gate": "P0-A34-CHINESE-EXAM-TRAIN-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a34_train.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "train_rows": len(train),
        "train_dataset_counts": dict(Counter(str(row.get("dataset_key")) for row in train)),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "policy": {
            "p0a32_validation_reused": False,
            "selection_manifest": "C-Eval dev only",
            "formal_cmmlu_test_opened": False,
            "mmlu_replay_selection": "deterministic hash, no evaluation feedback",
        },
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (NEW, CEVAL, MMLU, VALIDATION, TEACHER_AUDIT)
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
        print(f"P0-A34 train data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
