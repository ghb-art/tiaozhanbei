#!/usr/bin/env python3
"""Build one leakage-free P0-A42 training round from frozen train-only assets."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/p0a42"
AUDIT = ROOT / "reports/audit/gate_p0a42_data.json"
CONFIG = ROOT / "configs/p0a42_one_round.json"
P0A10 = ROOT / "data/p0a10/train.jsonl"
CODE = ROOT / "data/p0a25/code_train_pool.jsonl"
NLP = ROOT / "data/p0a34/train.jsonl"
VALIDATION = {
    "math": ROOT / "data/p0a10/math_validation.jsonl",
    "code": ROOT / "data/p0a25/code_validation.jsonl",
    "nlp": ROOT / "data/p0a34/nlp_validation.jsonl",
}


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
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise BuildError(f"Empty input: {path.relative_to(ROOT)}")
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def source_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_sample = str(metadata.get("source_sample_id") or "")
    if source_sample:
        return source_sample
    raw = str(row.get("sample_id", ""))
    if row.get("dataset_key") == "opencodeinstruct":
        return raw.rsplit("/", 1)[-1]
    return raw


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("protocol") != "P0-A42-ONE-ROUND-THREE-DOMAIN":
        raise BuildError("P0-A42 config identity changed")

    p0a10 = read_jsonl(P0A10)
    math_rows = [dict(row) for row in p0a10 if row.get("domain") == "math"]
    code_rows = [dict(row) for row in read_jsonl(CODE)]
    nlp_rows = [dict(row) for row in read_jsonl(NLP)]
    expected = {key: int(value["rows"]) for key, value in cfg["training"].items()}
    actual = {"math": len(math_rows), "code": len(code_rows), "nlp": len(nlp_rows)}
    if actual != expected:
        raise BuildError(f"Unexpected training counts: {actual} != {expected}")

    math_format_repairs = 0
    for row in math_rows:
        if "####" not in str(row.get("answer", "")):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            reference = str(metadata.get("reference_answer") or "").strip()
            if not reference:
                raise BuildError(f"Math row has no locked answer: {row.get('sample_id')}")
            row["answer"] = str(row["answer"]).rstrip() + f"\n#### {reference}"
            math_format_repairs += 1
        row["answer_token_weight"] = 4.0
        row["answer_token_position"] = "last"
        row["kl_weight"] = 0.18
        row["training_weight"] = 1.0
        row["p0a42_role"] = "math_answer_weighted"
    for row in code_rows:
        row["answer_token_weight"] = 1.0
        row["kl_weight"] = 0.10
        row["training_weight"] = 1.0
        row["p0a42_role"] = "code_execution_verified"
    for row in nlp_rows:
        row["answer_token_weight"] = 2.0
        row["answer_token_position"] = "last"
        row["training_weight"] = 1.0
        row["kl_weight"] = 0.05 if row.get("dataset_key") == "mmlu_aux_chinese" else 0.10
        row["p0a42_role"] = "nlp_native_mcq_majority"

    train = {"math": math_rows, "code": code_rows, "nlp": nlp_rows}
    overlaps: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for domain, path in VALIDATION.items():
        validation_rows = read_jsonl(path)
        validation_counts[domain] = len(validation_rows)
        train_ids = {source_identity(row) for row in train[domain]}
        validation_ids = {source_identity(row) for row in validation_rows}
        overlaps[domain] = len(train_ids & validation_ids)
    if any(overlaps.values()):
        raise BuildError(f"Training-validation overlap: {overlaps}")

    OUT.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for domain, rows in train.items():
        path = OUT / f"{domain}_train.jsonl"
        atomic_jsonl(path, rows)
        outputs[domain] = {
            "path": path.relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(path),
            "dataset_counts": dict(sorted(Counter(str(row.get("dataset_key", "")) for row in rows).items())),
        }

    audit = {
        "gate": "P0-A42-TRAIN-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a42_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "base_model": cfg["base_model"],
        "outputs": outputs,
        "validation_counts": validation_counts,
        "train_validation_overlap": overlaps,
        "formal_or_gate_rows_loaded": 0,
        "nlp_policy": {
            "native_chinese_rows": sum(1 for row in nlp_rows if row.get("dataset_key") != "mmlu_aux_chinese"),
            "translated_mmlu_rows": sum(1 for row in nlp_rows if row.get("dataset_key") == "mmlu_aux_chinese"),
            "openqa_to_synthetic_mcq_rows": 0,
            "label_source": "human label; teacher rationale only",
        },
        "math_format_repairs_from_locked_train_label": math_format_repairs,
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (CONFIG, P0A10, CODE, NLP, *VALIDATION.values())
        },
    }
    audit["report_hash"] = hashlib.sha256(
        json.dumps(audit, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT.with_name(f".{AUDIT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(AUDIT)
    print(f"Wrote {AUDIT.relative_to(ROOT)}")
    print(f"status=passed counts={actual} overlap={overlaps}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A42 data build failed: {exc}")
        raise SystemExit(1)
