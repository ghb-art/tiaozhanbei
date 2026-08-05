#!/usr/bin/env python3
"""Build a fresh, label-locked Chinese-MCQ translation request pool for P0-B1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_p0a7_mmlu_chinese import normalized_identity, teacher_messages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/datasets/mmlu/data/auxiliary_train"
HISTORICAL = (
    ROOT / "data/distill/p0a7_nlp_teacher_requests.jsonl",
    ROOT / "data/distill/p0a32_nlp_teacher_requests.jsonl",
)
OUTPUT = ROOT / "data/distill/p0b1_nlp_teacher_requests.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0b1_nlp_requests.json"
SEED = 20260804
TRAIN_CANDIDATES = {"obqa": 2500, "race": 27100}
VALIDATION_CANDIDATES = {"obqa": 300, "race": 2100}
CHOICES = tuple("ABCD")


class BuildError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def historical_hashes() -> tuple[set[str], dict[str, str]]:
    used: set[str] = set()
    hashes: dict[str, str] = {}
    for path in HISTORICAL:
        if not path.is_file():
            raise BuildError(f"Missing historical request manifest: {path.relative_to(ROOT)}")
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    used.add(str(json.loads(line)["source_question_hash"]))
    return used, hashes


def source_rows(domain: str, excluded: set[str]) -> list[dict[str, Any]]:
    path = SOURCE / f"{domain}.csv"
    if not path.is_file():
        raise BuildError(f"Missing MMLU auxiliary source: {path.relative_to(ROOT)}")
    unique: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), 1):
            if len(row) < 6:
                continue
            question = row[0].strip()
            options = [value.strip() for value in row[1:5]]
            label = row[5].strip().upper()
            if not question or any(not value for value in options) or label not in CHOICES:
                continue
            identity = normalized_identity(question, options)
            if identity in excluded or identity in unique:
                continue
            unique[identity] = {
                "request_id": f"p0b1/nlp/{domain}/{identity[:20]}",
                "validation_group_id": f"mmlu-aux/{identity[:20]}",
                "domain": domain,
                "source_row": row_number,
                "expected_label": label,
                "messages": teacher_messages(question, options),
                "source_question_hash": identity,
            }
    return sorted(
        unique.values(),
        key=lambda row: sha256_text(f"{SEED}:{domain}:{row['request_id']}"),
    )


def main() -> int:
    excluded, input_hashes = historical_hashes()
    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    for domain in sorted(set(TRAIN_CANDIDATES) | set(VALIDATION_CANDIDATES)):
        candidates = source_rows(domain, excluded)
        available[domain] = len(candidates)
        validation_count = VALIDATION_CANDIDATES[domain]
        train_count = TRAIN_CANDIDATES[domain]
        if len(candidates) < validation_count + train_count:
            raise BuildError(
                f"{domain} has {len(candidates)} fresh rows; "
                f"requires {validation_count + train_count}"
            )
        validation = candidates[:validation_count]
        train = candidates[validation_count : validation_count + train_count]
        for row in validation:
            row["split_role"] = "new_train_only_validation"
        for row in train:
            row["split_role"] = "train"
        selected.extend(validation)
        selected.extend(train)
        split_counts.update(row["split_role"] for row in validation + train)
        domain_counts.update(row["domain"] for row in validation + train)
    selected.sort(key=lambda row: str(row["request_id"]))
    identities = [str(row["source_question_hash"]) for row in selected]
    if len(identities) != len(set(identities)) or set(identities) & excluded:
        raise BuildError("P0-B1 request identity isolation failed")
    atomic_jsonl(OUTPUT, selected)
    report = {
        "gate": "P0-B1-FRESH-NLP-REQUESTS",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0b1_nlp_requests.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source_split": "MMLU auxiliary_train only",
        "historical_request_count": len(excluded),
        "historical_overlap": 0,
        "request_count": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "available_counts": available,
        "policy": {
            "source_label_sent_to_teacher": False,
            "teacher_must_independently_match_train_label": True,
            "formal_cmmlu_test_opened": False,
            "selection_basis": "Chinese four-choice aggregate task shape only",
        },
        "inputs": input_hashes,
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_hash": sha256_file(OUTPUT),
        "errors": [],
    }
    report["report_hash"] = sha256_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True)
    )
    atomic_json(AUDIT, report)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(selected)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B1 NLP request build failed: {exc}")
        raise SystemExit(1)
