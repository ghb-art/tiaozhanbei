#!/usr/bin/env python3
"""Build a disjoint second wave of MMLU auxiliary Chinese Teacher requests."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from generate_p0a7_mmlu_chinese import normalized_identity, teacher_messages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/datasets/mmlu/data/auxiliary_train"
HISTORICAL = ROOT / "data/distill/p0a7_nlp_teacher_requests.jsonl"
OUTPUT = ROOT / "data/distill/p0a32_nlp_teacher_requests.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a32_requests.json"
SEED = 20260802
QUOTAS = {
    "arc_easy": (800, 100),
    "arc_hard": (175, 25),
    "aux_law_90s": (440, 60),
    "mc_test": (800, 100),
    "obqa": (1425, 175),
    "race": (1760, 215),
}
CHOICES = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    try:
        if OUTPUT.exists() or AUDIT.exists():
            raise RuntimeError("P0-A32 request outputs already exist; overwrite refused")
        historical_rows = [
            json.loads(line)
            for line in HISTORICAL.open(encoding="utf-8")
            if line.strip()
        ]
        excluded = {str(row["source_question_hash"]) for row in historical_rows}
        if len(excluded) != 4397:
            raise RuntimeError(f"Historical identity count changed: {len(excluded)}")
        selected: list[dict] = []
        available_counts: dict[str, int] = {}
        for domain, (train_count, validation_count) in QUOTAS.items():
            path = SOURCE / f"{domain}.csv"
            candidates: dict[str, dict] = {}
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
                    if identity in excluded or identity in candidates:
                        continue
                    candidates[identity] = {
                        "request_id": f"p0a32/nlp/{domain}/{identity[:20]}",
                        "validation_group_id": f"mmlu-aux/{identity[:20]}",
                        "domain": domain,
                        "source_row": row_number,
                        "expected_label": label,
                        "messages": teacher_messages(question, options),
                        "source_question_hash": identity,
                    }
            available_counts[domain] = len(candidates)
            required = train_count + validation_count
            ordered = sorted(
                candidates.values(),
                key=lambda row: sha256_text(f"{SEED}:{row['request_id']}"),
            )
            if len(ordered) < required:
                raise RuntimeError(f"{domain} has {len(ordered)} unused rows; needs {required}")
            validation = ordered[:validation_count]
            train = ordered[validation_count:required]
            for row in validation:
                row["split_role"] = "new_train_only_validation"
            for row in train:
                row["split_role"] = "train"
            selected.extend(validation)
            selected.extend(train)
        ids = {str(row["source_question_hash"]) for row in selected}
        if len(selected) != 6075 or len(ids) != 6075 or ids & excluded:
            raise RuntimeError("P0-A32 request identity isolation failed")
        selected.sort(key=lambda row: str(row["request_id"]))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_counts = Counter(str(row["split_role"]) for row in selected)
        domain_counts = Counter(str(row["domain"]) for row in selected)
        report = {
            "gate": "P0-A32-MMLU-REQUESTS",
            "created_by": "model_compression/build_p0a32_mmlu_requests.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "source_split": "MMLU auxiliary_train only",
            "historical_request_count": len(excluded),
            "historical_overlap": 0,
            "request_count": len(selected),
            "split_counts": dict(split_counts),
            "domain_counts": dict(domain_counts),
            "available_counts": available_counts,
            "requests": OUTPUT.relative_to(ROOT).as_posix(),
            "requests_hash": sha256_file(OUTPUT),
            "formal_test_opened": False,
        }
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(selected)}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A32 request build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
