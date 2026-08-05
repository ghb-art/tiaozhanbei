#!/usr/bin/env python3
"""P0-B2 Code GRPO data builder.

Deterministically splits the P0-B1 execution-verified OpenCodeInstruct pool
(20,000 rows, HumanEval-v15 body-only contract, 10-unit-test execution
verified) into 16,000 GRPO train prompts, 1,000 greedy-execution validation
prompts and 3,000 untouched holdout prompts.

No HumanEval items, formal split items or gate300 items are allowed in any of
the three sets. The audit JSON records counts, pairwise overlap and hashes.
"""

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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def deterministic_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: sha256_text(f"{seed}:{row['sample_id']}"),
    )


def load_gate_code_ids() -> set[str]:
    path = ROOT / "data/capability_v2/gate300.jsonl"
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for row in read_jsonl(path):
        if row.get("domain") == "code":
            ids.add(str(row["sample_id"]))
    return ids


def load_internal_validation_ids() -> set[str]:
    path = ROOT / "data/p0b1/internal_validation.jsonl"
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for row in read_jsonl(path):
        ids.add(str(row["sample_id"]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/p0b2_code_grpo.json"))
    parser.add_argument("--audit", default=str(ROOT / "reports/audit/gate_p0b2_code_data.json"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]
    artifacts = config["artifacts"]
    seed = int(config["seed"])
    try:
        source = read_jsonl(ROOT / data_cfg["source"])
        if not source:
            raise ValueError("Empty source pool")
        domains = Counter(str(row.get("domain")) for row in source)
        if domains != Counter({"code": len(source)}):
            raise ValueError(f"Source must be code-only: {domains}")

        ordered = deterministic_order(source, seed)
        train_rows = int(data_cfg["train_rows"])
        validation_rows = int(data_cfg["validation_rows"])
        if train_rows + validation_rows + int(data_cfg["holdout_rows"]) > len(ordered):
            raise ValueError(
                f"Split sizes exceed pool {len(ordered)}: "
                f"{train_rows}+{validation_rows}+{data_cfg['holdout_rows']}"
            )
        train = ordered[:train_rows]
        validation = ordered[train_rows:train_rows + validation_rows]
        holdout = ordered[
            train_rows + validation_rows:
            train_rows + validation_rows + int(data_cfg["holdout_rows"])
        ]

        def ids(rows: list[dict[str, Any]]) -> set[str]:
            return {str(row["sample_id"]) for row in rows}

        train_ids, validation_ids, holdout_ids = ids(train), ids(validation), ids(holdout)
        if train_ids & validation_ids or train_ids & holdout_ids or validation_ids & holdout_ids:
            raise ValueError("Split overlap detected")
        forbidden = load_gate_code_ids() | load_internal_validation_ids()
        overlap = (train_ids | validation_ids | holdout_ids) & forbidden
        if overlap:
            raise ValueError(f"Overlap with gate/internal-validation: {sorted(overlap)[:5]}")
        formal_hits = {
            sample
            for sample in train_ids | validation_ids | holdout_ids
            if any(token in sample.lower() for token in data_cfg["forbidden_formal"])
        }
        if formal_hits:
            raise ValueError(f"Formal-set sample ids present: {sorted(formal_hits)[:5]}")

        train_path = ROOT / artifacts["train_data"]
        validation_path = ROOT / artifacts["validation_data"]
        holdout_path = ROOT / artifacts["holdout_data"]
        write_jsonl(train_path, train)
        write_jsonl(validation_path, validation)
        write_jsonl(holdout_path, holdout)

        created_ts = datetime.now(timezone.utc).isoformat()
        audit = {
            "gate": "P0-B2-CODE-GRPO-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0b2_code_data.py",
            "created_ts": created_ts,
            "status": "passed",
            "config": args.config,
            "config_hash": sha256_text(Path(args.config).read_text(encoding="utf-8")),
            "source": str(data_cfg["source"]),
            "source_rows": len(source),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "holdout_rows": len(holdout),
            "train_unique": len(train_ids),
            "validation_unique": len(validation_ids),
            "holdout_unique": len(holdout_ids),
            "train_validation_overlap": 0,
            "train_holdout_overlap": 0,
            "validation_holdout_overlap": 0,
            "gate_internal_overlap": 0,
            "formal_set_references": 0,
            "output_contract": data_cfg["output_contract"],
            "outputs": {
                "train": {
                    "path": str(artifacts["train_data"]),
                    "sha256": sha256_file(train_path),
                },
                "validation": {
                    "path": str(artifacts["validation_data"]),
                    "sha256": sha256_file(validation_path),
                },
                "holdout": {
                    "path": str(artifacts["holdout_data"]),
                    "sha256": sha256_file(holdout_path),
                },
            },
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path = ROOT / args.audit
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {audit_path}")
        print(
            f"P0-B2 code GRPO data: train={len(train)} validation={len(validation)} "
            f"holdout={len(holdout)}"
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B2 code data build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
