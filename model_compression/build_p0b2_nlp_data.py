#!/usr/bin/env python3
"""P0-B2 NLP staged-data builder.

Splits the P0-B1 audited 57,173-row shared pool into two train-only stages:
- Stage 1 (NLP): 30,000 Chinese four-choice rows (CMMLU-v15 contract).
- Stage 2 (Math replay): 7,173 GSM8K rows.

Neither stage reads formal items; rows keep their audited per-row
training_weight / answer_token_weight / kl_weight.
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/p0b1/train.jsonl")
    parser.add_argument("--nlp-output", default="data/p0b2/nlp_stage1.jsonl")
    parser.add_argument("--math-output", default="data/p0b2/math_stage2.jsonl")
    parser.add_argument("--audit", default="reports/audit/gate_p0b2_nlp_data.json")
    args = parser.parse_args()

    try:
        source_path = ROOT / args.source
        rows: list[dict[str, Any]] = []
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        nlp_rows = [row for row in rows if row.get("domain") == "nlp"]
        math_rows = [row for row in rows if row.get("domain") == "math"]
        if len(nlp_rows) != 30000 or len(math_rows) != 7173:
            raise ValueError(
                f"Unexpected staged sizes: nlp={len(nlp_rows)} math={len(math_rows)}"
            )
        for row in nlp_rows:
            if not row.get("messages") or not row.get("answer"):
                raise ValueError(f"NLP row missing content: {row.get('sample_id')}")
        for row in math_rows:
            if not row.get("messages") or not row.get("answer"):
                raise ValueError(f"Math row missing content: {row.get('sample_id')}")
        formal_hits = {
            row["sample_id"]
            for row in [*nlp_rows, *math_rows]
            if any(token in str(row["sample_id"]).lower() for token in ("cmmlu-test", "humaneval"))
        }
        if formal_hits:
            raise ValueError(f"Formal references present: {sorted(formal_hits)[:5]}")

        nlp_path, math_path = ROOT / args.nlp_output, ROOT / args.math_output
        write_jsonl(nlp_path, nlp_rows)
        write_jsonl(math_path, math_rows)

        audit = {
            "gate": "P0-B2-NLP-STAGED-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0b2_nlp_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "source": str(source_path),
            "source_rows": len(rows),
            "source_domains": dict(Counter(str(row.get("domain")) for row in rows)),
            "stage1_nlp_rows": len(nlp_rows),
            "stage1_nlp_unique": len({str(row["sample_id"]) for row in nlp_rows}),
            "stage2_math_rows": len(math_rows),
            "stage2_math_unique": len({str(row["sample_id"]) for row in math_rows}),
            "stage_overlap": len(
                {str(row["sample_id"]) for row in nlp_rows}
                & {str(row["sample_id"]) for row in math_rows}
            ),
            "formal_set_references": 0,
            "outputs": {
                "stage1_nlp": {"path": str(nlp_path), "sha256": sha256_file(nlp_path)},
                "stage2_math": {"path": str(math_path), "sha256": sha256_file(math_path)},
            },
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path = ROOT / args.audit
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {audit_path}")
        print(
            f"P0-B2 NLP staged data: stage1_nlp={len(nlp_rows)} "
            f"stage2_math={len(math_rows)}"
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B2 NLP staged data build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
