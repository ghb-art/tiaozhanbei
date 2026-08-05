#!/usr/bin/env python3
"""Materialize the existing frozen NLP100 without opening formal CMMLU test."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/capability_v2/gate300.jsonl"
OUTPUT = ROOT / "data/p0a31/nlp_gate100.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a31_data.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    try:
        if OUTPUT.exists() or AUDIT.exists():
            raise RuntimeError("P0-A31 outputs already exist; overwrite refused")
        rows = [
            json.loads(line)
            for line in SOURCE.open(encoding="utf-8")
            if line.strip()
        ]
        nlp = [dict(row, split_role="p0a31_frozen_gate") for row in rows if row.get("domain") == "nlp"]
        ids = {str(row.get("sample_id", "")) for row in nlp}
        if len(nlp) != 100 or len(ids) != 100:
            raise RuntimeError("Frozen gate does not contain 100 unique NLP rows")
        if any(row.get("dataset_key") != "cmmlu" for row in nlp):
            raise RuntimeError("P0-A31 contains a non-CMMLU row")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8") as handle:
            for row in nlp:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report = {
            "gate": "P0-A31-NLP100-DATA",
            "created_by": "model_compression/build_p0a31_nlp_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "source": SOURCE.relative_to(ROOT).as_posix(),
            "source_hash": sha256_file(SOURCE),
            "output": OUTPUT.relative_to(ROOT).as_posix(),
            "output_hash": sha256_file(OUTPUT),
            "rows": 100,
            "formal_test_opened": False,
        }
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(ROOT)} rows=100")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A31 data build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
