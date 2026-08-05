#!/usr/bin/env python3
"""Select one P0-A42 route per domain using aggregate train-only accuracy."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "reports/audit/p0a42"
OUTPUT = ROOT / "reports/audit/gate_p0a42_domain_selection.json"

CANDIDATES: dict[str, list[dict[str, str]]] = {
    "math": [
        {"name": "original-base", "audit": "base_math.json", "adapter": ""},
        {"name": "math-108", "audit": "math_108.json", "adapter": "models/checkpoints/p0a42/math/checkpoint-108"},
        {"name": "math-216", "audit": "math_216.json", "adapter": "models/checkpoints/p0a42/math/checkpoint-216"},
    ],
    "code": [
        {"name": "p0a25-code-192", "audit": "current_code.json", "adapter": "models/checkpoints/p0a25/code-failure-repair/checkpoint-192"},
        {"name": "code-96", "audit": "code_96.json", "adapter": "models/checkpoints/p0a42/code/checkpoint-96"},
        {"name": "code-192", "audit": "code_192.json", "adapter": "models/checkpoints/p0a42/code/checkpoint-192"},
        {"name": "original-base", "audit": "base_code.json", "adapter": ""},
    ],
    "nlp": [
        {"name": "p0a10-nlp-136", "audit": "current_nlp.json", "adapter": "models/checkpoints/p0a10/nlp-specialist/checkpoint-136"},
        {"name": "nlp-50", "audit": "nlp_50.json", "adapter": "models/checkpoints/p0a42/nlp/checkpoint-50"},
        {"name": "nlp-100", "audit": "nlp_100.json", "adapter": "models/checkpoints/p0a42/nlp/checkpoint-100"},
        {"name": "original-base", "audit": "base_nlp.json", "adapter": ""},
    ],
}

EXPECTED_GATES = {
    "math": "P0-A42-MATH-TRAIN-ONLY-VALIDATION",
    "code": "P0-A42-CODE-TRAIN-ONLY-VALIDATION",
    "nlp": "P0-A42-NLP-TRAIN-ONLY-VALIDATION",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(domain: str, entry: dict[str, str]) -> dict[str, Any]:
    path = AUDIT_ROOT / entry["audit"]
    if not path.is_file():
        raise RuntimeError(f"Missing P0-A42 audit: {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or value.get("gate") != EXPECTED_GATES[domain]:
        raise RuntimeError(f"Rejected P0-A42 audit: {path.relative_to(ROOT)}")
    if value.get("domain") != domain or int(value.get("generation_error_count", -1)) != 0:
        raise RuntimeError(f"Invalid P0-A42 aggregate: {path.relative_to(ROOT)}")
    adapter = ROOT / entry["adapter"] if entry["adapter"] else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"Missing P0-A42 adapter: {adapter.relative_to(ROOT)}")
    return {
        **entry,
        "audit": path.relative_to(ROOT).as_posix(),
        "audit_hash": sha256_file(path),
        "correct_count": int(value["correct_count"]),
        "sample_count": int(value["sample_count"]),
        "accuracy": float(value["accuracy"]),
        "canonical_format_rate": float(value["canonical_format_rate"]),
    }


def main() -> int:
    selected: dict[str, dict[str, Any]] = {}
    all_candidates: dict[str, list[dict[str, Any]]] = {}
    for domain, entries in CANDIDATES.items():
        values = [load_candidate(domain, entry) for entry in entries]
        sample_counts = {value["sample_count"] for value in values}
        if len(sample_counts) != 1:
            raise RuntimeError(f"P0-A42 {domain} sample counts differ")
        # The declaration order is the tie-break policy: existing stable route,
        # then the half checkpoint, then the full checkpoint, then bare base.
        winner_index, winner = max(
            enumerate(values), key=lambda item: (item[1]["correct_count"], -item[0])
        )
        del winner_index
        selected[domain] = winner
        all_candidates[domain] = values
    report = {
        "gate": "P0-A42-DOMAIN-SELECTION",
        "check_version": "1.0",
        "created_by": "scripts/select_p0a42_domains.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "selection_metric": "aggregate_correct_count_only",
        "tie_break": "stable route, half checkpoint, full checkpoint, bare base",
        "candidates": all_candidates,
        "selected": selected,
        "gate300_loaded": False,
        "formal_full_loaded": False,
        "per_item_feedback_read": False,
    }
    report["report_hash"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print("selected=" + json.dumps({key: value["name"] for key, value in selected.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A42 selection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
