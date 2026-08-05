#!/usr/bin/env python3
"""Select the stronger new P0-A48 checkpoint using aggregate train-only validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/audit/p0a44"
OUT = ROOT / "reports/audit/gate_p0a48_hf_selection.json"


def load(label: str) -> dict:
    path = REPORTS / f"p0a48_{label}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or int(value.get("generation_error_count", -1)) != 0:
        raise RuntimeError(f"Rejected report: {path.relative_to(ROOT)}")
    return value


def main() -> int:
    base = {name: load(f"hf_base_{name}") for name in ("ceval", "cmmlu")}
    base_total = sum(int(value["correct_count"]) for value in base.values())
    candidates = []
    for step in (64, 128):
        values = {name: load(f"hf_{step}_{name}") for name in ("ceval", "cmmlu")}
        total = sum(int(value["correct_count"]) for value in values.values())
        candidates.append({
            "step": step,
            "correct_by_dataset": {name: int(value["correct_count"]) for name, value in values.items()},
            "correct_total": total,
            "gain_items": total - base_total,
        })
    selected = max(candidates, key=lambda item: (item["correct_total"], -item["step"]))
    report = {
        "gate": "P0-A48-HF-NLP-SELECTION",
        "check_version": "1.0",
        "created_by": "scripts/select_p0a48_nlp.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "base_correct_by_dataset": {name: int(value["correct_count"]) for name, value in base.items()},
        "base_correct_total": base_total,
        "candidates": candidates,
        "selected": selected,
        "selected_adapter": f"models/checkpoints/p0a48/nlp/checkpoint-{selected['step']}",
        "mandatory_second_full_authorized": True,
        "formal_test_items_loaded": 0,
    }
    report["report_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} selected={selected} base={base_total}/595")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
