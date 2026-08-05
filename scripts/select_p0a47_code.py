#!/usr/bin/env python3
"""Select the stronger newly trained P0-A47 checkpoint; full evaluation is mandatory."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/audit/p0a44"
OUT = ROOT / "reports/audit/gate_p0a47_hf_selection.json"


def load(label: str) -> dict:
    path = REPORTS / f"p0a47_{label}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or int(value.get("generation_error_count", -1)) != 0:
        raise RuntimeError(f"Rejected report: {path.relative_to(ROOT)}")
    return value


def main() -> int:
    initial = load("hf_initial")
    candidates = []
    for step in (128, 256):
        report = load(f"hf_{step}")
        candidates.append({"step": step, "correct_count": int(report["correct_count"]), "accuracy": float(report["accuracy"])})
    selected = max(candidates, key=lambda item: (item["correct_count"], -item["step"]))
    gain = selected["correct_count"] - int(initial["correct_count"])
    report = {
        "gate": "P0-A47-HF-CODE-SELECTION", "check_version": "1.0", "created_by": "scripts/select_p0a47_code.py",
        "created_ts": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "initial_correct_count": int(initial["correct_count"]), "candidates": candidates,
        "selected": selected, "selected_adapter": f"models/checkpoints/p0a47/code/checkpoint-{selected['step']}",
        "gain_over_initial_items": gain, "preferred_gain_items": 20,
        "preferred_gain_met": gain >= 20, "mandatory_full_authorized": True,
        "formal_test_items_loaded": 0,
    }
    report["report_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True); temporary = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} selected={selected} gain={gain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
