#!/usr/bin/env python3
"""Select one P0-A46 NLP checkpoint from aggregate train-only validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a46_nlp_isolated.json"
REPORT_DIR = ROOT / "reports/audit/p0a44"
OUTPUT = ROOT / "reports/audit/gate_p0a46_hf_selection.json"


def read(name: str) -> dict:
    path = REPORT_DIR / f"p0a46_{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or int(value.get("generation_error_count", -1)) != 0:
        raise RuntimeError(f"Rejected validation report: {path.relative_to(ROOT)}")
    return value


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validation = config["internal_validation"]
    minimum_gain = int(validation["minimum_hf_gain_items"])
    base = {name: read(f"hf_base_{name}") for name in ("ceval", "cmmlu")}
    base_total = sum(int(value["correct_count"]) for value in base.values())
    candidates: list[dict] = []
    for step in (159, 318):
        reports = {name: read(f"hf_{step}_{name}") for name in ("ceval", "cmmlu")}
        total = sum(int(value["correct_count"]) for value in reports.values())
        not_worse = all(
            int(reports[name]["correct_count"]) >= int(base[name]["correct_count"])
            for name in base
        )
        candidates.append(
            {
                "step": step,
                "correct_by_dataset": {
                    name: int(reports[name]["correct_count"]) for name in reports
                },
                "correct_total": total,
                "gain_items": total - base_total,
                "not_worse_each_dataset": not_worse,
                "eligible": bool(not_worse and total - base_total >= minimum_gain),
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["correct_total"], -item["step"])) if eligible else None
    status = "passed" if selected else "failed"
    audit = {
        "gate": "P0-A46-HF-NLP-SELECTION",
        "check_version": "1.0",
        "created_by": "scripts/select_p0a46_nlp.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "base_correct_by_dataset": {
            name: int(value["correct_count"]) for name, value in base.items()
        },
        "base_correct_total": base_total,
        "minimum_gain_items": minimum_gain,
        "candidates": candidates,
        "selected": selected,
        "selected_adapter": (
            f"models/checkpoints/p0a46/nlp/checkpoint-{selected['step']}" if selected else ""
        ),
        "formal_test_items_loaded": 0,
        "policy": "aggregate train-only validation; no gate300 feedback",
    }
    audit["report_hash"] = hashlib.sha256(
        json.dumps(audit, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} status={status} "
        f"base={base_total}/595 selected={selected}"
    )
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
