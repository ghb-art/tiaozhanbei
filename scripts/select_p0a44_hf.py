#!/usr/bin/env python3
"""Select P0-A44 adapters using aggregate train-only validation reports."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/audit/gate_p0a44_hf_selection.json"
REPORT = ROOT / "reports/audit/p0a44"


def read(name: str) -> dict:
    path = REPORT / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or value.get("generation_error_count") != 0:
        raise RuntimeError(f"Rejected report: {path}")
    return value


def main() -> int:
    code_base = read("hf_code_base")
    nlp_base = {name: read(f"hf_nlp_base_{name}") for name in ("ceval", "cmmlu")}
    code_candidates = []
    for step in (176, 352):
        report = read(f"hf_code_{step}")
        code_candidates.append((int(report["correct_count"]), step, report))
    code_candidates.sort(reverse=True)
    best_code = code_candidates[0]
    code_selected = best_code[1] if best_code[0] >= int(code_base["correct_count"]) else 0
    nlp_candidates = []
    for step in (64, 128):
        reports = {name: read(f"hf_nlp_{step}_{name}") for name in ("ceval", "cmmlu")}
        total = sum(int(value["correct_count"]) for value in reports.values())
        eligible = all(int(reports[name]["correct_count"]) >= int(nlp_base[name]["correct_count"]) for name in reports)
        nlp_candidates.append((eligible, total, step, reports))
    nlp_candidates.sort(reverse=True)
    best_nlp = nlp_candidates[0]
    nlp_selected = best_nlp[2] if best_nlp[0] else 0
    selected = {
        "math": {"adapter": "", "reason": "formal Math frozen"},
        "code": {"step": code_selected, "adapter": f"models/checkpoints/p0a44/code/checkpoint-{code_selected}" if code_selected else "", "base_correct": int(code_base["correct_count"]), "candidate_correct": best_code[0]},
        "nlp": {"step": nlp_selected, "adapter": f"models/checkpoints/p0a44/nlp/checkpoint-{nlp_selected}" if nlp_selected else "", "base_correct": sum(int(v["correct_count"]) for v in nlp_base.values()), "candidate_correct": best_nlp[1], "candidate_not_worse_each_validation": bool(best_nlp[0])},
    }
    audit = {
        "gate": "P0-A44-HF-SELECTION", "check_version": "1.0", "created_by": "scripts/select_p0a44_hf.py",
        "created_ts": datetime.now(timezone.utc).isoformat(), "status": "passed", "selected": selected,
        "formal_test_items_loaded": 0, "policy": "aggregate train-only validation only; base fallback is allowed",
    }
    audit["report_hash"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} status=passed selected={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
