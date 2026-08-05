#!/usr/bin/env python3
"""Freeze final P0-A44 Q4 routes from aggregate deployment validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/audit/p0a44"
HF = ROOT / "reports/audit/gate_p0a44_hf_selection.json"
OUT = ROOT / "reports/audit/gate_p0a44_q4_selection.json"


def read(name: str) -> dict:
    path = REPORT / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or value.get("generation_error_count") != 0:
        raise RuntimeError(f"Rejected report: {path}")
    return value


def main() -> int:
    hf = json.loads(HF.read_text(encoding="utf-8"))
    if hf.get("status") != "passed":
        raise RuntimeError("HF selection not passed")
    code_step = int(hf["selected"]["code"]["step"])
    nlp_step = int(hf["selected"]["nlp"]["step"])
    code_base = read("q4_code_base")
    code_candidate = read("q4_code_candidate")
    code_ok = code_step > 0 and int(code_candidate["correct_count"]) >= int(code_base["correct_count"])
    nlp_base = {name: read(f"q4_nlp_base_{name}") for name in ("ceval", "cmmlu")}
    nlp_candidate = {name: read(f"q4_nlp_candidate_{name}") for name in ("ceval", "cmmlu")}
    nlp_ok = nlp_step > 0 and all(
        int(nlp_candidate[name]["correct_count"]) >= int(nlp_base[name]["correct_count"])
        for name in nlp_base
    )
    selected = {
        "math": {"step": 0, "adapter": ""},
        "code": {
            "step": code_step if code_ok else 0,
            "adapter": f"models/adapters/p0a44/code-step-{code_step}-f16.gguf" if code_ok else "",
            "base_correct": int(code_base["correct_count"]),
            "candidate_correct": int(code_candidate["correct_count"]),
        },
        "nlp": {
            "step": nlp_step if nlp_ok else 0,
            "adapter": f"models/adapters/p0a44/nlp-step-{nlp_step}-f16.gguf" if nlp_ok else "",
            "base_correct": sum(int(value["correct_count"]) for value in nlp_base.values()),
            "candidate_correct": sum(int(value["correct_count"]) for value in nlp_candidate.values()),
            "not_worse_each_validation": nlp_ok,
        },
    }
    audit = {
        "gate": "P0-A44-Q4-SELECTION", "check_version": "1.0",
        "created_by": "scripts/select_p0a44_q4.py", "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "selected": selected, "formal_test_items_loaded": 0,
        "deployment": {"weight_type": "Q4_K_M", "kv_cache": "q8_0", "thinking": "off"},
    }
    audit["report_hash"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} status=passed selected={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
