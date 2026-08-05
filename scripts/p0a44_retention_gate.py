#!/usr/bin/env python3
"""Compute the final, aggregate-only P0-A44 formal retention decision."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a44_aligned_retrain.json"
BASELINE = ROOT / "reports/sealed/p0a4/baseline14b_awq_full.jsonl"
CANDIDATE = ROOT / "reports/sealed/p0a44/edge_aligned_router_q4_full.jsonl"
OUTPUT = ROOT / "reports/audit/gate_p0a44_edge_aligned_router_q4_full_retention.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    if OUTPUT.exists(): raise RuntimeError("P0-A44 retention already exists")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))["formal_full"]
    expected={key:int(value) for key,value in cfg["datasets"].items()}
    base_rows,candidate_rows=load(BASELINE),load(CANDIDATE)
    base={str(row["sample_id"]):row for row in base_rows}; candidate={str(row["sample_id"]):row for row in candidate_rows}
    if len(base)!=len(base_rows) or len(candidate)!=len(candidate_rows): raise RuntimeError("Duplicate formal ids")
    missing=sorted(set(base)-set(candidate)); extra=sorted(set(candidate)-set(base))
    mismatch=sum(base[key].get("prompt_hash")!=candidate[key].get("prompt_hash") for key in set(base)&set(candidate))
    base_counts=Counter(str(row["dataset_key"]) for row in base_rows); candidate_counts=Counter(str(row["dataset_key"]) for row in candidate_rows)
    if dict(base_counts)!=expected or dict(candidate_counts)!=expected: raise RuntimeError("Formal counts changed")
    base_correct=Counter(str(row["dataset_key"]) for row in base_rows if row.get("correct") is True)
    candidate_correct=Counter(str(row["dataset_key"]) for row in candidate_rows if row.get("correct") is True)
    base_accuracy={task:base_correct[task]/base_counts[task] for task in TASKS}
    candidate_accuracy={task:candidate_correct[task]/candidate_counts[task] for task in TASKS}
    ratios={task:min(candidate_accuracy[task]/base_accuracy[task],1.0) for task in TASKS}
    macro=sum(ratios.values())/len(ratios); errors=sum(bool(row.get("generation_error")) for row in candidate_rows)
    passed=not missing and not extra and mismatch==0 and errors==0 and all(value>=float(cfg["minimum_retention_per_domain"]) for value in ratios.values()) and macro>=float(cfg["minimum_capped_macro_retention"])
    report={
        "gate":"P0-A44-EDGE-ALIGNED-OFFICIAL-FULL-RETENTION","check_version":"1.0","created_by":"scripts/p0a44_retention_gate.py",
        "created_ts":datetime.now(timezone.utc).isoformat(),"status":"passed" if passed else "failed",
        "decision":"meets_full_retention_requirement" if passed else "does_not_meet_full_retention_requirement",
        "feedback_policy":"aggregate_only_no_retraining","baseline_trace_hash":sha(BASELINE),"candidate_trace_hash":sha(CANDIDATE),
        "expected_counts":expected,"matched_sample_ids":not missing and not extra,"missing_sample_count":len(missing),"extra_sample_count":len(extra),
        "prompt_mismatch_count":mismatch,"baseline_correct_counts":dict(base_correct),"candidate_correct_counts":dict(candidate_correct),
        "baseline_accuracy_by_dataset":base_accuracy,"candidate_accuracy_by_dataset":candidate_accuracy,"retention_ratios":ratios,
        "capped_macro_ratio":macro,"generation_error_count":errors,"formal_full_completed":True,"item_level_feedback_allowed_for_training":False,
    }
    report["report_hash"]=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); temporary=OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");temporary.replace(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} status={report['status']} ratios={ratios} macro={macro:.6f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
