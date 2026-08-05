#!/usr/bin/env python3
"""Generic aggregate-only full capability retention gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "humaneval", "cmmlu")
EXPECTED = {"gsm8k": 1319, "humaneval": 164, "cmmlu": 11582}


def resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="reports/sealed/p0a4/baseline14b_awq_full.jsonl")
    parser.add_argument("--minimum", type=float, default=0.8)
    args = parser.parse_args()
    baseline, candidate, output = resolved(args.baseline), resolved(args.candidate), resolved(args.output)
    if output.exists():
        raise RuntimeError(f"Retention output already exists: {output.relative_to(ROOT)}")
    base_rows, candidate_rows = load(baseline), load(candidate)
    base = {str(row["sample_id"]): row for row in base_rows}; student = {str(row["sample_id"]): row for row in candidate_rows}
    if len(base) != len(base_rows) or len(student) != len(candidate_rows): raise RuntimeError("Duplicate full ids")
    missing, extra = set(base) - set(student), set(student) - set(base)
    mismatch = sum(base[key].get("prompt_hash") != student[key].get("prompt_hash") for key in set(base) & set(student))
    base_counts = Counter(str(row["dataset_key"]) for row in base_rows); student_counts = Counter(str(row["dataset_key"]) for row in candidate_rows)
    if dict(base_counts) != EXPECTED or dict(student_counts) != EXPECTED: raise RuntimeError("Full counts changed")
    base_correct = Counter(str(row["dataset_key"]) for row in base_rows if row.get("correct") is True)
    student_correct = Counter(str(row["dataset_key"]) for row in candidate_rows if row.get("correct") is True)
    base_accuracy = {task: base_correct[task] / base_counts[task] for task in TASKS}
    student_accuracy = {task: student_correct[task] / student_counts[task] for task in TASKS}
    ratios = {task: min(student_accuracy[task] / base_accuracy[task], 1.0) for task in TASKS}
    macro = sum(ratios.values()) / 3
    errors = sum(bool(row.get("generation_error")) for row in candidate_rows)
    passed = not missing and not extra and not mismatch and not errors and all(value >= args.minimum for value in ratios.values()) and macro >= args.minimum
    report = {
        "gate": f"{args.stage}-EDGE-FULL-RETENTION", "check_version": "1.0", "created_by": "scripts/full_retention_gate.py",
        "created_ts": datetime.now(timezone.utc).isoformat(), "status": "passed" if passed else "failed",
        "decision": "meets_full_retention_requirement" if passed else "does_not_meet_full_retention_requirement",
        "feedback_policy": "aggregate_domain_metrics_only", "baseline_trace_hash": sha(baseline), "candidate_trace_hash": sha(candidate),
        "expected_counts": EXPECTED, "matched_sample_ids": not missing and not extra,
        "missing_sample_count": len(missing), "extra_sample_count": len(extra), "prompt_mismatch_count": mismatch,
        "baseline_correct_counts": dict(base_correct), "candidate_correct_counts": dict(student_correct),
        "baseline_accuracy_by_dataset": base_accuracy, "candidate_accuracy_by_dataset": student_accuracy,
        "retention_ratios": ratios, "capped_macro_ratio": macro, "generation_error_count": errors,
        "formal_full_completed": True, "item_level_feedback_allowed_for_training": False,
    }
    report["report_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, output)
    print(f"Wrote {output.relative_to(ROOT)} status={report['status']} ratios={ratios} macro={macro:.6f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
