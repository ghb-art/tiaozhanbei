#!/usr/bin/env python3
"""Combine frozen Math/NLP evidence with the new P0-A24 Code100 result."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/eval/p0a5_baseline14b_gate300.jsonl"
CANDIDATE = ROOT / "data/eval/p0a24_code_gate100.jsonl"
EVAL_AUDIT = ROOT / "reports/audit/gate_p0a24_code_gate100_eval.json"
SELECTION = ROOT / "reports/audit/p0a23/code_selection.json"
MATH_AUDIT = ROOT / "reports/audit/gate_p0a4_official_full_retention.json"
NLP_AUDIT = ROOT / "reports/audit/gate_p0a17_code_nlp_retention.json"
OUTPUT = ROOT / "reports/audit/gate_p0a24_code_retention.json"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        if OUTPUT.exists():
            raise RuntimeError("P0-A24 retention artifact already exists")
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        if selection.get("status") != "passed" or selection.get("selected_step") not in {96, 192}:
            raise RuntimeError("P0-A23 selection did not pass")
        eval_audit = json.loads(EVAL_AUDIT.read_text(encoding="utf-8"))
        if (
            eval_audit.get("status") != "passed"
            or eval_audit.get("generation_error_count") != 0
            or eval_audit.get("selected_step") != selection.get("selected_step")
        ):
            raise RuntimeError("P0-A24 evaluation audit did not pass or selected step changed")
        math = json.loads(MATH_AUDIT.read_text(encoding="utf-8"))
        nlp = json.loads(NLP_AUDIT.read_text(encoding="utf-8"))
        math_ratio = float(math["ratios"]["math_ratio"])
        nlp_ratio = float(nlp["retention_ratios"]["nlp"])
        baseline_rows = [row for row in read(BASELINE) if row.get("domain") == "code"]
        candidate_rows = read(CANDIDATE)
        baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
        candidate_by_id = {str(row["sample_id"]): row for row in candidate_rows}
        if set(baseline_by_id) != set(candidate_by_id) or len(candidate_rows) != 100:
            raise RuntimeError("P0-A24 baseline/candidate sample mismatch")
        prompt_mismatch = sum(
            baseline_by_id[key].get("prompt_hash")
            != candidate_by_id[key].get("prompt_hash")
            for key in baseline_by_id
        )
        if prompt_mismatch:
            raise RuntimeError(f"P0-A24 prompt mismatch: {prompt_mismatch}")
        baseline_correct = sum(row.get("correct") is True for row in baseline_rows)
        candidate_correct = sum(row.get("correct") is True for row in candidate_rows)
        baseline_accuracy = baseline_correct / 100
        candidate_accuracy = candidate_correct / 100
        code_ratio = min(candidate_accuracy / baseline_accuracy, 1.0)
        thresholds = {"math_full": 0.80, "code_gate": 0.78, "nlp_frozen": 0.78}
        failures: list[str] = []
        if math_ratio < thresholds["math_full"]:
            failures.append("math_full_retention")
        if code_ratio < thresholds["code_gate"]:
            failures.append("code_gate_retention")
        if nlp_ratio < thresholds["nlp_frozen"]:
            failures.append("nlp_frozen_retention")
        status = "passed" if not failures else "failed"
        report = {
            "gate": "P0-A24-CODE-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a24_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "proceed_to_edge_deployment" if status == "passed" else "stop_code_candidate",
            "selected_step": int(selection["selected_step"]),
            "math_full_retention": math_ratio,
            "nlp_frozen_retention": nlp_ratio,
            "baseline_code_accuracy": baseline_accuracy,
            "candidate_code_accuracy": candidate_accuracy,
            "baseline_code_correct": baseline_correct,
            "candidate_code_correct": candidate_correct,
            "code_retention": code_ratio,
            "thresholds": thresholds,
            "prompt_mismatch_count": prompt_mismatch,
            "generation_error_count": 0,
            "failures": failures,
            "recommended_full": all(
                value >= 0.82 for value in (math_ratio, code_ratio, nlp_ratio)
            ),
            "selection_audit_hash": sha256_file(SELECTION),
            "candidate_trace_hash": sha256_file(CANDIDATE),
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(
            f"status={status} math={math_ratio:.6f} code={code_ratio:.6f} "
            f"nlp={nlp_ratio:.6f} correct={candidate_correct}/{baseline_correct}"
        )
        return 0 if status == "passed" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A24 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
