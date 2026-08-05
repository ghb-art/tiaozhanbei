#!/usr/bin/env python3
"""Combine frozen full Math evidence with new Code/NLP gate retention."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/eval/p0a5_baseline14b_gate300.jsonl"
CANDIDATE = ROOT / "data/eval/p0a17_code_nlp_gate200.jsonl"
EVAL_AUDIT = ROOT / "reports/audit/gate_p0a17_code_nlp_gate200_eval.json"
MATH_AUDIT = ROOT / "reports/audit/gate_p0a4_official_full_retention.json"
OUTPUT = ROOT / "reports/audit/gate_p0a17_code_nlp_retention.json"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        if OUTPUT.exists():
            raise RuntimeError("P0-A17 retention artifact already exists")
        eval_audit = json.loads(EVAL_AUDIT.read_text(encoding="utf-8"))
        if eval_audit.get("status") != "passed" or eval_audit.get("generation_error_count") != 0:
            raise RuntimeError("P0-A17 evaluation audit did not pass")
        math = json.loads(MATH_AUDIT.read_text(encoding="utf-8"))
        math_ratio = float(math["ratios"]["math_ratio"])
        baseline_rows = [row for row in read(BASELINE) if row.get("domain") in {"code", "nlp"}]
        candidate_rows = read(CANDIDATE)
        baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
        candidate_by_id = {str(row["sample_id"]): row for row in candidate_rows}
        if set(baseline_by_id) != set(candidate_by_id) or len(candidate_rows) != 200:
            raise RuntimeError("P0-A17 baseline/candidate sample mismatch")
        prompt_mismatch = sum(
            baseline_by_id[key].get("prompt_hash") != candidate_by_id[key].get("prompt_hash")
            for key in baseline_by_id
        )
        if prompt_mismatch:
            raise RuntimeError(f"P0-A17 prompt mismatch: {prompt_mismatch}")
        base_correct = Counter(str(row["domain"]) for row in baseline_rows if row.get("correct") is True)
        candidate_correct = Counter(str(row["domain"]) for row in candidate_rows if row.get("correct") is True)
        baseline_accuracy = {domain: base_correct[domain] / 100 for domain in ("code", "nlp")}
        candidate_accuracy = {domain: candidate_correct[domain] / 100 for domain in ("code", "nlp")}
        ratios = {
            domain: min(candidate_accuracy[domain] / baseline_accuracy[domain], 1.0)
            for domain in ("code", "nlp")
        }
        thresholds = {"math_full": 0.80, "code_gate": 0.78, "nlp_gate": 0.78}
        failures = []
        if math_ratio < thresholds["math_full"]:
            failures.append("math_full_retention")
        if ratios["code"] < thresholds["code_gate"]:
            failures.append("code_gate_retention")
        if ratios["nlp"] < thresholds["nlp_gate"]:
            failures.append("nlp_gate_retention")
        status = "passed" if not failures else "failed"
        report = {
            "gate": "P0-A17-CODE-NLP-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a17_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "proceed_to_edge_deployment" if status == "passed" else "retrain_failed_domains",
            "math_full_retention": math_ratio,
            "math_evidence": MATH_AUDIT.relative_to(ROOT).as_posix(),
            "math_evidence_hash": sha256_file(MATH_AUDIT),
            "baseline_accuracy": baseline_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "retention_ratios": ratios,
            "thresholds": thresholds,
            "prompt_mismatch_count": prompt_mismatch,
            "generation_error_count": 0,
            "failures": failures,
            "recommended_full": math_ratio >= 0.82 and all(value >= 0.82 for value in ratios.values()),
            "candidate_trace_hash": sha256_file(CANDIDATE),
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(f"status={status} math_full={math_ratio:.6f} code={ratios['code']:.6f} nlp={ratios['nlp']:.6f}")
        return 0 if status == "passed" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A17 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
