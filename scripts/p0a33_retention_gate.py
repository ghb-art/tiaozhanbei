#!/usr/bin/env python3
"""Compare the selected P0-A32 NLP candidate with the frozen 14B NLP100 baseline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a5_gate import build_messages
from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINE_AUDIT = ROOT / "reports/audit/gate_p0a5_baseline14b_gate300_eval.json"
BASELINE_TRACE = ROOT / "data/eval/p0a5_baseline14b_gate300.jsonl"
MANIFEST = ROOT / "data/p0a31/nlp_gate100.jsonl"
SELECTION = ROOT / "reports/audit/p0a32/nlp_selection.json"
CANDIDATE_AUDIT = ROOT / "reports/audit/p0a33/candidate.json"
CANDIDATE_TRACE = ROOT / "reports/audit/p0a33/candidate_trace.jsonl"
OUTPUT = ROOT / "reports/audit/gate_p0a33_nlp_retention.json"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        baseline_audit = json.loads(BASELINE_AUDIT.read_text(encoding="utf-8"))
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        candidate = json.loads(CANDIDATE_AUDIT.read_text(encoding="utf-8"))
        selected_step = selection.get("selected_step")
        if selection.get("status") != "passed" or selected_step not in (128, 256):
            raise RuntimeError("P0-A32 has no eligible selected checkpoint")
        if baseline_audit.get("status") != "passed" or baseline_audit.get("accuracy_by_domain", {}).get("nlp") != 0.79:
            raise RuntimeError("Frozen 14B NLP baseline changed")
        expected = {
            "status": "passed",
            "gate": "P0-A33-TRAINED-NLP100-EVAL",
            "sample_count": 100,
            "generation_error_count": 0,
            "thinking": "off",
            "max_tokens": 256,
            "formal_full_loaded": False,
        }
        for key, value in expected.items():
            if candidate.get(key) != value:
                raise RuntimeError(f"Candidate {key} mismatch")
        manifest = jsonl(MANIFEST)
        ids = {str(row["sample_id"]) for row in manifest}
        baseline_rows = {
            str(row["sample_id"]): row
            for row in jsonl(BASELINE_TRACE)
            if str(row.get("sample_id")) in ids
        }
        candidate_rows = {str(row["sample_id"]): row for row in jsonl(CANDIDATE_TRACE)}
        if len(baseline_rows) != 100 or len(candidate_rows) != 100 or set(baseline_rows) != set(candidate_rows):
            raise RuntimeError("P0-A33 sample ids differ from frozen baseline")
        prompt_mismatch = 0
        for row in manifest:
            sample_id = str(row["sample_id"])
            expected_hash = sha256_text(
                json.dumps(build_messages(row), ensure_ascii=False, sort_keys=True)
            )
            if baseline_rows[sample_id].get("prompt_hash") != expected_hash:
                raise RuntimeError(f"Frozen baseline prompt changed: {sample_id}")
            if candidate_rows[sample_id].get("prompt_hash") != expected_hash:
                prompt_mismatch += 1
        if prompt_mismatch:
            raise RuntimeError(f"Candidate prompt mismatch: {prompt_mismatch}")
        baseline_correct = sum(int(bool(row.get("correct"))) for row in baseline_rows.values())
        candidate_correct = int(candidate["correct_count"])
        if baseline_correct != 79:
            raise RuntimeError(f"Frozen baseline count changed: {baseline_correct}")
        retention = min(candidate_correct / baseline_correct, 1.0)
        screen_passed = retention >= 0.78
        recommended_full = retention >= 0.82
        status = "passed" if recommended_full else "failed"
        report = {
            "gate": "P0-A33-TRAINED-NLP100-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a33_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": (
                "proceed_to_quantized_nlp_deployment"
                if recommended_full
                else "freeze_nlp_failure_and_stop_model_tuning"
            ),
            "selected_step": int(selected_step),
            "baseline_correct_count": baseline_correct,
            "candidate_correct_count": candidate_correct,
            "nlp_retention": retention,
            "minimum_screen_retention": 0.78,
            "recommended_full_retention": 0.82,
            "screen_passed": screen_passed,
            "recommended_full": recommended_full,
            "generation_error_count": 0,
            "prompt_mismatch_count": prompt_mismatch,
            "selection_hash": sha256_file(SELECTION),
            "baseline_trace_hash": sha256_file(BASELINE_TRACE),
            "candidate_trace_hash": sha256_file(CANDIDATE_TRACE),
            "per_item_feedback_read": False,
            "formal_test_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(
            f"status={status} candidate={candidate_correct}/100 baseline=79/100 "
            f"retention={retention:.6f}"
        )
        return 0 if recommended_full else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A33 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
