#!/usr/bin/env python3
"""Verify the deployed Q4_K_M + Code-LoRA candidate against frozen P0-A26."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINE_AUDIT = ROOT / "reports/audit/p0a26/baseline.json"
BASELINE_TRACE = ROOT / "reports/audit/p0a26/baseline_trace.jsonl"
HF_RETENTION = ROOT / "reports/audit/gate_p0a26_code_retention.json"
CANDIDATE_AUDIT = ROOT / "reports/audit/p0a27/candidate.json"
CANDIDATE_TRACE = ROOT / "reports/audit/p0a27/candidate_trace.jsonl"
OUTPUT = ROOT / "reports/audit/gate_p0a27_quantized_code_retention.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        baseline = read_json(BASELINE_AUDIT)
        hf_gate = read_json(HF_RETENTION)
        candidate = read_json(CANDIDATE_AUDIT)
        if baseline.get("status") != "passed" or hf_gate.get("status") != "passed":
            raise RuntimeError("Frozen P0-A26 prerequisite is not passed")
        expected = {
            "status": "passed",
            "gate": "P0-A27-QUANTIZED-CODE100-EVAL",
            "sample_count": 100,
            "generation_error_count": 0,
            "thinking": "off",
            "max_tokens": 768,
            "gate300_loaded": False,
            "formal_full_loaded": False,
        }
        for key, value in expected.items():
            if candidate.get(key) != value:
                raise RuntimeError(
                    f"Candidate audit {key}={candidate.get(key)!r}, expected {value!r}"
                )
        if baseline.get("manifest_hash") != candidate.get("manifest_hash"):
            raise RuntimeError("P0-A26/P0-A27 manifest hash mismatch")

        baseline_rows = {str(row["sample_id"]): row for row in read_jsonl(BASELINE_TRACE)}
        candidate_rows = {str(row["sample_id"]): row for row in read_jsonl(CANDIDATE_TRACE)}
        if len(baseline_rows) != 100 or len(candidate_rows) != 100:
            raise RuntimeError("P0-A27 trace does not contain 100 unique samples")
        if set(baseline_rows) != set(candidate_rows):
            raise RuntimeError("P0-A26/P0-A27 sample ids differ")
        prompt_mismatch = sum(
            baseline_rows[key].get("prompt_hash")
            != candidate_rows[key].get("prompt_hash")
            for key in baseline_rows
        )
        if prompt_mismatch:
            raise RuntimeError(f"P0-A27 prompt mismatch: {prompt_mismatch}")

        baseline_correct = int(baseline["correct_count"])
        hf_correct = int(hf_gate["candidate_correct_count"])
        candidate_correct = int(candidate["correct_count"])
        if baseline_correct <= 0:
            raise RuntimeError("Frozen baseline correct count is invalid")
        retention = min(candidate_correct / baseline_correct, 1.0)
        quantized_vs_hf = candidate_correct / hf_correct if hf_correct else 0.0
        status = "passed" if retention >= 0.78 else "failed"
        report = {
            "gate": "P0-A27-QUANTIZED-CODE100-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a27_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "proceed_to_memory_gate" if status == "passed" else "optimize_quantized_runtime",
            "baseline_correct_count": baseline_correct,
            "hf_candidate_correct_count": hf_correct,
            "quantized_candidate_correct_count": candidate_correct,
            "baseline_accuracy": float(baseline["accuracy"]),
            "quantized_candidate_accuracy": float(candidate["accuracy"]),
            "code_retention": retention,
            "quantized_vs_hf_ratio": quantized_vs_hf,
            "minimum_retention": 0.78,
            "recommended_full": retention >= 0.82,
            "prompt_mismatch_count": prompt_mismatch,
            "generation_error_count": 0,
            "baseline_audit_hash": sha256_file(BASELINE_AUDIT),
            "hf_retention_hash": sha256_file(HF_RETENTION),
            "candidate_audit_hash": sha256_file(CANDIDATE_AUDIT),
            "baseline_trace_hash": sha256_file(BASELINE_TRACE),
            "candidate_trace_hash": sha256_file(CANDIDATE_TRACE),
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(
            f"status={status} quantized={candidate_correct}/100 "
            f"baseline={baseline_correct}/100 retention={retention:.6f} "
            f"vs_hf={quantized_vs_hf:.6f}"
        )
        return 0 if status == "passed" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A27 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
