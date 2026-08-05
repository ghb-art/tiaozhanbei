#!/usr/bin/env python3
"""Compare one P0-A25 candidate with the frozen 14B on fresh Code100."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINE_AUDIT = ROOT / "reports/audit/p0a26/baseline.json"
CANDIDATE_AUDIT = ROOT / "reports/audit/p0a26/candidate.json"
BASELINE_TRACE = ROOT / "reports/audit/p0a26/baseline_trace.jsonl"
CANDIDATE_TRACE = ROOT / "reports/audit/p0a26/candidate_trace.jsonl"
SELECTION = ROOT / "reports/audit/p0a25/code_selection.json"
OUTPUT = ROOT / "reports/audit/gate_p0a26_code_retention.json"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        if OUTPUT.exists():
            raise RuntimeError("P0-A26 retention artifact already exists")
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        if selection.get("status") != "passed" or selection.get("selected_step") != 192:
            raise RuntimeError("P0-A25 selected candidate changed")
        base_audit = json.loads(BASELINE_AUDIT.read_text(encoding="utf-8"))
        candidate_audit = json.loads(CANDIDATE_AUDIT.read_text(encoding="utf-8"))
        expected = {
            "status": "passed",
            "gate": "P0-A26-FRESH-CODE100-EVAL",
            "sample_count": 100,
            "generation_error_count": 0,
            "thinking": "off",
            "max_tokens": 768,
            "gate300_loaded": False,
            "formal_full_loaded": False,
        }
        for label, audit in (("baseline", base_audit), ("candidate", candidate_audit)):
            for key, value in expected.items():
                if audit.get(key) != value:
                    raise RuntimeError(
                        f"{label} audit {key}={audit.get(key)!r}, expected {value!r}"
                    )
        if base_audit["manifest_hash"] != candidate_audit["manifest_hash"]:
            raise RuntimeError("P0-A26 manifest hash mismatch")
        baseline = read(BASELINE_TRACE)
        candidate = read(CANDIDATE_TRACE)
        baseline_by_id = {str(row["sample_id"]): row for row in baseline}
        candidate_by_id = {str(row["sample_id"]): row for row in candidate}
        if len(baseline) != 100 or len(candidate) != 100 or set(baseline_by_id) != set(candidate_by_id):
            raise RuntimeError("P0-A26 trace sample mismatch")
        prompt_mismatch = sum(
            baseline_by_id[key].get("prompt_hash")
            != candidate_by_id[key].get("prompt_hash")
            for key in baseline_by_id
        )
        if prompt_mismatch:
            raise RuntimeError(f"P0-A26 prompt mismatch: {prompt_mismatch}")
        baseline_correct = int(base_audit["correct_count"])
        candidate_correct = int(candidate_audit["correct_count"])
        if baseline_correct <= 0:
            raise RuntimeError("P0-A26 baseline has no correct answers")
        retention = min(candidate_correct / baseline_correct, 1.0)
        status = "passed" if retention >= 0.78 else "failed"
        report = {
            "gate": "P0-A26-FRESH-CODE100-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a26_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "proceed_to_edge_deployment" if status == "passed" else "stop_code_candidate",
            "selected_step": 192,
            "baseline_accuracy": float(base_audit["accuracy"]),
            "candidate_accuracy": float(candidate_audit["accuracy"]),
            "baseline_correct_count": baseline_correct,
            "candidate_correct_count": candidate_correct,
            "code_retention": retention,
            "minimum_retention": 0.78,
            "recommended_full": retention >= 0.82,
            "prompt_mismatch_count": prompt_mismatch,
            "generation_error_count": 0,
            "selection_audit_hash": sha256_file(SELECTION),
            "baseline_trace_hash": sha256_file(BASELINE_TRACE),
            "candidate_trace_hash": sha256_file(CANDIDATE_TRACE),
            "p0a24_gate_trace_loaded": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(
            f"status={status} candidate={candidate_correct}/100 "
            f"baseline={baseline_correct}/100 retention={retention:.6f}"
        )
        return 0 if status == "passed" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A26 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
