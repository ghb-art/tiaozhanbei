#!/usr/bin/env python3
"""Compare 14B and 1.7B fairly under the P0-A38 768-token runtime."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "reports/audit/p0a38"
OUTPUT = ROOT / "reports/audit/gate_p0a38_nlp_retention.json"


def load(name: str) -> dict:
    path = AUDIT_ROOT / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A38-NLP-LONG-OUTPUT-GATE",
        "domain": "nlp",
        "sample_count": 100,
        "generation_error_count": 0,
        "thinking": "off",
        "max_tokens": 768,
        "formal_full_loaded": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"{path.name} {key} mismatch")
    return value


def main() -> int:
    try:
        baseline = load("baseline14b")
        student = load("student17b")
        if baseline["manifest_hash"] != student["manifest_hash"]:
            raise RuntimeError("P0-A38 manifest mismatch")
        baseline_accuracy = float(baseline["accuracy"])
        student_accuracy = float(student["accuracy"])
        if baseline_accuracy <= 0:
            raise RuntimeError("P0-A38 baseline accuracy is zero")
        retention = min(student_accuracy / baseline_accuracy, 1.0)
        passed = retention + 1e-12 >= 0.80
        report = {
            "gate": "P0-A38-NLP-LONG-OUTPUT-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a38_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "runtime": {"thinking": "off", "max_tokens": 768},
            "baseline_correct_count": int(baseline["correct_count"]),
            "baseline_accuracy": baseline_accuracy,
            "student_correct_count": int(student["correct_count"]),
            "student_accuracy": student_accuracy,
            "nlp_retention": retention,
            "minimum_retention": 0.80,
            "recommended_retention": 0.82,
            "recommended_line_met": retention + 1e-12 >= 0.82,
            "same_manifest_prompt_scorer_runtime": True,
            "generation_error_count": 0,
            "formal_full_opened": False,
            "per_item_feedback_read": False,
            "inputs": {
                "reports/audit/p0a38/baseline14b.json": sha256_file(AUDIT_ROOT / "baseline14b.json"),
                "reports/audit/p0a38/student17b.json": sha256_file(AUDIT_ROOT / "student17b.json"),
            },
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(OUTPUT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(
            f"status={report['status']} baseline={report['baseline_correct_count']}/100 "
            f"student={report['student_correct_count']}/100 retention={retention:.6f}"
        )
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A38 retention failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
