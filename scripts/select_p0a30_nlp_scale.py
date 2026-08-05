#!/usr/bin/env python3
"""Select an NLP LoRA scale only when it improves untouched CMMLU dev."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/audit/p0a30/nlp_scale_selection.json")
    args = parser.parse_args()
    try:
        paths = {
            "1.0": "reports/audit/p0a30/scale_1p0.json",
            "0.75": "reports/audit/p0a30/scale_0p75.json",
            "1.25": "reports/audit/p0a30/scale_1p25.json",
        }
        audits = {scale: read(path) for scale, path in paths.items()}
        for scale, audit in audits.items():
            expected = {
                "status": "passed",
                "gate": "P0-A30-NLP-SCALE-VALIDATION",
                "domain": "nlp",
                "sample_count": 235,
                "generation_error_count": 0,
                "thinking": "off",
            }
            for key, value in expected.items():
                if audit.get(key) != value:
                    raise RuntimeError(f"scale {scale} {key} mismatch")
        hashes = {audit["manifest_hash"] for audit in audits.values()}
        if len(hashes) != 1:
            raise RuntimeError("P0-A30 manifest hashes differ")
        reference = int(audits["1.0"]["correct_count"])
        ranked = sorted(
            ((int(audit["correct_count"]), scale) for scale, audit in audits.items() if scale != "1.0"),
            key=lambda item: (-item[0], abs(float(item[1]) - 1.0), float(item[1])),
        )
        best_correct, best_scale = ranked[0]
        gain_questions = best_correct - reference
        gain_accuracy = gain_questions / 235
        passed = gain_questions >= 3
        selected_scale = best_scale if passed else "1.0"
        selected_path = {
            "0.75": "models/adapters/p0a30/nlp-step136-scale-0p75",
            "1.0": "models/checkpoints/p0a10/nlp-specialist/checkpoint-136",
            "1.25": "models/adapters/p0a30/nlp-step136-scale-1p25",
        }[selected_scale]
        report = {
            "gate": "P0-A30-NLP-SCALE-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a30_nlp_scale.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "decision": "run_one_frozen_nlp_gate" if passed else "retain_scale_1p0_and_train",
            "reference_scale": 1.0,
            "reference_correct_count": reference,
            "candidate_correct_counts": {scale: int(audit["correct_count"]) for scale, audit in audits.items() if scale != "1.0"},
            "best_candidate_scale": float(best_scale),
            "best_candidate_correct_count": best_correct,
            "gain_questions": gain_questions,
            "gain_accuracy": gain_accuracy,
            "minimum_gain_questions": 3,
            "selected_scale": float(selected_scale),
            "selected_adapter": selected_path,
            "formal_test_opened": False,
            "inputs": {path: sha256_file(ROOT / path) for path in paths.values()},
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(ROOT / args.output, report)
        print(f"Wrote {args.output}")
        print(f"status={report['status']} reference={reference}/235 best={best_correct}/235 scale={best_scale}")
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A30 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
