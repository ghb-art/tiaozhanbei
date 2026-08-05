#!/usr/bin/env python3
"""Promote original Qwen3 only if it beats the P0-A10 reference on C-Eval."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    try:
        references = {
            "synth": read("reports/audit/p0a36/initial_nlp.json"),
            "ceval": read("reports/audit/p0a37/initial_nlp.json"),
        }
        candidates = {
            "synth": read("reports/audit/p0a39/original_synth.json"),
            "ceval": read("reports/audit/p0a39/original_ceval.json"),
        }
        expected = {
            "synth": (256, "P0-A39-ORIGINAL-QWEN3-SYNTH-VALIDATION"),
            "ceval": (260, "P0-A39-ORIGINAL-QWEN3-CEVAL-VALIDATION"),
        }
        metrics = {}
        for name in ("synth", "ceval"):
            sample_count, gate = expected[name]
            reference = references[name]
            candidate = candidates[name]
            if candidate.get("status") != "passed" or candidate.get("gate") != gate:
                raise RuntimeError(f"Invalid P0-A39 {name} candidate")
            if candidate.get("sample_count") != sample_count or candidate.get("max_tokens") != 256:
                raise RuntimeError(f"P0-A39 {name} runtime mismatch")
            if candidate.get("generation_error_count") != 0 or candidate.get("thinking") != "off":
                raise RuntimeError(f"P0-A39 {name} generation mismatch")
            if candidate["manifest_hash"] != reference["manifest_hash"]:
                raise RuntimeError(f"P0-A39 {name} manifest mismatch")
            metrics[name] = {
                "reference_correct_count": int(reference["correct_count"]),
                "candidate_correct_count": int(candidate["correct_count"]),
                "gain_questions": int(candidate["correct_count"]) - int(reference["correct_count"]),
                "candidate_accuracy": float(candidate["accuracy"]),
            }
        passed = metrics["ceval"]["gain_questions"] >= 3 and metrics["synth"]["gain_questions"] >= -3
        report = {
            "gate": "P0-A39-ORIGINAL-QWEN3-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a39_original.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "metrics": metrics,
            "minimum_ceval_gain_questions": 3,
            "minimum_synth_gain_questions": -3,
            "selected_model": "models/pretrained/Qwen--Qwen3-1.7B" if passed else None,
            "decision": "run_frozen_nlp100" if passed else "retain_p0a10",
            "frozen_nlp100_opened": False,
            "formal_full_opened": False,
            "per_item_feedback_read": False,
            "inputs": {
                path: sha256_file(ROOT / path)
                for path in (
                    "reports/audit/p0a36/initial_nlp.json",
                    "reports/audit/p0a37/initial_nlp.json",
                    "reports/audit/p0a39/original_synth.json",
                    "reports/audit/p0a39/original_ceval.json",
                )
            },
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        output = ROOT / "reports/audit/p0a39/original_selection.json"
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(f"status={report['status']} metrics={metrics}")
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A39 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
