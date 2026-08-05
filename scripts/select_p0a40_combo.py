#!/usr/bin/env python3
"""Select original Qwen3 plus the P0-A10 NLP adapter on aggregate transfer results."""

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
        specs = {
            "synth": {
                "reference": "reports/audit/p0a36/initial_nlp.json",
                "candidate": "reports/audit/p0a40/combo_synth.json",
                "gate": "P0-A40-ORIGINAL-PLUS-NLP-SYNTH-VALIDATION",
                "rows": 256,
            },
            "ceval": {
                "reference": "reports/audit/p0a37/initial_nlp.json",
                "candidate": "reports/audit/p0a40/combo_ceval.json",
                "gate": "P0-A40-ORIGINAL-PLUS-NLP-CEVAL-VALIDATION",
                "rows": 260,
            },
        }
        metrics = {}
        input_paths = []
        for name, spec in specs.items():
            reference = read(spec["reference"])
            candidate = read(spec["candidate"])
            input_paths.extend((spec["reference"], spec["candidate"]))
            expected = {
                "status": "passed",
                "gate": spec["gate"],
                "sample_count": spec["rows"],
                "max_tokens": 256,
                "thinking": "off",
                "generation_error_count": 0,
            }
            if any(candidate.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Invalid P0-A40 {name} audit")
            if candidate["manifest_hash"] != reference["manifest_hash"]:
                raise RuntimeError(f"P0-A40 {name} manifest mismatch")
            metrics[name] = {
                "reference_correct_count": int(reference["correct_count"]),
                "candidate_correct_count": int(candidate["correct_count"]),
                "gain_questions": int(candidate["correct_count"]) - int(reference["correct_count"]),
                "candidate_accuracy": float(candidate["accuracy"]),
            }
        passed = metrics["ceval"]["gain_questions"] >= 3 and metrics["synth"]["gain_questions"] >= -3
        report = {
            "gate": "P0-A40-ORIGINAL-PLUS-NLP-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a40_combo.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "metrics": metrics,
            "minimum_ceval_gain_questions": 3,
            "minimum_synth_gain_questions": -3,
            "selected_base": "models/pretrained/Qwen--Qwen3-1.7B" if passed else None,
            "selected_adapter": "models/checkpoints/p0a10/nlp-specialist/checkpoint-136" if passed else None,
            "decision": "run_frozen_nlp100" if passed else "train_fresh_original_base_adapter",
            "frozen_nlp100_opened": False,
            "formal_full_opened": False,
            "per_item_feedback_read": False,
            "inputs": {path: sha256_file(ROOT / path) for path in input_paths},
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        output = ROOT / "reports/audit/p0a40/combo_selection.json"
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(f"status={report['status']} metrics={metrics}")
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A40 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
