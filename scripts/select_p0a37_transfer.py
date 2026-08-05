#!/usr/bin/env python3
"""Check whether P0-A36 gains transfer from synthetic holdout to C-Eval dev."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "reports/audit/p0a37"


def load(name: str) -> dict:
    path = AUDIT_ROOT / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A37-NLP-TRANSFER-VALIDATION",
        "domain": "nlp",
        "sample_count": 260,
        "generation_error_count": 0,
        "thinking": "off",
        "max_tokens": 256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"{path.name} {key} mismatch")
    return value


def main() -> int:
    try:
        initial = load("initial_nlp")
        candidates = {step: load(f"nlp_{step}") for step in (64, 128)}
        hashes = {initial["manifest_hash"], *[value["manifest_hash"] for value in candidates.values()]}
        if len(hashes) != 1:
            raise RuntimeError("P0-A37 manifest mismatch")
        initial_correct = int(initial["correct_count"])
        rows = [
            {
                "step": step,
                "correct_count": int(value["correct_count"]),
                "accuracy": float(value["accuracy"]),
                "gain_questions": int(value["correct_count"]) - initial_correct,
                "audit": f"reports/audit/p0a37/nlp_{step}.json",
                "audit_hash": sha256_file(AUDIT_ROOT / f"nlp_{step}.json"),
            }
            for step, value in candidates.items()
        ]
        best = sorted(rows, key=lambda row: (-row["correct_count"], row["step"]))[0]
        passed = best["gain_questions"] >= 3
        report = {
            "gate": "P0-A37-NLP-TRANSFER-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a37_transfer.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "initial_correct_count": initial_correct,
            "initial_accuracy": float(initial["accuracy"]),
            "candidates": rows,
            "minimum_gain_questions": 3,
            "best_step": best["step"],
            "best_gain_questions": best["gain_questions"],
            "decision": "data_route_transfers" if passed else "weak_training_does_not_transfer",
            "frozen_nlp100_opened": False,
            "formal_full_opened": False,
            "per_item_feedback_read": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        output = AUDIT_ROOT / "transfer_selection.json"
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(
            f"status={report['status']} initial={initial_correct}/260 "
            f"best_step={best['step']} gain={best['gain_questions']}"
        )
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A37 transfer selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
