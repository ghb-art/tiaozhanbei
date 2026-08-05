#!/usr/bin/env python3
"""Select native thinking only when it materially improves aggregate NLP accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a35").resolve()


def load(path: Path, thinking: str) -> dict:
    if not path.is_file() or AUDIT_ROOT not in path.resolve().parents:
        raise RuntimeError(f"Invalid P0-A35 audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A35-NLP-RUNTIME-VALIDATION",
        "domain": "nlp",
        "sample_count": 260,
        "generation_error_count": 0,
        "thinking": thinking,
        "max_tokens": 768,
        "gate300_loaded": False,
        "formal_full_loaded": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(
                f"{path.name} {key}={value.get(key)!r}, expected {expected_value!r}"
            )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-audit", required=True)
    parser.add_argument("--on-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        off_path = (ROOT / args.off_audit).resolve()
        on_path = (ROOT / args.on_audit).resolve()
        output = (ROOT / args.output).resolve()
        if output != AUDIT_ROOT / "runtime_selection.json":
            raise RuntimeError(f"Unexpected output: {output}")
        off = load(off_path, "off")
        on = load(on_path, "on")
        if off["manifest_hash"] != on["manifest_hash"]:
            raise RuntimeError("Runtime candidates used different manifests")
        off_correct = int(off["correct_count"])
        on_correct = int(on["correct_count"])
        gain = on_correct - off_correct
        canonical_ok = float(on["canonical_format_rate"]) >= float(off["canonical_format_rate"])
        passed = gain >= 6 and canonical_ok
        report = {
            "gate": "P0-A35-NLP-RUNTIME-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a35_runtime.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "off_correct_count": off_correct,
            "off_accuracy": float(off["accuracy"]),
            "off_canonical_format_rate": float(off["canonical_format_rate"]),
            "on_correct_count": on_correct,
            "on_accuracy": float(on["accuracy"]),
            "on_canonical_format_rate": float(on["canonical_format_rate"]),
            "gain_questions": gain,
            "minimum_gain_questions": 6,
            "canonical_not_worse": canonical_ok,
            "selected_thinking": "on" if passed else "off",
            "selected_adapter": "models/checkpoints/p0a10/nlp-specialist/checkpoint-136",
            "frozen_nlp100_opened": False,
            "formal_full_opened": False,
            "per_item_feedback_read": False,
            "inputs": {
                off_path.relative_to(ROOT).as_posix(): sha256_file(off_path),
                on_path.relative_to(ROOT).as_posix(): sha256_file(on_path),
            },
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(f"status={report['status']} off={off_correct}/260 on={on_correct}/260 gain={gain}")
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"P0-A35 runtime selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
