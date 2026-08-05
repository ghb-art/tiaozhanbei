#!/usr/bin/env python3
"""Select P0-A36 from aggregate results on its held-out balanced MCQs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a36").resolve()
ALLOWED_STEPS = {64, 128}
MINIMUM_GAIN_QUESTIONS = 6


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    return path.relative_to(ROOT.resolve()).as_posix()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A36 audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A36-BALANCED-MCQ-VALIDATION",
        "domain": "nlp",
        "sample_count": 256,
        "generation_error_count": 0,
        "thinking": "off",
        "max_tokens": 256,
        "gate300_loaded": False,
        "formal_full_loaded": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise SelectionError(
                f"{path.name} {key}={value.get(key)!r}, expected {expected_value!r}"
            )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-audit", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        initial_path = resolve(args.initial_audit)
        initial = load(initial_path)
        initial_correct = int(initial["correct_count"])
        initial_canonical = float(initial["canonical_format_rate"])
        candidates: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in args.candidate:
            step_text, separator, path_text = raw.partition("=")
            if not separator or not step_text.isdigit():
                raise SelectionError("Candidate must use STEP=PATH")
            step = int(step_text)
            if step not in ALLOWED_STEPS or step in seen:
                raise SelectionError(f"Unregistered or duplicate step: {step}")
            seen.add(step)
            path = resolve(path_text)
            value = load(path)
            if value["manifest_hash"] != initial["manifest_hash"]:
                raise SelectionError(f"Manifest mismatch: {path}")
            correct = int(value["correct_count"])
            gain = correct - initial_correct
            canonical = float(value["canonical_format_rate"])
            failures: list[str] = []
            if gain < MINIMUM_GAIN_QUESTIONS:
                failures.append(f"gain_questions={gain}<required={MINIMUM_GAIN_QUESTIONS}")
            if canonical + 1e-12 < initial_canonical:
                failures.append(
                    f"canonical={canonical:.6f}<initial={initial_canonical:.6f}"
                )
            candidates.append(
                {
                    "step": step,
                    "audit": display(path),
                    "audit_hash": sha256_file(path),
                    "correct_count": correct,
                    "accuracy": float(value["accuracy"]),
                    "canonical_format_rate": canonical,
                    "gain_questions": gain,
                    "eligible": not failures,
                    "failures": failures,
                }
            )
        if seen != ALLOWED_STEPS:
            raise SelectionError(f"Exactly steps {sorted(ALLOWED_STEPS)} are required")
        eligible = sorted(
            (item for item in candidates if item["eligible"]),
            key=lambda item: (-int(item["correct_count"]), int(item["step"])),
        )
        selected = eligible[0] if eligible else None
        output = resolve(args.output)
        if output != AUDIT_ROOT / "nlp_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        report = {
            "gate": "P0-A36-BALANCED-MCQ-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a36_nlp.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "initial_audit": display(initial_path),
            "initial_audit_hash": sha256_file(initial_path),
            "initial_correct_count": initial_correct,
            "initial_accuracy": float(initial["accuracy"]),
            "initial_canonical_format_rate": initial_canonical,
            "minimum_gain_questions": MINIMUM_GAIN_QUESTIONS,
            "candidates": sorted(candidates, key=lambda item: int(item["step"])),
            "selected_step": int(selected["step"]) if selected else None,
            "selected_model_id": f"p0a36-nlp-{selected['step']}" if selected else None,
            "per_item_feedback_read": False,
            "frozen_nlp100_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(output, report)
        print(f"Wrote {display(output)}")
        print(
            f"status={report['status']} initial={initial_correct}/256 "
            f"selected_step={report['selected_step']}"
        )
        return 0 if selected else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A36 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
