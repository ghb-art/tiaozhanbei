#!/usr/bin/env python3
"""Select P0-A34 using aggregate results on the untouched C-Eval dev split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a34").resolve()
ALLOWED_STEPS = {64, 128}
MINIMUM_GAIN = 0.02


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    return path.relative_to(ROOT.resolve()).as_posix()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A34 audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A34-CHINESE-EXAM-VALIDATION",
        "domain": "nlp",
        "sample_count": 260,
        "generation_error_count": 0,
        "thinking": "off",
        "max_tokens": 256,
        "gate300_loaded": False,
        "formal_full_loaded": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise SelectionError(
                f"Audit {path.name} has {key}={value.get(key)!r}, expected {expected_value!r}"
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
        initial_accuracy = float(initial["accuracy"])
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
            accuracy = float(value["accuracy"])
            canonical = float(value["canonical_format_rate"])
            gain = accuracy - initial_accuracy
            failures: list[str] = []
            if gain + 1e-12 < MINIMUM_GAIN:
                failures.append(f"gain={gain:.6f}<required={MINIMUM_GAIN:.6f}")
            if canonical + 1e-12 < initial_canonical:
                failures.append(
                    f"canonical={canonical:.6f}<initial={initial_canonical:.6f}"
                )
            candidates.append(
                {
                    "step": step,
                    "audit": display(path),
                    "audit_hash": sha256_file(path),
                    "accuracy": accuracy,
                    "correct_count": int(value["correct_count"]),
                    "canonical_format_rate": canonical,
                    "gain_over_initial_adapter": gain,
                    "eligible": not failures,
                    "failures": failures,
                }
            )
        if seen != ALLOWED_STEPS:
            raise SelectionError(f"Exactly steps {sorted(ALLOWED_STEPS)} are required")
        eligible = sorted(
            (item for item in candidates if item["eligible"]),
            key=lambda item: (-float(item["accuracy"]), int(item["step"])),
        )
        selected = eligible[0] if eligible else None
        output = resolve(args.output)
        if output != AUDIT_ROOT / "nlp_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        report = {
            "gate": "P0-A34-CHINESE-EXAM-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a34_nlp.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "initial_audit": display(initial_path),
            "initial_audit_hash": sha256_file(initial_path),
            "initial_accuracy": initial_accuracy,
            "initial_correct_count": int(initial["correct_count"]),
            "initial_canonical_format_rate": initial_canonical,
            "minimum_gain_over_initial_adapter": MINIMUM_GAIN,
            "candidates": sorted(candidates, key=lambda item: int(item["step"])),
            "selected_step": int(selected["step"]) if selected else None,
            "selected_model_id": f"p0a34-nlp-{selected['step']}" if selected else None,
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
            f"status={report['status']} initial={initial_accuracy:.6f} "
            f"selected_step={report['selected_step']}"
        )
        return 0 if selected else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A34 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
