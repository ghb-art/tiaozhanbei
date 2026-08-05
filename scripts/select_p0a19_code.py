#!/usr/bin/env python3
"""Select one P0-A19 mixed-Code checkpoint from aggregate MBPP evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a19_code_mixed_distill.json"
AUDIT_ROOT = (ROOT / "reports/audit/p0a19").resolve()
ALLOWED_STEPS = {128, 256}


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    return path.relative_to(ROOT.resolve()).as_posix()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A19 audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A19-CODE-MIXED-VALIDATION",
        "domain": "code",
        "sample_count": 255,
        "generation_error_count": 0,
        "thinking": "off",
        "max_tokens": 768,
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
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        validation = json.loads(CONFIG.read_text(encoding="utf-8"))["validation"]
        minimum_gain = float(validation["minimum_absolute_gain"])
        base_path = resolve(args.base_audit)
        base = load(base_path)
        base_accuracy = float(base["accuracy"])
        base_canonical = float(base["canonical_format_rate"])
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
            if value["manifest_hash"] != base["manifest_hash"]:
                raise SelectionError(f"Manifest mismatch: {path}")
            accuracy = float(value["accuracy"])
            canonical = float(value["canonical_format_rate"])
            gain = accuracy - base_accuracy
            failures: list[str] = []
            if gain + 1e-12 < minimum_gain:
                failures.append(f"gain={gain:.6f}<required={minimum_gain:.6f}")
            if canonical + 1e-12 < base_canonical:
                failures.append(f"canonical={canonical:.6f}<base={base_canonical:.6f}")
            candidates.append(
                {
                    "step": step,
                    "audit": display(path),
                    "audit_hash": sha256_file(path),
                    "accuracy": accuracy,
                    "correct_count": int(value["correct_count"]),
                    "canonical_format_rate": canonical,
                    "gain": gain,
                    "eligible": not failures,
                    "failures": failures,
                }
            )
        if seen != ALLOWED_STEPS:
            raise SelectionError(f"Exactly steps {sorted(ALLOWED_STEPS)} must be evaluated")
        eligible = sorted(
            (item for item in candidates if item["eligible"]),
            key=lambda item: (-float(item["accuracy"]), int(item["step"])),
        )
        selected = eligible[0] if eligible else None
        output = resolve(args.output)
        if output != AUDIT_ROOT / "code_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        report = {
            "gate": "P0-A19-CODE-MIXED-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a19_code.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "base_audit": display(base_path),
            "base_audit_hash": sha256_file(base_path),
            "base_accuracy": base_accuracy,
            "base_correct_count": int(base["correct_count"]),
            "base_canonical_format_rate": base_canonical,
            "minimum_absolute_gain": minimum_gain,
            "candidates": sorted(candidates, key=lambda item: int(item["step"])),
            "selected_step": int(selected["step"]) if selected else None,
            "selected_model_id": f"p0a19-code-{selected['step']}" if selected else None,
            "p0a18_per_item_feedback_read": False,
            "validation_per_item_feedback_read": False,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(output, report)
        print(f"Wrote {display(output)}")
        print(
            f"status={report['status']} base={base_accuracy:.6f} "
            f"selected_step={report['selected_step']}"
        )
        return 0 if selected else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A19 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
