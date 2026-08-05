#!/usr/bin/env python3
"""Select the sole P0-A16 joint Math runtime using aggregate audits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a16").resolve()


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load(path: Path, thinking: str) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid audit: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A16-MATH-JOINT-RUNTIME-VALIDATION",
        "sample_count": 1041,
        "thinking": thinking,
        "max_tokens": 768,
        "generation_error_count": 0,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise SelectionError(
                f"Audit {path.name} has {key}={data.get(key)!r}, expected {value!r}"
            )
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        base_path, candidate_path = resolve(args.base_audit), resolve(args.candidate_audit)
        base, candidate = load(base_path, "off"), load(candidate_path, "on")
        if base["manifest_hash"] != candidate["manifest_hash"]:
            raise SelectionError("P0-A16 manifest mismatch")
        base_accuracy = float(base["accuracy"])
        candidate_accuracy = float(candidate["accuracy"])
        gain = candidate_accuracy - base_accuracy
        canonical = float(candidate["canonical_format_rate"])
        failures: list[str] = []
        if gain < 0.02:
            failures.append(f"gain={gain:.6f}<required=0.020000")
        if canonical < 0.95:
            failures.append(f"canonical={canonical:.6f}<required=0.950000")
        status = "passed" if not failures else "failed"
        output = resolve(args.output)
        if output != AUDIT_ROOT / "math_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        report = {
            "gate": "P0-A16-MATH-JOINT-RUNTIME-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a16_math.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "base_audit": base_path.relative_to(ROOT).as_posix(),
            "base_audit_hash": sha256_file(base_path),
            "candidate_audit": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_audit_hash": sha256_file(candidate_path),
            "base_accuracy": base_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "gain": gain,
            "minimum_gain": 0.02,
            "candidate_canonical_format_rate": canonical,
            "selected_model_id": candidate["served_model_id"] if status == "passed" else None,
            "selected_runtime": "step64_thinking" if status == "passed" else None,
            "failures": failures,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(
            f"status={status} base={base_accuracy:.6f} candidate={candidate_accuracy:.6f} gain={gain:.6f}"
        )
        return 0 if status == "passed" else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A16 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
