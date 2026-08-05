#!/usr/bin/env python3
"""Select the preregistered P0-A14 vote3 runtime from aggregate audits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a14").resolve()


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load(path: Path, mode: str, responses: int) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A14 audit path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A14-MATH-SELF-CONSISTENCY-VALIDATION",
        "mode": mode,
        "sample_count": 727,
        "responses_per_sample": responses,
        "thinking": True,
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
        base_path = resolve(args.base_audit)
        candidate_path = resolve(args.candidate_audit)
        output = resolve(args.output)
        if output != AUDIT_ROOT / "runtime_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        base = load(base_path, "single", 1)
        candidate = load(candidate_path, "vote3", 3)
        if (
            base["manifest_hash"] != candidate["manifest_hash"]
            or base["served_model_id"] != candidate["served_model_id"]
        ):
            raise SelectionError("P0-A14 base/candidate identity mismatch")
        base_accuracy = float(base["accuracy"])
        candidate_accuracy = float(candidate["accuracy"])
        gain = candidate_accuracy - base_accuracy
        canonical = float(candidate["canonical_format_rate"])
        latency_ratio = float(candidate["mean_latency_ms"]) / max(
            float(base["mean_latency_ms"]), 1e-9
        )
        failures: list[str] = []
        if gain < 0.02:
            failures.append(f"gain={gain:.6f}<required=0.020000")
        if canonical < 0.95:
            failures.append(f"canonical_format_rate={canonical:.6f}<required=0.950000")
        status = "passed" if not failures else "failed"
        report = {
            "gate": "P0-A14-MATH-SELF-CONSISTENCY-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a14_runtime.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "promote_vote3" if status == "passed" else "close_gate",
            "base_audit": base_path.relative_to(ROOT).as_posix(),
            "base_audit_hash": sha256_file(base_path),
            "candidate_audit": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_audit_hash": sha256_file(candidate_path),
            "base_accuracy": base_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "gain": gain,
            "minimum_gain": 0.02,
            "candidate_canonical_format_rate": canonical,
            "minimum_canonical_format_rate": 0.95,
            "mean_latency_ratio": latency_ratio,
            "generation_error_count": 0,
            "selected_runtime": "thinking_vote3" if status == "passed" else None,
            "failures": failures,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(
            f"status={status} base={base_accuracy:.6f} candidate={candidate_accuracy:.6f} "
            f"gain={gain:.6f} latency_ratio={latency_ratio:.3f}"
        )
        return 0 if status == "passed" else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A14 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
