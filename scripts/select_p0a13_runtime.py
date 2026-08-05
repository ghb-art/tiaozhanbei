#!/usr/bin/env python3
"""Select the single preregistered P0-A13 Math runtime profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = (ROOT / "reports/audit/p0a13").resolve()
EXPECTED_MANIFEST = "data/p0a13/math_validation.jsonl"


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_audit(path: Path, thinking: str) -> dict[str, Any]:
    if not path.is_file() or ALLOWED_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A13 audit path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "gate": "P0-A13-MATH-RUNTIME-VALIDATION",
        "manifest": EXPECTED_MANIFEST,
        "domain": "math",
        "sample_count": 2237,
        "thinking": thinking,
        "max_tokens": 768,
        "generation_error_count": 0,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise SelectionError(
                f"Audit {path.name} has {key}={data.get(key)!r}, expected {expected!r}"
            )
    return data


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--minimum-canonical-format-rate", type=float, default=0.95)
    args = parser.parse_args()
    try:
        base_path = resolve(args.base_audit)
        candidate_path = resolve(args.candidate_audit)
        output = resolve(args.output)
        if output != ALLOWED_ROOT / "runtime_selection.json":
            raise SelectionError(f"Unexpected selection output: {output}")
        if not 0 <= args.minimum_gain <= 1:
            raise SelectionError("Invalid minimum gain")
        base = load_audit(base_path, "off")
        candidate = load_audit(candidate_path, "on")
        if base.get("manifest_hash") != candidate.get("manifest_hash"):
            raise SelectionError("Base/candidate validation manifest mismatch")
        if base.get("served_model_id") != candidate.get("served_model_id"):
            raise SelectionError("P0-A13 comparison must use the same base model id")
        base_accuracy = float(base["accuracy"])
        candidate_accuracy = float(candidate["accuracy"])
        gain = candidate_accuracy - base_accuracy
        failures: list[str] = []
        if gain < args.minimum_gain:
            failures.append(f"gain={gain:.6f}<required={args.minimum_gain:.6f}")
        canonical = float(candidate["canonical_format_rate"])
        if canonical < args.minimum_canonical_format_rate:
            failures.append(
                f"canonical_format_rate={canonical:.6f}<required="
                f"{args.minimum_canonical_format_rate:.6f}"
            )
        status = "passed" if not failures else "failed"
        report = {
            "gate": "P0-A13-MATH-RUNTIME-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a13_runtime.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "decision": "promote_math_thinking" if status == "passed" else "keep_base_and_close_gate",
            "base_audit": base_path.relative_to(ROOT).as_posix(),
            "base_audit_hash": sha256_file(base_path),
            "candidate_audit": candidate_path.relative_to(ROOT).as_posix(),
            "candidate_audit_hash": sha256_file(candidate_path),
            "base_accuracy": base_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "gain": gain,
            "minimum_gain": args.minimum_gain,
            "candidate_canonical_format_rate": canonical,
            "minimum_canonical_format_rate": args.minimum_canonical_format_rate,
            "generation_error_count": int(candidate["generation_error_count"]),
            "selected_runtime": "thinking_on" if status == "passed" else None,
            "failures": failures,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(
            f"status={status} base={base_accuracy:.6f} "
            f"candidate={candidate_accuracy:.6f} gain={gain:.6f}"
        )
        return 0 if status == "passed" else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A13 runtime selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
