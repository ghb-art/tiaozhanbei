#!/usr/bin/env python3
"""Select a Code thinking runtime from aggregate-only P0-A20 audits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a20").resolve()
ALLOWED = {"base-thinking", "step256-thinking"}


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    return path.relative_to(ROOT.resolve()).as_posix()


def load(path: Path, thinking: str) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A20-CODE-THINKING-VALIDATION",
        "domain": "code",
        "sample_count": 239,
        "generation_error_count": 0,
        "thinking": thinking,
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
        base_path = resolve(args.base_audit)
        base = load(base_path, "off")
        base_accuracy = float(base["accuracy"])
        base_canonical = float(base["canonical_format_rate"])
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in args.candidate:
            name, separator, path_text = raw.partition("=")
            if not separator or name not in ALLOWED or name in seen:
                raise SelectionError(f"Unregistered or duplicate candidate: {name}")
            seen.add(name)
            path = resolve(path_text)
            value = load(path, "on")
            if value["manifest_hash"] != base["manifest_hash"]:
                raise SelectionError(f"Manifest mismatch: {path}")
            accuracy = float(value["accuracy"])
            canonical = float(value["canonical_format_rate"])
            gain = accuracy - base_accuracy
            failures: list[str] = []
            if gain + 1e-12 < 0.03:
                failures.append(f"gain={gain:.6f}<required=0.030000")
            if canonical + 1e-12 < base_canonical:
                failures.append(f"canonical={canonical:.6f}<base={base_canonical:.6f}")
            candidates.append(
                {
                    "name": name,
                    "audit": display(path),
                    "audit_hash": sha256_file(path),
                    "served_model_id": value["served_model_id"],
                    "accuracy": accuracy,
                    "correct_count": int(value["correct_count"]),
                    "canonical_format_rate": canonical,
                    "gain": gain,
                    "mean_latency_ms": float(value["mean_latency_ms"]),
                    "eligible": not failures,
                    "failures": failures,
                }
            )
        if seen != ALLOWED:
            raise SelectionError(f"Exactly candidates {sorted(ALLOWED)} are required")
        preference = {"base-thinking": 0, "step256-thinking": 1}
        eligible = sorted(
            (item for item in candidates if item["eligible"]),
            key=lambda item: (-float(item["accuracy"]), preference[str(item["name"])]),
        )
        selected = eligible[0] if eligible else None
        output = resolve(args.output)
        if output != AUDIT_ROOT / "code_runtime_selection.json":
            raise SelectionError(f"Unexpected output: {output}")
        report = {
            "gate": "P0-A20-CODE-THINKING-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a20_code.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "base_audit": display(base_path),
            "base_audit_hash": sha256_file(base_path),
            "base_accuracy": base_accuracy,
            "base_correct_count": int(base["correct_count"]),
            "base_canonical_format_rate": base_canonical,
            "minimum_absolute_gain": 0.03,
            "candidates": sorted(candidates, key=lambda item: preference[str(item["name"])]),
            "selected_runtime": str(selected["name"]) if selected else None,
            "selected_model_id": str(selected["served_model_id"]) if selected else None,
            "per_item_feedback_read": False,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(output, report)
        print(f"Wrote {display(output)}")
        print(
            f"status={report['status']} base={base_accuracy:.6f} "
            f"selected={report['selected_runtime']}"
        )
        return 0 if selected else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A20 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
