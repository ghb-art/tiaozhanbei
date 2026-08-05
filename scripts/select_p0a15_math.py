#!/usr/bin/env python3
"""Select one of the two preregistered P0-A15 Math LoRA scales."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a15").resolve()


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or AUDIT_ROOT not in path.parents:
        raise SelectionError(f"Invalid P0-A15 audit: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "gate": "P0-A15-MATH-SCALED-ADAPTER-VALIDATION",
        "sample_count": 346,
        "thinking": "on",
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
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if len(args.candidate) != 2:
            raise SelectionError("Exactly two P0-A15 candidates are required")
        base_path = resolve(args.base_audit)
        base = load(base_path)
        parsed: list[tuple[float, Path, dict[str, Any]]] = []
        for item in args.candidate:
            scale_text, path_text = item.split("=", 1)
            scale = float(scale_text)
            if scale not in {0.5, 1.0}:
                raise SelectionError(f"Unexpected P0-A15 scale: {scale}")
            path = resolve(path_text)
            parsed.append((scale, path, load(path)))
        if {scale for scale, _, _ in parsed} != {0.5, 1.0}:
            raise SelectionError("P0-A15 candidate scales must be 0.5 and 1.0")
        manifest_hashes = {base["manifest_hash"]} | {
            data["manifest_hash"] for _, _, data in parsed
        }
        if len(manifest_hashes) != 1:
            raise SelectionError("P0-A15 manifest mismatch")
        base_accuracy = float(base["accuracy"])
        candidates: list[dict[str, Any]] = []
        for scale, path, data in sorted(parsed):
            accuracy = float(data["accuracy"])
            gain = accuracy - base_accuracy
            canonical = float(data["canonical_format_rate"])
            failures: list[str] = []
            if gain < 0.02:
                failures.append(f"gain={gain:.6f}<required=0.020000")
            if canonical < 0.95:
                failures.append(f"canonical={canonical:.6f}<required=0.950000")
            candidates.append(
                {
                    "scale": scale,
                    "served_model_id": data["served_model_id"],
                    "audit": path.relative_to(ROOT).as_posix(),
                    "audit_hash": sha256_file(path),
                    "accuracy": accuracy,
                    "gain": gain,
                    "canonical_format_rate": canonical,
                    "eligible": not failures,
                    "failures": failures,
                }
            )
        eligible = [item for item in candidates if item["eligible"]]
        selected = sorted(eligible, key=lambda item: (-item["accuracy"], item["scale"]))[0] if eligible else None
        status = "passed" if selected else "failed"
        output = resolve(args.output)
        if output != AUDIT_ROOT / "math_selection.json":
            raise SelectionError(f"Unexpected P0-A15 selection output: {output}")
        report = {
            "gate": "P0-A15-MATH-SCALED-ADAPTER-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a15_math.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "base_audit": base_path.relative_to(ROOT).as_posix(),
            "base_audit_hash": sha256_file(base_path),
            "base_accuracy": base_accuracy,
            "minimum_gain": 0.02,
            "minimum_canonical_format_rate": 0.95,
            "candidates": candidates,
            "selected_scale": selected["scale"] if selected else None,
            "selected_model_id": selected["served_model_id"] if selected else None,
            "gate300_opened": False,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(output, report)
        print(f"Wrote {output.relative_to(ROOT)}")
        print(
            f"status={status} base={base_accuracy:.6f} "
            f"selected_scale={report['selected_scale']}"
        )
        return 0 if status == "passed" else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A15 selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
