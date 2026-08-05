#!/usr/bin/env python3
"""Select one preregistered P0-A8 Code checkpoint from aggregate audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports/audit/p0a8").resolve()
EXPECTED_MANIFEST = (ROOT / "data/p0a8/code_internal_validation.jsonl").resolve()
ALLOWED_STEPS = {64, 128}


class SelectionError(RuntimeError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_audit(path: Path) -> dict[str, Any]:
    if path.suffix != ".json" or AUDIT_ROOT not in path.parents or not path.is_file():
        raise SelectionError(f"Invalid P0-A8 audit path: {display(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gate") != "P0-A8-CODE-TRAIN-ONLY-VALIDATION":
        raise SelectionError(f"Unexpected gate: {display(path)}")
    if value.get("status") != "passed":
        raise SelectionError(f"Audit is not passed: {display(path)}")
    if int(value.get("sample_count", 0)) != 86:
        raise SelectionError(f"Unexpected sample count: {display(path)}")
    if int(value.get("generation_error_count", -1)) != 0:
        raise SelectionError(f"Generation errors are nonzero: {display(path)}")
    if value.get("formal_test_loaded") is not False or value.get("humaneval_loaded") is not False:
        raise SelectionError(f"Formal-test marker is not false: {display(path)}")
    manifest = resolve(str(value.get("manifest", "")))
    if manifest != EXPECTED_MANIFEST or not manifest.is_file():
        raise SelectionError(f"Unexpected validation manifest: {display(path)}")
    if value.get("manifest_hash") != sha256_file(EXPECTED_MANIFEST):
        raise SelectionError(f"Manifest hash mismatch: {display(path)}")
    return value


def parse_candidate(value: str) -> tuple[int, Path]:
    step_text, separator, path_text = value.partition("=")
    if not separator or not step_text.isdigit():
        raise argparse.ArgumentTypeError("candidate must be STEP=PATH")
    step = int(step_text)
    if step not in ALLOWED_STEPS:
        raise argparse.ArgumentTypeError("only steps 64 and 128 are preregistered")
    return step, resolve(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_gain < 0:
        raise SelectionError("minimum gain must be non-negative")
    base_path = resolve(args.base_audit)
    base = load_audit(base_path)
    base_accuracy = float(base["accuracy"])
    candidates: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for raw in args.candidate:
        step, path = parse_candidate(raw)
        if step in seen_steps:
            raise SelectionError(f"Duplicate candidate step: {step}")
        seen_steps.add(step)
        value = load_audit(path)
        gain = float(value["accuracy"]) - base_accuracy
        failures: list[str] = []
        if gain + 1e-12 < args.minimum_gain:
            failures.append(f"code_gain={gain:.6f}<required={args.minimum_gain:.6f}")
        candidates.append(
            {
                "step": step,
                "audit": display(path),
                "audit_hash": sha256_file(path),
                "accuracy": float(value["accuracy"]),
                "canonical_format_rate": float(value["canonical_format_rate"]),
                "gain": gain,
                "eligible": not failures,
                "failures": failures,
            }
        )
    if seen_steps != ALLOWED_STEPS:
        raise SelectionError(f"Both preregistered steps are required: {sorted(seen_steps)}")
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (-float(item["accuracy"]), int(item["step"])),
    )
    selected = eligible[0] if eligible else None
    output = resolve(args.output)
    if output.suffix != ".json" or output.parent != AUDIT_ROOT:
        raise SelectionError("Selection output must be directly inside reports/audit/p0a8")
    report = {
        "gate": "P0-A8-CODE-CHECKPOINT-SELECTION",
        "check_version": "1.0",
        "created_by": "scripts/select_p0a8_code.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if selected else "failed",
        "base_audit": display(base_path),
        "base_audit_hash": sha256_file(base_path),
        "base_accuracy": base_accuracy,
        "minimum_gain": args.minimum_gain,
        "candidates": sorted(candidates, key=lambda item: int(item["step"])),
        "selected_step": int(selected["step"]) if selected else None,
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {display(output)}")
    print(
        f"status={report['status']} selected_step={report['selected_step']} "
        f"eligible={len(eligible)}/{len(candidates)}"
    )
    return 0 if selected else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A8 selection failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
