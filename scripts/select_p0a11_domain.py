#!/usr/bin/env python3
"""Select one of two preregistered P0-A11 train-only checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "p0a11": {
        "audit_root": (ROOT / "reports/audit/p0a11").resolve(),
        "validation_gate": "P0-A11-TRAIN-ONLY-VALIDATION",
        "selection_gate": "P0-A11-DOMAIN-CHECKPOINT-SELECTION",
    },
    "p0a12": {
        "audit_root": (ROOT / "reports/audit/p0a12").resolve(),
        "validation_gate": "P0-A12-EXTERNAL-MATH-VALIDATION",
        "selection_gate": "P0-A12-MATH-CHECKPOINT-SELECTION",
    },
}


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    return path.relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, domain: str, protocol: str) -> dict[str, Any]:
    audit_root = PROTOCOLS[protocol]["audit_root"]
    if path.suffix != ".json" or audit_root not in path.parents or not path.is_file():
        raise SelectionError(f"Invalid audit path: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gate") != PROTOCOLS[protocol]["validation_gate"]:
        raise SelectionError(f"Unexpected gate: {path}")
    if value.get("status") != "passed" or value.get("domain") != domain:
        raise SelectionError(f"Rejected audit: {path}")
    if int(value.get("sample_count", 0)) <= 0:
        raise SelectionError(f"Invalid sample count: {path}")
    if int(value.get("generation_error_count", -1)) != 0:
        raise SelectionError(f"Generation errors: {path}")
    if value.get("gate300_loaded") is not False or value.get("formal_full_loaded") is not False:
        raise SelectionError(f"Forbidden data marker: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="p0a11")
    parser.add_argument("--domain", required=True, choices=("math", "code"))
    parser.add_argument("--steps", required=True)
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--minimum-gain", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed = {int(value) for value in args.steps.split(",") if value.strip()}
    if len(allowed) != 2:
        raise SelectionError("Exactly two preregistered steps are required")
    base_path = resolve(args.base_audit)
    base = load(base_path, args.domain, args.protocol)
    base_accuracy = float(base["accuracy"])
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in args.candidate:
        step_text, separator, path_text = raw.partition("=")
        if not separator or not step_text.isdigit():
            raise SelectionError("Candidate must be STEP=PATH")
        step = int(step_text)
        if step not in allowed or step in seen:
            raise SelectionError(f"Unregistered or duplicate step: {step}")
        seen.add(step)
        path = resolve(path_text)
        value = load(path, args.domain, args.protocol)
        if value.get("manifest_hash") != base.get("manifest_hash"):
            raise SelectionError(f"Manifest mismatch: {path}")
        if int(value["sample_count"]) != int(base["sample_count"]):
            raise SelectionError(f"Sample count mismatch: {path}")
        gain = float(value["accuracy"]) - base_accuracy
        eligible = gain + 1e-12 >= args.minimum_gain
        candidates.append({
            "step": step,
            "audit": display(path),
            "audit_hash": sha256_file(path),
            "accuracy": float(value["accuracy"]),
            "canonical_format_rate": float(value["canonical_format_rate"]),
            "gain": gain,
            "eligible": eligible,
            "failures": [] if eligible else [f"gain={gain:.6f}<required={args.minimum_gain:.6f}"],
        })
    if seen != allowed:
        raise SelectionError(f"Missing steps: {sorted(allowed-seen)}")
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (-float(item["accuracy"]), int(item["step"])),
    )
    selected = eligible[0] if eligible else None
    output = resolve(args.output)
    audit_root = PROTOCOLS[args.protocol]["audit_root"]
    if output.parent != audit_root or output.suffix != ".json":
        raise SelectionError(
            f"Output must be directly inside reports/audit/{args.protocol}"
        )
    report = {
        "gate": PROTOCOLS[args.protocol]["selection_gate"],
        "check_version": "1.0",
        "created_by": "scripts/select_p0a11_domain.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if selected else "failed",
        "domain": args.domain,
        "allowed_steps": sorted(allowed),
        "minimum_gain": args.minimum_gain,
        "base_audit": display(base_path),
        "base_accuracy": base_accuracy,
        "candidates": sorted(candidates, key=lambda item: int(item["step"])),
        "selected_step": int(selected["step"]) if selected else None,
    }
    report["report_hash"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {display(output)}")
    print(f"status={report['status']} selected_step={report['selected_step']}")
    return 0 if selected else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A11 selection failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
