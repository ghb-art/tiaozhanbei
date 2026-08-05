#!/usr/bin/env python3
"""Select one preregistered P0-A7 NLP checkpoint from aggregate audits only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class SelectionError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"Missing audit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise SelectionError(f"Audit is not passed: {path}")
    if value.get("gate") != "P0-A7-NLP-TRAIN-ONLY-VALIDATION":
        raise SelectionError(f"Unexpected gate: {path}")
    if int(value.get("sample_count", 0)) != 256:
        raise SelectionError(f"Unexpected sample count: {path}")
    if value.get("formal_test_loaded") is not False:
        raise SelectionError(f"Formal-test marker is not false: {path}")
    return value


def parse_candidate(value: str) -> tuple[int, Path]:
    step_text, separator, path_text = value.partition("=")
    if not separator or not step_text.isdigit():
        raise argparse.ArgumentTypeError("candidate must be STEP=PATH")
    step = int(step_text)
    if step not in {94, 188}:
        raise argparse.ArgumentTypeError("only steps 94 and 188 are preregistered")
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
    base_path = resolve(args.base_audit)
    base = load(base_path)
    base_accuracy = float(base["accuracy"])
    candidates: list[dict[str, Any]] = []
    for raw in args.candidate:
        step, path = parse_candidate(raw)
        value = load(path)
        if value.get("manifest_hash") != base.get("manifest_hash"):
            raise SelectionError(f"Manifest mismatch: {path}")
        gain = float(value["accuracy"]) - base_accuracy
        failures: list[str] = []
        if int(value.get("generation_error_count", 0)) != 0:
            failures.append("generation_errors_nonzero")
        if gain + 1e-12 < args.minimum_gain:
            failures.append(f"nlp_gain={gain:.6f}<required={args.minimum_gain:.6f}")
        candidates.append(
            {
                "step": step,
                "audit": path.relative_to(ROOT).as_posix(),
                "audit_hash": sha256_file(path),
                "accuracy": float(value["accuracy"]),
                "gain": gain,
                "eligible": not failures,
                "failures": failures,
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    eligible.sort(key=lambda item: (-float(item["accuracy"]), int(item["step"])))
    selected = eligible[0] if eligible else None
    output = resolve(args.output)
    if output.parent.resolve() != (ROOT / "reports/audit/p0a7").resolve():
        raise SelectionError("Selection output must be inside reports/audit/p0a7")
    report = {
        "gate": "P0-A7-NLP-CHECKPOINT-SELECTION",
        "check_version": "1.0",
        "created_by": "scripts/select_p0a7_nlp.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if selected else "failed",
        "base_audit": base_path.relative_to(ROOT).as_posix(),
        "base_audit_hash": sha256_file(base_path),
        "base_accuracy": base_accuracy,
        "minimum_gain": args.minimum_gain,
        "candidates": sorted(candidates, key=lambda item: int(item["step"])),
        "selected_step": int(selected["step"]) if selected else None,
    }
    report["report_hash"] = sha256_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"Wrote {output.relative_to(ROOT)}")
    print(
        f"status={report['status']} selected_step={report['selected_step']} "
        f"eligible={len(eligible)}/{len(candidates)}"
    )
    return 0 if selected else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as exc:
        print(f"P0-A7 selection failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
