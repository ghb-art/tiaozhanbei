#!/usr/bin/env python3
"""Select a P0-A6 checkpoint using frozen quick-validation summaries only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (ROOT / "reports" / "audit" / "p0a6").resolve()
QUICK_MANIFEST = (ROOT / "data" / "p0a6" / "quick_validation.jsonl").resolve()
FULL_MANIFEST = (ROOT / "data" / "p0a6" / "full_validation.jsonl").resolve()
DOMAINS = ("math", "code", "nlp")
THRESHOLDS = {"math": -0.01, "code": 0.03, "nlp": 0.03}
TOLERANCE = 1e-12


class SelectionError(RuntimeError):
    """Raised when selection inputs violate the frozen P0-A6 protocol."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_audit_path(path: Path, label: str, must_exist: bool = True) -> None:
    if not is_relative_to(path, AUDIT_ROOT):
        raise SelectionError(
            f"{label} must be inside reports/audit/p0a6: {display_path(path)}"
        )
    if path.suffix != ".json":
        raise SelectionError(f"{label} must be a JSON file: {display_path(path)}")
    if must_exist and not path.is_file():
        raise SelectionError(f"Missing {label}: {display_path(path)}")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SelectionError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"{label} is not a JSON object")
    return value


def parse_accuracy(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SelectionError(f"{label}.accuracy_by_domain is not an object")
    try:
        accuracy = {domain: float(value[domain]) for domain in DOMAINS}
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError(
            f"{label}.accuracy_by_domain must contain numeric math/code/nlp"
        ) from exc
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in accuracy.values()):
        raise SelectionError(f"{label} contains an invalid accuracy: {accuracy}")
    return accuracy


def parse_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise SelectionError(f"{label}.expected_counts is not an object")
    try:
        counts = {domain: int(value[domain]) for domain in DOMAINS}
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError(
            f"{label}.expected_counts must contain integer math/code/nlp"
        ) from exc
    if any(count <= 0 for count in counts.values()):
        raise SelectionError(f"{label} contains a non-positive count: {counts}")
    return counts


def validate_eval_audit(
    path: Path,
    label: str,
    expected_manifest: Path = QUICK_MANIFEST,
) -> dict[str, Any]:
    require_audit_path(path, label)
    audit = read_json(path, label)
    if audit.get("gate") != "P0-A6-INTERNAL-EVAL":
        raise SelectionError(f"{label} is not a P0-A6 internal evaluation")
    if audit.get("created_by") != "scripts/evaluate_p0a6_internal.py":
        raise SelectionError(f"{label} has an unexpected evaluator identity")
    if audit.get("status") != "passed":
        raise SelectionError(f"{label} status is not passed")
    try:
        generation_errors = int(audit.get("generation_error_count", -1))
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} has an invalid generation error count") from exc
    if generation_errors != 0:
        raise SelectionError(f"{label} generation_error_count is not zero")
    manifest = resolve_path(str(audit.get("manifest", "")))
    if manifest != expected_manifest:
        raise SelectionError(
            f"{label} must use {display_path(expected_manifest)}, got "
            f"{display_path(manifest)}"
        )
    manifest_hash = str(audit.get("manifest_hash", ""))
    if not manifest_hash or not re_full_sha256(manifest_hash):
        raise SelectionError(f"{label} has no valid manifest_hash")
    if not expected_manifest.is_file():
        raise SelectionError(
            f"Frozen validation manifest is missing: {display_path(expected_manifest)}"
        )
    if sha256_file(expected_manifest) != manifest_hash:
        raise SelectionError(
            f"{label} manifest_hash does not match the frozen validation manifest"
        )
    accuracy = parse_accuracy(audit.get("accuracy_by_domain"), label)
    expected_counts = parse_counts(audit.get("expected_counts"), label)
    actual_counts = parse_counts(audit.get("actual_counts"), label)
    if actual_counts != expected_counts:
        raise SelectionError(f"{label} actual_counts differ from expected_counts")
    computed_macro = mean(accuracy.values())
    try:
        reported_macro = float(audit.get("macro_accuracy"))
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} has an invalid macro_accuracy") from exc
    if not math.isclose(computed_macro, reported_macro, abs_tol=1e-9, rel_tol=1e-9):
        raise SelectionError(
            f"{label} macro_accuracy does not match domain accuracies"
        )
    return {
        "path": path,
        "file_hash": sha256_file(path),
        "manifest": manifest,
        "manifest_hash": manifest_hash,
        "expected_counts": expected_counts,
        "accuracy": accuracy,
        "macro_accuracy": computed_macro,
        "candidate_name": str(audit.get("candidate_name", "")),
        "served_model_id": str(audit.get("served_model_id", "")),
    }


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def parse_candidate(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise SelectionError(f"Candidate must use STEP=PATH syntax: {value!r}")
    raw_step, raw_path = value.split("=", 1)
    try:
        step = int(raw_step)
    except ValueError as exc:
        raise SelectionError(f"Candidate step is not an integer: {raw_step!r}") from exc
    if step <= 0:
        raise SelectionError(f"Candidate step must be positive: {step}")
    if not raw_path.strip():
        raise SelectionError(f"Candidate path is empty for step {step}")
    return step, resolve_path(raw_path)


def gains_from_base(
    base: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    return {domain: candidate[domain] - base[domain] for domain in DOMAINS}


def qualifies(gains: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for domain in DOMAINS:
        if gains[domain] + TOLERANCE < THRESHOLDS[domain]:
            failures.append(
                f"{domain}_gain={gains[domain]:.12g}<required={THRESHOLDS[domain]:.12g}"
            )
    return not failures, failures


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a P0-A6 checkpoint from frozen quick-validation audits."
    )
    parser.add_argument("--base-audit", required=True)
    parser.add_argument(
        "--validation-manifest",
        default=str(QUICK_MANIFEST),
        help="Frozen quick or full internal-validation manifest used by every audit.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="STEP=PATH",
        help="Repeat for each candidate checkpoint evaluation audit.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_path = resolve_path(args.base_audit)
        output_path = resolve_path(args.output)
        expected_manifest = resolve_path(args.validation_manifest)
        if expected_manifest not in {QUICK_MANIFEST, FULL_MANIFEST}:
            raise SelectionError(
                "Validation manifest must be the frozen P0-A6 quick or full manifest"
            )
        require_audit_path(output_path, "output", must_exist=False)
        base = validate_eval_audit(base_path, "base audit", expected_manifest)
        parsed_candidates = [parse_candidate(value) for value in args.candidate]
        steps = [step for step, _ in parsed_candidates]
        if len(steps) != len(set(steps)):
            raise SelectionError(f"Duplicate candidate steps: {steps}")
        input_paths = {base_path, *(path for _, path in parsed_candidates)}
        if output_path in input_paths:
            raise SelectionError("Output path must differ from all input audits")

        candidate_results: list[dict[str, Any]] = []
        for step, path in parsed_candidates:
            candidate = validate_eval_audit(
                path, f"candidate step {step}", expected_manifest
            )
            if candidate["manifest_hash"] != base["manifest_hash"]:
                raise SelectionError(
                    f"Candidate step {step} does not use the same frozen manifest hash"
                )
            if candidate["expected_counts"] != base["expected_counts"]:
                raise SelectionError(
                    f"Candidate step {step} has different expected counts"
                )
            gains = gains_from_base(base["accuracy"], candidate["accuracy"])
            eligible, failures = qualifies(gains)
            candidate_results.append(
                {
                    "step": step,
                    "audit": display_path(path),
                    "audit_hash": candidate["file_hash"],
                    "candidate_name": candidate["candidate_name"],
                    "served_model_id": candidate["served_model_id"],
                    "accuracy_by_domain": candidate["accuracy"],
                    "macro_accuracy": candidate["macro_accuracy"],
                    "gain_by_domain": gains,
                    "eligible": eligible,
                    "failures": failures,
                }
            )

        eligible = [item for item in candidate_results if item["eligible"]]
        selected = sorted(
            eligible,
            key=lambda item: (-float(item["macro_accuracy"]), int(item["step"])),
        )[0] if eligible else None
        created_ts = datetime.now(timezone.utc).isoformat()
        report: dict[str, Any] = {
            "gate": "P0-A6-CHECKPOINT-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a6_checkpoint.py",
            "created_ts": created_ts,
            "status": "passed" if selected else "failed",
            "selection_policy": {
                "math_minimum_gain": THRESHOLDS["math"],
                "code_minimum_gain": THRESHOLDS["code"],
                "nlp_minimum_gain": THRESHOLDS["nlp"],
                "ranking": ["macro_accuracy_desc", "step_asc"],
            },
            "manifest": display_path(base["manifest"]),
            "validation_scope": (
                "quick" if expected_manifest == QUICK_MANIFEST else "full"
            ),
            "manifest_hash": base["manifest_hash"],
            "expected_counts": base["expected_counts"],
            "base_audit": display_path(base_path),
            "base_audit_hash": base["file_hash"],
            "base_accuracy_by_domain": base["accuracy"],
            "base_macro_accuracy": base["macro_accuracy"],
            "candidate_count": len(candidate_results),
            "eligible_candidate_count": len(eligible),
            "candidates": sorted(candidate_results, key=lambda item: item["step"]),
            "selected_step": selected["step"] if selected else None,
            "selected_path": selected["audit"] if selected else "",
            "selected_accuracy_by_domain": (
                selected["accuracy_by_domain"] if selected else {}
            ),
            "selected_macro_accuracy": selected["macro_accuracy"] if selected else None,
            "selected_gain_by_domain": selected["gain_by_domain"] if selected else {},
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(output_path, report)
        print(f"Wrote {display_path(output_path)}")
        print(
            f"status={report['status']} selected_step={report['selected_step']} "
            f"eligible={len(eligible)}/{len(candidate_results)}"
        )
        return 0 if selected else 1
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A6 checkpoint selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
