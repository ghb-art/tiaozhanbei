#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a4_distillation.json"
DATASET_TO_RATIO = {
    "gsm8k": "math_ratio",
    "humaneval": "code_ratio",
    "cmmlu": "nlp_ratio",
}


class GateError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GateError(f"Missing trace: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GateError(f"Non-object trace row {display_path(path)}:{line_number}")
            rows.append(value)
    if not rows:
        raise GateError(f"Empty trace: {display_path(path)}")
    return rows


def index_trace(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        dataset = str(row.get("dataset_key", ""))
        if not sample_id or dataset not in DATASET_TO_RATIO:
            raise GateError(f"{label} contains an invalid sample identity")
        if sample_id in indexed:
            raise GateError(f"{label} duplicate sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def compare(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    expected_counts: dict[str, int],
    min_ratio: float,
    min_macro: float,
    max_errors: int,
) -> dict[str, Any]:
    baseline = index_trace(baseline_rows, "baseline")
    candidate = index_trace(candidate_rows, "candidate")
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    missing = sorted(baseline_ids - candidate_ids)
    extra = sorted(candidate_ids - baseline_ids)

    baseline_counts = Counter(str(row["dataset_key"]) for row in baseline.values())
    candidate_counts = Counter(str(row["dataset_key"]) for row in candidate.values())
    baseline_correct = Counter(
        str(row["dataset_key"]) for row in baseline.values() if row.get("correct") is True
    )
    candidate_correct = Counter(
        str(row["dataset_key"]) for row in candidate.values() if row.get("correct") is True
    )
    baseline_errors = sum(bool(row.get("generation_error")) for row in baseline.values())
    candidate_errors = sum(bool(row.get("generation_error")) for row in candidate.values())
    prompt_mismatch_count = sum(
        1
        for sample_id in baseline_ids & candidate_ids
        if baseline[sample_id].get("prompt_hash")
        and candidate[sample_id].get("prompt_hash")
        and baseline[sample_id].get("prompt_hash") != candidate[sample_id].get("prompt_hash")
    )

    baseline_accuracy: dict[str, float] = {}
    candidate_accuracy: dict[str, float] = {}
    ratios: dict[str, float] = {}
    failures: dict[str, dict[str, float]] = {}
    for dataset, ratio_name in DATASET_TO_RATIO.items():
        denominator = baseline_counts[dataset]
        numerator = candidate_counts[dataset]
        baseline_acc = baseline_correct[dataset] / denominator if denominator else 0.0
        candidate_acc = candidate_correct[dataset] / numerator if numerator else 0.0
        ratio = candidate_acc / baseline_acc if baseline_acc else 0.0
        baseline_accuracy[dataset] = baseline_acc
        candidate_accuracy[dataset] = candidate_acc
        ratios[ratio_name] = ratio
        if ratio < min_ratio:
            failures[ratio_name] = {"actual": ratio, "required": min_ratio}
    capped_macro = sum(min(value, 1.0) for value in ratios.values()) / len(ratios)
    if capped_macro < min_macro:
        failures["capped_macro_ratio"] = {"actual": capped_macro, "required": min_macro}

    counts_ok = dict(baseline_counts) == expected_counts and dict(candidate_counts) == expected_counts
    identity_ok = not missing and not extra
    passed = (
        counts_ok
        and identity_ok
        and prompt_mismatch_count == 0
        and baseline_errors == 0
        and candidate_errors <= max_errors
        and not failures
    )
    return {
        "passed": passed,
        "expected_counts": expected_counts,
        "baseline_counts": dict(sorted(baseline_counts.items())),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "matched_sample_ids": identity_ok,
        "missing_sample_count": len(missing),
        "missing_sample_ids_hash": sha256_text("\n".join(missing) + "\n"),
        "extra_sample_count": len(extra),
        "extra_sample_ids_hash": sha256_text("\n".join(extra) + "\n"),
        "prompt_mismatch_count": prompt_mismatch_count,
        "baseline_accuracy_by_dataset": baseline_accuracy,
        "candidate_accuracy_by_dataset": candidate_accuracy,
        "ratios": ratios,
        "capped_macro_ratio": capped_macro,
        "ratio_failures": failures,
        "baseline_generation_error_count": baseline_errors,
        "generation_error_count": candidate_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate-only P0-A4 retention gate.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gate", choices=("smoke96", "selection170", "official_full"), required=True)
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        gate = dict(config["gates"][args.gate])
        expected = (
            {key: int(value) for key, value in config["data"]["official_counts"].items()}
            if args.gate == "official_full"
            else {key: int(value) for key, value in gate["expected_counts"].items()}
        )
        baseline_path = resolve_path(args.baseline_trace)
        candidate_path = resolve_path(args.candidate_trace)
        result = compare(
            load_trace(baseline_path),
            load_trace(candidate_path),
            expected,
            float(gate["min_ratio_per_task"]),
            float(gate["min_capped_macro_ratio"]),
            int(gate["max_generation_errors"]),
        )
        audit = {
            "gate": f"P0-A4-{args.gate.upper()}-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a4_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if result.pop("passed") else "failed",
            "feedback_policy": (
                config["gates"]["selection170"]["feedback"]
                if args.gate == "selection170"
                else "aggregate_only"
            ),
            "candidate_name": args.candidate_name,
            "baseline_trace": display_path(baseline_path),
            "baseline_trace_hash": sha256_file(baseline_path),
            "candidate_trace": display_path(candidate_path),
            "candidate_trace_hash": sha256_file(candidate_path),
            "min_ratio_per_task": float(gate["min_ratio_per_task"]),
            "min_capped_macro_ratio": float(gate["min_capped_macro_ratio"]),
            **result,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError, GateError, json.JSONDecodeError) as exc:
        print(f"P0-A4 retention gate failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {display_path(output)}")
    print(
        f"status={audit['status']} ratios={audit['ratios']} "
        f"macro={audit['capped_macro_ratio']:.6f}"
    )
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
