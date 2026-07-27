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
DATASET_TO_RATIO = {
    "gsm8k": "math_ratio",
    "humaneval": "code_ratio",
    "cmmlu": "nlp_ratio",
}
FROZEN_DATASET_COUNTS = {"gsm8k": 64, "humaneval": 42, "cmmlu": 64}


class RetentionError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
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
        raise RetentionError(f"Missing trace: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RetentionError(f"Trace row {line_number} is not an object: {display_path(path)}")
            rows.append(value)
    if not rows:
        raise RetentionError(f"Empty trace: {display_path(path)}")
    return rows


def index_trace(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        dataset = str(row.get("dataset_key", "")).strip()
        if not sample_id or dataset not in DATASET_TO_RATIO:
            raise RetentionError(f"{label} trace contains an invalid sample identity")
        if sample_id in indexed:
            raise RetentionError(f"{label} trace contains duplicate sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def compare_traces(
    teacher_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    min_ratio: float,
) -> dict[str, Any]:
    teacher = index_trace(teacher_rows, "teacher")
    candidate = index_trace(candidate_rows, "candidate")
    missing_from_candidate = sorted(set(teacher) - set(candidate))
    extra_in_candidate = sorted(set(candidate) - set(teacher))
    dataset_mismatches = sorted(
        sample_id
        for sample_id in set(teacher) & set(candidate)
        if str(teacher[sample_id].get("dataset_key"))
        != str(candidate[sample_id].get("dataset_key"))
    )

    teacher_counts: Counter[str] = Counter()
    teacher_correct: Counter[str] = Counter()
    teacher_generation_errors: Counter[str] = Counter()
    for row in teacher.values():
        dataset = str(row["dataset_key"])
        teacher_counts[dataset] += 1
        if row.get("correct") is True:
            teacher_correct[dataset] += 1
        if row.get("generation_error"):
            teacher_generation_errors[dataset] += 1

    candidate_counts: Counter[str] = Counter()
    candidate_correct: Counter[str] = Counter()
    candidate_generation_errors: Counter[str] = Counter()
    for row in candidate.values():
        dataset = str(row["dataset_key"])
        candidate_counts[dataset] += 1
        if row.get("correct") is True:
            candidate_correct[dataset] += 1
        if row.get("generation_error"):
            candidate_generation_errors[dataset] += 1

    teacher_accuracy: dict[str, float] = {}
    candidate_accuracy: dict[str, float] = {}
    ratios: dict[str, float] = {}
    ratio_failures: dict[str, dict[str, float]] = {}
    for dataset, ratio_name in DATASET_TO_RATIO.items():
        teacher_acc = (
            teacher_correct[dataset] / teacher_counts[dataset]
            if teacher_counts[dataset]
            else 0.0
        )
        candidate_acc = (
            candidate_correct[dataset] / candidate_counts[dataset]
            if candidate_counts[dataset]
            else 0.0
        )
        ratio = candidate_acc / teacher_acc if teacher_acc else 0.0
        teacher_accuracy[dataset] = teacher_acc
        candidate_accuracy[dataset] = candidate_acc
        ratios[ratio_name] = ratio
        if ratio < min_ratio:
            ratio_failures[ratio_name] = {"actual": ratio, "required": min_ratio}

    capped_macro_ratio = sum(min(value, 1.0) for value in ratios.values()) / len(ratios)
    identity_ok = not missing_from_candidate and not extra_in_candidate and not dataset_mismatches
    complete_frozen_dev = (
        dict(teacher_counts) == FROZEN_DATASET_COUNTS
        and dict(candidate_counts) == FROZEN_DATASET_COUNTS
    )
    teacher_error_count = sum(teacher_generation_errors.values())
    candidate_error_count = sum(candidate_generation_errors.values())
    passed = (
        identity_ok
        and complete_frozen_dev
        and not ratio_failures
        and teacher_error_count == 0
        and candidate_error_count == 0
        and capped_macro_ratio >= min_ratio
    )
    return {
        "status": "passed" if passed else "failed",
        "sample_count": len(candidate_rows),
        "teacher_sample_count": len(teacher_rows),
        "expected_dataset_counts": FROZEN_DATASET_COUNTS,
        "teacher_dataset_counts": dict(sorted(teacher_counts.items())),
        "dataset_counts": dict(sorted(candidate_counts.items())),
        "complete_frozen_dev": complete_frozen_dev,
        "teacher_correct_counts": dict(sorted(teacher_correct.items())),
        "candidate_correct_counts": dict(sorted(candidate_correct.items())),
        "teacher_accuracy_by_dataset": teacher_accuracy,
        "candidate_accuracy_by_dataset": candidate_accuracy,
        "ratios": ratios,
        "capped_macro_ratio": capped_macro_ratio,
        "min_ratio": min_ratio,
        "ratio_failures": ratio_failures,
        "teacher_generation_error_count": teacher_error_count,
        "teacher_generation_errors_by_dataset": dict(sorted(teacher_generation_errors.items())),
        "generation_error_count": candidate_error_count,
        "generation_errors_by_dataset": dict(sorted(candidate_generation_errors.items())),
        "missing_from_candidate_count": len(missing_from_candidate),
        "missing_from_candidate_hash": sha256_text("\n".join(missing_from_candidate) + "\n"),
        "extra_in_candidate_count": len(extra_in_candidate),
        "extra_in_candidate_hash": sha256_text("\n".join(extra_in_candidate) + "\n"),
        "dataset_mismatch_count": len(dataset_mismatches),
        "dataset_mismatch_hash": sha256_text("\n".join(dataset_mismatches) + "\n"),
        "matched_sample_ids": identity_ok,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an edge candidate with Qwen14 on the identical frozen 170-row Dev set."
    )
    parser.add_argument("--teacher-trace", required=True)
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.min_ratio <= 1:
        print("--min-ratio must be in (0, 1]", file=sys.stderr)
        return 2
    teacher_path = resolve_path(args.teacher_trace)
    candidate_path = resolve_path(args.candidate_trace)
    output_path = resolve_path(args.output)
    try:
        result = compare_traces(
            load_trace(teacher_path),
            load_trace(candidate_path),
            args.min_ratio,
        )
        report = {
            "gate": "P0-A3-matched-dev-retention",
            "check_version": "1.0",
            "created_by": "scripts/summarize_edge_candidate_dev.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "candidate_name": args.candidate_name,
            "teacher_trace": display_path(teacher_path),
            "teacher_trace_sha256": sha256_file(teacher_path),
            "candidate_trace": display_path(candidate_path),
            "candidate_trace_sha256": sha256_file(candidate_path),
            "formal_test_labels_used": False,
            **result,
        }
        report["report_hash"] = sha256_text(
            json.dumps(
                {key: value for key, value in report.items() if key != "report_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {display_path(output_path)}")
        print(
            f"status={report['status']} ratios={report['ratios']} "
            f"macro={report['capped_macro_ratio']:.6f}"
        )
        return 0 if report["passed"] else 1
    except (json.JSONDecodeError, OSError, RetentionError) as exc:
        print(f"Retention comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
