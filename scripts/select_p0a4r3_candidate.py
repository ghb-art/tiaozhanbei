#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4r3_shared_distillation.json"


class SelectionError(RuntimeError):
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


def load_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"Missing audit: {display_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionError(f"Invalid audit object: {display_path(path)}")
    return value


def candidate_spec(value: str) -> tuple[int, Path, Path]:
    fields = value.split(":", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must be INDEX:EVAL_AUDIT:MERGED_MODEL_DIR"
        )
    index = int(fields[0])
    return index, resolve_path(fields[1]), resolve_path(fields[2])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select at most one of two preregistered P0-A4R3 shared candidates."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--baseline-audit",
        default="reports/audit/gate_p0a4r3_v1_train_only.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=candidate_spec,
        required=True,
        help="INDEX:EVAL_AUDIT:MERGED_MODEL_DIR; repeat at most twice",
    )
    parser.add_argument(
        "--output",
        default="reports/audit/gate_p0a4r3_candidate_selection.json",
    )
    args = parser.parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        selection = config["selection"]
        limit = int(config["policy"]["max_preregistered_candidates"])
        if len(args.candidate) > limit:
            raise SelectionError(
                f"Candidate count {len(args.candidate)} exceeds preregistered limit {limit}"
            )
        indices = [item[0] for item in args.candidate]
        if len(set(indices)) != len(indices) or any(index not in (1, 2) for index in indices):
            raise SelectionError("Candidate indices must be unique and drawn from {1, 2}")
        baseline_path = resolve_path(args.baseline_audit)
        baseline = load_audit(baseline_path)
        if baseline.get("status") != "passed":
            raise SelectionError("Baseline evaluation audit is not passed")
        baseline_accuracy = {
            key: float(value)
            for key, value in baseline.get("accuracy_by_dataset", {}).items()
        }
        if set(baseline_accuracy) != {"gsm8k", "humaneval", "cmmlu"}:
            raise SelectionError("Baseline audit does not contain the three train-only tasks")
        validation_hash = str(baseline.get("validation_data_sha256", ""))
        if not validation_hash:
            raise SelectionError("Baseline audit is missing validation_data_sha256")

        records: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        for index, audit_path, merged_dir in args.candidate:
            audit = load_audit(audit_path)
            accuracy = {
                key: float(value)
                for key, value in audit.get("accuracy_by_dataset", {}).items()
            }
            reasons: list[str] = []
            if audit.get("status") != "passed":
                reasons.append("candidate_evaluation_not_passed")
            if str(audit.get("validation_data_sha256", "")) != validation_hash:
                reasons.append("validation_identity_mismatch")
            if int(audit.get("generation_error_count", -1)) != int(
                selection["require_generation_errors"]
            ):
                reasons.append("generation_error")
            if set(accuracy) != set(baseline_accuracy):
                reasons.append("missing_task_accuracy")
            if not merged_dir.is_dir():
                reasons.append("missing_merged_model")
            math_ratio = (
                accuracy.get("gsm8k", 0.0) / baseline_accuracy["gsm8k"]
                if baseline_accuracy["gsm8k"] > 0
                else 0.0
            )
            deltas = {
                task: accuracy.get(task, 0.0) - baseline_accuracy[task]
                for task in baseline_accuracy
            }
            code_nlp_gain = (deltas["humaneval"] + deltas["cmmlu"]) / 2.0
            if math_ratio < float(selection["math_min_baseline_ratio"]):
                reasons.append("math_replay_regression")
            if deltas["humaneval"] < float(selection["code_min_baseline_delta"]):
                reasons.append("code_regression")
            if deltas["cmmlu"] < float(selection["nlp_min_baseline_delta"]):
                reasons.append("nlp_regression")
            if code_nlp_gain < float(selection["minimum_code_nlp_macro_gain"]):
                reasons.append("insufficient_code_nlp_gain")
            record = {
                "candidate_index": index,
                "rank": (
                    int(config["training"]["student_shared"]["candidate_overrides"]["2"]["lora_rank"])
                    if index == 2
                    else int(config["training"]["student_shared"]["lora_rank"])
                ),
                "eval_audit": display_path(audit_path),
                "eval_audit_hash": sha256_file(audit_path),
                "merged_model": display_path(merged_dir),
                "accuracy_by_dataset": accuracy,
                "math_baseline_ratio": math_ratio,
                "delta_by_dataset": deltas,
                "code_nlp_macro_gain": code_nlp_gain,
                "minimum_code_nlp_delta": min(deltas["humaneval"], deltas["cmmlu"]),
                "eligible": not reasons,
                "reasons": reasons,
            }
            records.append(record)
            if not reasons:
                eligible.append(record)
        eligible.sort(
            key=lambda item: (
                -float(item["minimum_code_nlp_delta"]),
                -float(item["code_nlp_macro_gain"]),
                int(item["rank"]),
            )
        )
        selected = eligible[0] if eligible else None
        report = {
            "gate": "P0-A4R3-CANDIDATE-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a4r3_candidate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "policy": config["policy"],
            "selection_policy": selection,
            "baseline_audit": display_path(baseline_path),
            "baseline_audit_hash": sha256_file(baseline_path),
            "validation_data_sha256": validation_hash,
            "baseline_accuracy": baseline_accuracy,
            "candidate_count": len(records),
            "candidates": records,
            "selected_candidate_index": (
                int(selected["candidate_index"]) if selected else None
            ),
            "selected_rank": int(selected["rank"]) if selected else None,
            "selected_model": str(selected["merged_model"]) if selected else "",
            "formal_test_reference_count": 0,
            "old_code42_item_feedback_used": False,
            "smoke96_item_feedback_used": False,
            "selection170_feedback_used": False,
            "formal_full_feedback_used": False,
        }
        report["report_hash"] = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"P0-A4R3 selection status={report['status']} "
            f"selected={report['selected_candidate_index']}",
            flush=True,
        )
        return 0 if selected else 1
    except (SelectionError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"P0-A4R3 selection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
