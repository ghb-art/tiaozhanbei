#!/usr/bin/env python3
"""Compute aggregate-only official-full retention for P0-A43."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a43_edge_full.json"
BASELINE = ROOT / "reports/sealed/p0a4/baseline14b_awq_full.jsonl"
CANDIDATE = ROOT / "reports/sealed/p0a43/edge_best_router_q4_full.jsonl"
OUTPUT = ROOT / "reports/audit/gate_p0a43_edge_best_router_q4_full_retention.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GateError(f"Missing trace: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    try:
        if OUTPUT.exists():
            raise GateError("P0-A43 retention already exists; repeat is forbidden")
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        formal = config["formal_test"]
        expected = {key: int(value) for key, value in formal["counts"].items()}
        baseline_rows = load(BASELINE)
        candidate_rows = load(CANDIDATE)
        baseline = {str(row["sample_id"]): row for row in baseline_rows}
        candidate = {str(row["sample_id"]): row for row in candidate_rows}
        if len(baseline) != len(baseline_rows) or len(candidate) != len(candidate_rows):
            raise GateError("Duplicate sample ids in full trace")
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        prompt_mismatches = sum(
            baseline[sample_id].get("prompt_hash") != candidate[sample_id].get("prompt_hash")
            for sample_id in set(baseline) & set(candidate)
        )
        base_counts = Counter(str(row["dataset_key"]) for row in baseline_rows)
        candidate_counts = Counter(str(row["dataset_key"]) for row in candidate_rows)
        if dict(base_counts) != expected or dict(candidate_counts) != expected:
            raise GateError(f"Full counts changed: baseline={dict(base_counts)} candidate={dict(candidate_counts)}")
        base_correct = Counter(str(row["dataset_key"]) for row in baseline_rows if row.get("correct") is True)
        candidate_correct = Counter(str(row["dataset_key"]) for row in candidate_rows if row.get("correct") is True)
        base_accuracy = {task: base_correct[task] / base_counts[task] for task in TASKS}
        candidate_accuracy = {task: candidate_correct[task] / candidate_counts[task] for task in TASKS}
        ratios = {
            task: min(candidate_accuracy[task] / base_accuracy[task], 1.0)
            for task in TASKS
        }
        macro = sum(ratios.values()) / len(ratios)
        errors = sum(bool(row.get("generation_error")) for row in candidate_rows)
        minimum = float(formal["minimum_retention_per_domain"])
        macro_minimum = float(formal["minimum_capped_macro_retention"])
        identity_ok = not missing and not extra and prompt_mismatches == 0
        passed = identity_ok and errors == 0 and all(value >= minimum for value in ratios.values()) and macro >= macro_minimum
        report = {
            "gate": "P0-A43-EDGE-BEST-ROUTER-OFFICIAL-FULL-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a43_retention_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "decision": "meets_full_retention_requirement" if passed else "does_not_meet_full_retention_requirement",
            "feedback_policy": "aggregate_only_no_retraining",
            "candidate_name": "edge-best-router-q4-k-m-q8-kv",
            "baseline_trace": BASELINE.relative_to(ROOT).as_posix(),
            "baseline_trace_hash": sha256_file(BASELINE),
            "candidate_trace": CANDIDATE.relative_to(ROOT).as_posix(),
            "candidate_trace_hash": sha256_file(CANDIDATE),
            "expected_counts": expected,
            "baseline_counts": dict(base_counts),
            "candidate_counts": dict(candidate_counts),
            "matched_sample_ids": not missing and not extra,
            "missing_sample_count": len(missing),
            "missing_sample_ids_hash": sha256_text("\n".join(missing) + "\n"),
            "extra_sample_count": len(extra),
            "extra_sample_ids_hash": sha256_text("\n".join(extra) + "\n"),
            "prompt_mismatch_count": prompt_mismatches,
            "baseline_correct_counts": dict(base_correct),
            "candidate_correct_counts": dict(candidate_correct),
            "baseline_accuracy_by_dataset": base_accuracy,
            "candidate_accuracy_by_dataset": candidate_accuracy,
            "retention_ratios": ratios,
            "capped_macro_ratio": macro,
            "minimum_retention_per_domain": minimum,
            "minimum_capped_macro_retention": macro_minimum,
            "baseline_generation_error_count": sum(bool(row.get("generation_error")) for row in baseline_rows),
            "generation_error_count": errors,
            "formal_full_completed": True,
            "item_level_feedback_allowed_for_training": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, GateError) as exc:
        print(f"P0-A43 retention failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"status={report['status']} ratios={ratios} macro={macro:.6f}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
