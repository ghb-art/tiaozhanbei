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
DEFAULT_CONFIG = ROOT / "configs/p0a5_capability.json"
DOMAINS = ("math", "code", "nlp")


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


def read_trace(path: Path, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or str(row.get("domain", "")) not in DOMAINS:
                raise GateError(f"{label} invalid identity at line {line_number}")
            if sample_id in indexed:
                raise GateError(f"{label} duplicate sample: {sample_id}")
            indexed[sample_id] = row
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare P0-A5 baseline and Student gate traces.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--student-trace", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        gate = config["gate300"]
        baseline_path = resolve_path(args.baseline_trace)
        student_path = resolve_path(args.student_trace)
        baseline = read_trace(baseline_path, "baseline")
        student = read_trace(student_path, "student")
        if set(baseline) != set(student):
            raise GateError(
                f"Trace identity mismatch: missing={len(set(baseline)-set(student))} "
                f"extra={len(set(student)-set(baseline))}"
            )
        prompt_mismatches = sum(
            baseline[key].get("prompt_hash") != student[key].get("prompt_hash")
            for key in baseline
        )
        baseline_counts = Counter(str(row["domain"]) for row in baseline.values())
        student_counts = Counter(str(row["domain"]) for row in student.values())
        expected = Counter(
            {key: int(value) for key, value in gate["expected_counts"].items()}
        )
        if baseline_counts != expected or student_counts != expected:
            raise GateError(
                f"Unexpected domain counts: baseline={baseline_counts}, student={student_counts}"
            )
        baseline_correct = Counter(
            str(row["domain"]) for row in baseline.values() if row.get("correct") is True
        )
        student_correct = Counter(
            str(row["domain"]) for row in student.values() if row.get("correct") is True
        )
        baseline_accuracy = {
            domain: baseline_correct[domain] / baseline_counts[domain] for domain in DOMAINS
        }
        student_accuracy = {
            domain: student_correct[domain] / student_counts[domain] for domain in DOMAINS
        }
        ratios = {
            domain: (
                student_accuracy[domain] / baseline_accuracy[domain]
                if baseline_accuracy[domain]
                else 0.0
            )
            for domain in DOMAINS
        }
        capped_macro = sum(min(value, 1.0) for value in ratios.values()) / len(DOMAINS)
        generation_errors = sum(
            bool(row.get("generation_error")) for row in student.values()
        )
        initial = float(gate["initial_ratio"])
        recommended = float(gate["recommended_full_ratio"])
        recommended_macro = float(gate["recommended_full_capped_macro"])
        max_errors = int(gate["maximum_generation_errors"])
        initial_pass = (
            all(ratios[domain] >= initial for domain in DOMAINS)
            and generation_errors <= max_errors
            and prompt_mismatches == 0
        )
        recommended_pass = (
            initial_pass
            and all(ratios[domain] >= recommended for domain in DOMAINS)
            and capped_macro >= recommended_macro
        )
        if recommended_pass:
            decision = "eligible_for_quantization_memory_and_formal_full"
        elif initial_pass:
            decision = "eligible_for_second_preregistered_candidate"
        else:
            decision = "reject"
        audit = {
            "gate": "P0-A5-GATE300-RETENTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a5_gate.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if initial_pass else "failed",
            "decision": decision,
            "recommended_full": recommended_pass,
            "candidate_name": args.candidate_name,
            "config": display_path(config_path),
            "config_hash": sha256_file(config_path),
            "baseline_trace": display_path(baseline_path),
            "baseline_trace_hash": sha256_file(baseline_path),
            "student_trace": display_path(student_path),
            "student_trace_hash": sha256_file(student_path),
            "prompt_mismatch_count": prompt_mismatches,
            "baseline_accuracy": baseline_accuracy,
            "student_accuracy": student_accuracy,
            "retention_ratios": ratios,
            "capped_macro_ratio": capped_macro,
            "initial_threshold": initial,
            "recommended_full_threshold": recommended,
            "generation_error_count": generation_errors,
            "maximum_generation_errors": max_errors,
        }
        audit["report_hash"] = hashlib.sha256(
            json.dumps(audit, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {display_path(output)}")
        print(
            f"status={audit['status']} decision={decision} ratios={ratios} "
            f"macro={capped_macro:.6f}"
        )
        return 0 if initial_pass else 1
    except (GateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
