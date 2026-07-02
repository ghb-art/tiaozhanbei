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
DEFAULT_PROBE_TRACE = ROOT / "data" / "distill" / "student_probe_trace.jsonl"
DEFAULT_OUTPUT_REPAIR = ROOT / "data" / "distill" / "counterfactual_repair_trace.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_repair_mining.json"


class RepairMiningError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def counterfactual_type(reasons: list[str]) -> str:
    if "parse_error" in reasons:
        return "parse_recovery"
    if "action_mismatch" in reasons:
        return "action_boundary"
    if "risk_attr_mismatch" in reasons:
        return "risk_boundary"
    if "event_type_mismatch" in reasons:
        return "event_boundary"
    if "low_student_confidence" in reasons or "confidence_gap" in reasons:
        return "confidence_boundary"
    if "review_intent_mismatch" in reasons:
        return "review_intent_boundary"
    return "teacher_replay_boundary"


def build_boundary_prompt(probe_row: dict[str, Any], reasons: list[str]) -> str:
    teacher = probe_row.get("teacher_decision_tuple", {})
    student = probe_row.get("student_decision_tuple", {})
    return (
        f"Repair sample {probe_row.get('sample_id')} because {', '.join(reasons)}. "
        f"Student action={student.get('action')} risk={student.get('risk_attr')} "
        f"confidence={student.get('confidence')}; teacher action={teacher.get('action')} "
        f"risk={teacher.get('risk_attr')} confidence={teacher.get('confidence')}."
    )


def build_repair_row(probe_row: dict[str, Any], created_ts: str) -> dict[str, Any]:
    reasons = [str(item) for item in probe_row.get("repair_candidate_reasons", [])]
    teacher_decision = dict(probe_row.get("teacher_decision_tuple", {}))
    student_decision = dict(probe_row.get("student_decision_tuple", {}))
    target = {
        "decision_tuple": teacher_decision,
        "short_rationale": teacher_decision.get("short_rationale", ""),
        "repair_action": "align_student_to_teacher_decision",
    }
    row = {
        "repair_version": "1.0",
        "created_by": "model_compression/mine_counterfactual_repairs.py",
        "created_ts": created_ts,
        "sample_id": probe_row["sample_id"],
        "dataset_key": probe_row["dataset_key"],
        "split": probe_row["split"],
        "task_type": probe_row["task_type"],
        "student_probe_row_hash": probe_row.get("probe_row_hash", ""),
        "teacher_trace_row_hash": probe_row.get("teacher_trace_row_hash", ""),
        "repair_source": "student_probe_teacher_trace_alignment",
        "counterfactual_type": counterfactual_type(reasons),
        "repair_reasons": reasons,
        "boundary_prompt": build_boundary_prompt(probe_row, reasons),
        "student_decision_tuple": student_decision,
        "teacher_decision_tuple": teacher_decision,
        "target_json": target,
        "used_for_training": probe_row.get("split") == "train",
    }
    row["repair_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def select_candidates(
    probe_rows: list[dict[str, Any]],
    include_non_candidates: bool,
    sample_limit: int | None,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in probe_rows
        if include_non_candidates or row.get("repair_candidate") is True
    ]
    if sample_limit is not None:
        candidates = candidates[:sample_limit]
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine counterfactual repair traces from student probe rows.")
    parser.add_argument("--probe-trace", "--probe_trace", default=str(DEFAULT_PROBE_TRACE))
    parser.add_argument("--output-repair", "--output_repair", default=str(DEFAULT_OUTPUT_REPAIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--sample-limit", "--sample_limit", type=int, default=None)
    parser.add_argument("--min-repairs", "--min_repairs", type=int, default=1)
    parser.add_argument(
        "--include-non-candidates",
        action="store_true",
        help="Also emit non-candidate rows. Useful only for schema dry-runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        print("--sample-limit must be positive", file=sys.stderr)
        return 2
    if args.min_repairs < 0:
        print("--min-repairs must be non-negative", file=sys.stderr)
        return 2

    probe_path = resolve_path(args.probe_trace)
    output_path = resolve_path(args.output_repair)
    audit_path = resolve_path(args.audit)
    probe_rows = load_jsonl(probe_path)
    selected = select_candidates(probe_rows, args.include_non_candidates, args.sample_limit)
    created_ts = datetime.now(timezone.utc).isoformat()
    repair_rows = [build_repair_row(row, created_ts) for row in selected]
    write_jsonl(output_path, repair_rows)

    dataset_counts = Counter(row["dataset_key"] for row in repair_rows)
    type_counts = Counter(row["counterfactual_type"] for row in repair_rows)
    reason_counts = Counter(reason for row in repair_rows for reason in row.get("repair_reasons", []))
    status = "passed" if len(repair_rows) >= args.min_repairs else "failed"
    audit = {
        "gate": "G-KD-TRACE-repair-mining-smoke" if args.sample_limit else "G-KD-TRACE-repair-mining",
        "check_version": "1.0",
        "created_by": "model_compression/mine_counterfactual_repairs.py",
        "created_ts": created_ts,
        "status": status,
        "probe_trace_path": display_path(probe_path),
        "student_probe_trace_hash": sha256_file(probe_path),
        "output_repair_path": display_path(output_path),
        "counterfactual_repair_trace_hash": sha256_file(output_path),
        "input_probe_count": len(probe_rows),
        "repair_candidate_input_count": sum(1 for row in probe_rows if row.get("repair_candidate") is True),
        "repair_trace_count": len(repair_rows),
        "min_repairs": args.min_repairs,
        "sample_limit": args.sample_limit,
        "include_non_candidates": bool(args.include_non_candidates),
        "selected_sample_ids_hash": sha256_text(
            "\n".join(str(row.get("sample_id", "")) for row in repair_rows) + "\n"
        ),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "counterfactual_type_counts": dict(sorted(type_counts.items())),
        "repair_reason_counts": dict(sorted(reason_counts.items())),
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"counterfactual_repair_trace_hash={audit['counterfactual_repair_trace_hash']}")
    if status != "passed":
        print("Counterfactual repair mining failed.", file=sys.stderr)
        return 1
    print("Counterfactual repair mining passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RepairMiningError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
