from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from run_student_probe import (
    DEFAULT_STUDENT_MODEL_ID,
    DEFAULT_TEACHER_TRACE,
    agreement_flags,
    call_local_student,
    display_path,
    load_jsonl,
    load_local_student,
    parse_comma_values,
    select_rows,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_TRACE = ROOT / "data" / "distill" / "quant_behavior_trace.smoke.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_quant_behavior_smoke.json"


class QuantBehaviorError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_local_pass(
    rows: list[dict[str, Any]],
    model_dir: Path,
    adapter_path: Path,
    device: str,
    dtype: str,
    max_tokens: int,
    quantize_adapter: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer, model, adapter_config = load_local_student(
        model_dir,
        adapter_path,
        device,
        dtype,
        quantize_adapter=quantize_adapter,
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        result = call_local_student(tokenizer, model, row, device, max_tokens)
        results.append(result)
        mode = "int4" if quantize_adapter else "fp"
        print(
            f"[{mode}] {index}/{len(rows)} {row.get('sample_id')} "
            f"parse_ok={not result.get('parse_errors')}",
            flush=True,
        )
    del model
    del tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results, adapter_config


def build_trace_row(
    teacher_row: dict[str, Any],
    baseline: dict[str, Any],
    quantized: dict[str, Any],
    created_ts: str,
) -> dict[str, Any]:
    baseline_decision = dict(baseline.get("decision_tuple", {}))
    quant_decision = dict(quantized.get("decision_tuple", {}))
    agreement = agreement_flags(baseline_decision, quant_decision)
    baseline_parse_ok = not baseline.get("parse_errors")
    quant_parse_ok = not quantized.get("parse_errors")
    behavior_diverged = (
        baseline_parse_ok != quant_parse_ok
        or not agreement.get("event_type_match", False)
        or not agreement.get("risk_attr_match", False)
        or not agreement.get("action_match", False)
        or not agreement.get("review_intent_match", False)
    )
    row = {
        "quant_behavior_version": "1.0",
        "created_by": "model_compression/verify_quant_behavior.py",
        "created_ts": created_ts,
        "sample_id": teacher_row["sample_id"],
        "dataset_key": teacher_row["dataset_key"],
        "split": teacher_row["split"],
        "task_type": teacher_row["task_type"],
        "teacher_trace_row_hash": teacher_row.get("trace_row_hash", ""),
        "baseline_prompt_hash": baseline.get("prompt_hash", ""),
        "quant_prompt_hash": quantized.get("prompt_hash", ""),
        "baseline_parse_ok": baseline_parse_ok,
        "quant_parse_ok": quant_parse_ok,
        "baseline_parse_errors": baseline.get("parse_errors", []),
        "quant_parse_errors": quantized.get("parse_errors", []),
        "baseline_decision_tuple": baseline_decision,
        "quant_decision_tuple": quant_decision,
        "decision_agreement": agreement,
        "behavior_diverged": behavior_diverged,
        "baseline_latency_ms": baseline.get("latency_ms", 0.0),
        "quant_latency_ms": quantized.get("latency_ms", 0.0),
        "baseline_response_text": baseline.get("response_text", ""),
        "quant_response_text": quantized.get("response_text", ""),
    }
    row["quant_behavior_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify fake INT4 LoRA adapter behavior against FP adapter behavior.")
    parser.add_argument("--teacher-trace", "--teacher_trace", default=str(DEFAULT_TEACHER_TRACE))
    parser.add_argument("--local-model-dir", "--local_model_dir", required=True)
    parser.add_argument("--adapter-path", "--adapter_path", required=True)
    parser.add_argument("--output-trace", "--output_trace", default=str(DEFAULT_OUTPUT_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dataset", action="append", default=[], help="Dataset key filter; repeat or comma-separate.")
    parser.add_argument("--sample-limit", "--sample_limit", type=int, default=3)
    parser.add_argument("--student-model-id", "--student_model_id", default=DEFAULT_STUDENT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--min-parse-rate", type=float, default=0.0)
    parser.add_argument("--max-divergence-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        print("--sample-limit must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.min_parse_rate <= 1:
        print("--min-parse-rate must be in [0, 1]", file=sys.stderr)
        return 2
    if not 0 <= args.max_divergence_rate <= 1:
        print("--max-divergence-rate must be in [0, 1]", file=sys.stderr)
        return 2

    teacher_trace_path = resolve_path(args.teacher_trace)
    model_dir = resolve_path(args.local_model_dir)
    adapter_path = resolve_path(args.adapter_path)
    output_path = resolve_path(args.output_trace)
    audit_path = resolve_path(args.audit)
    dataset_filter = set(parse_comma_values(args.dataset)) if args.dataset else None

    teacher_rows = load_jsonl(teacher_trace_path)
    selected_rows = select_rows(teacher_rows, dataset_filter, args.sample_limit)
    created_ts = datetime.now(timezone.utc).isoformat()

    baseline_results, adapter_config = run_local_pass(
        selected_rows,
        model_dir,
        adapter_path,
        args.device,
        args.dtype,
        args.max_tokens,
        quantize_adapter=False,
    )
    quant_results, _ = run_local_pass(
        selected_rows,
        model_dir,
        adapter_path,
        args.device,
        args.dtype,
        args.max_tokens,
        quantize_adapter=True,
    )

    trace_rows = [
        build_trace_row(teacher_row, baseline, quantized, created_ts)
        for teacher_row, baseline, quantized in zip(selected_rows, baseline_results, quant_results)
    ]
    write_jsonl(output_path, trace_rows)

    baseline_parse_count = sum(1 for row in trace_rows if row["baseline_parse_ok"])
    quant_parse_count = sum(1 for row in trace_rows if row["quant_parse_ok"])
    divergence_count = sum(1 for row in trace_rows if row["behavior_diverged"])
    quant_parse_rate = quant_parse_count / len(trace_rows) if trace_rows else 0.0
    divergence_rate = divergence_count / len(trace_rows) if trace_rows else 1.0
    dataset_counts = Counter(row["dataset_key"] for row in trace_rows)
    status = (
        "passed"
        if trace_rows and quant_parse_rate >= args.min_parse_rate and divergence_rate <= args.max_divergence_rate
        else "failed"
    )
    audit = {
        "gate": "G-KD-TRACE-quant-behavior-smoke" if args.sample_limit else "G-KD-TRACE-quant-behavior",
        "check_version": "1.0",
        "created_by": "model_compression/verify_quant_behavior.py",
        "created_ts": created_ts,
        "status": status,
        "teacher_trace_path": display_path(teacher_trace_path),
        "teacher_trace_hash": sha256_file(teacher_trace_path),
        "local_model_dir": display_path(model_dir),
        "student_model_id": args.student_model_id,
        "adapter_path": display_path(adapter_path),
        "adapter_config": adapter_config,
        "output_trace_path": display_path(output_path),
        "quant_behavior_trace_hash": sha256_file(output_path),
        "sample_limit": args.sample_limit,
        "selected_sample_count": len(selected_rows),
        "baseline_parse_ok_count": baseline_parse_count,
        "quant_parse_ok_count": quant_parse_count,
        "quant_parse_rate": quant_parse_rate,
        "min_parse_rate": args.min_parse_rate,
        "behavior_divergence_count": divergence_count,
        "behavior_divergence_rate": divergence_rate,
        "max_divergence_rate": args.max_divergence_rate,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "selected_sample_ids_hash": sha256_text(
            "\n".join(str(row.get("sample_id", "")) for row in selected_rows) + "\n"
        ),
        "adapter_behavior_mode": "fake_int4_dequantized_lora",
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"quant_behavior_trace_hash={audit['quant_behavior_trace_hash']}")
    if status != "passed":
        print("Quant behavior smoke failed.", file=sys.stderr)
        return 1
    print("Quant behavior smoke passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QuantBehaviorError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
