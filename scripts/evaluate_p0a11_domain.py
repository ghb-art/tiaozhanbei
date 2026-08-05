#!/usr/bin/env python3
"""Evaluate P0-A11 Math or Code on its frozen train-only holdout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import (
    EvaluationError,
    build_messages,
    discover_model,
    generate,
    score_row,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from evaluate_p0a5_gate import build_messages as build_frozen_gate_messages


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "p0a11": {
        "audit_root": (ROOT / "reports/audit/p0a11").resolve(),
        "split_role": "p0a11_internal_validation",
        "gate": "P0-A11-TRAIN-ONLY-VALIDATION",
        "version": "p0a11-domain-v1",
    },
    "p0a12": {
        "audit_root": (ROOT / "reports/audit/p0a12").resolve(),
        "split_role": "p0a12_external_validation",
        "gate": "P0-A12-EXTERNAL-MATH-VALIDATION",
        "version": "p0a12-math-v1",
    },
    "p0a13": {
        "audit_root": (ROOT / "reports/audit/p0a13").resolve(),
        "split_role": "p0a13_external_validation",
        "gate": "P0-A13-MATH-RUNTIME-VALIDATION",
        "version": "p0a13-math-runtime-v1",
    },
    "p0a15": {
        "audit_root": (ROOT / "reports/audit/p0a15").resolve(),
        "split_role": "p0a15_external_validation",
        "gate": "P0-A15-MATH-SCALED-ADAPTER-VALIDATION",
        "version": "p0a15-math-scaled-adapter-v1",
    },
    "p0a16": {
        "audit_root": (ROOT / "reports/audit/p0a16").resolve(),
        "split_role": "p0a16_external_validation",
        "gate": "P0-A16-MATH-JOINT-RUNTIME-VALIDATION",
        "version": "p0a16-math-joint-runtime-v1",
    },
    "p0a18": {
        "audit_root": (ROOT / "reports/audit/p0a18").resolve(),
        "split_role": "p0a18_external_validation",
        "gate": "P0-A18-CODE-TRANSFER-VALIDATION",
        "version": "p0a18-code-transfer-v1",
    },
    "p0a19": {
        "audit_root": (ROOT / "reports/audit/p0a19").resolve(),
        "split_role": "p0a19_external_validation",
        "gate": "P0-A19-CODE-MIXED-VALIDATION",
        "version": "p0a19-code-mixed-v1",
    },
    "p0a20": {
        "audit_root": (ROOT / "reports/audit/p0a20").resolve(),
        "split_role": "p0a20_external_validation",
        "gate": "P0-A20-CODE-THINKING-VALIDATION",
        "version": "p0a20-code-thinking-v1",
    },
    "p0a21": {
        "audit_root": (ROOT / "reports/audit/p0a21").resolve(),
        "split_role": "p0a21_external_validation",
        "gate": "P0-A21-FRESH-CODE-VALIDATION",
        "version": "p0a21-fresh-code-v1",
    },
    "p0a23": {
        "audit_root": (ROOT / "reports/audit/p0a23").resolve(),
        "split_role": "p0a23_external_validation",
        "gate": "P0-A23-CONTINUED-CODE-VALIDATION",
        "version": "p0a23-continued-code-v1",
    },
    "p0a25": {
        "audit_root": (ROOT / "reports/audit/p0a25").resolve(),
        "split_role": "p0a25_external_validation",
        "gate": "P0-A25-FAILURE-REPAIR-VALIDATION",
        "version": "p0a25-failure-repair-v1",
    },
    "p0a25_mining": {
        "audit_root": (ROOT / "reports/audit/p0a25").resolve(),
        "split_role": "p0a25_train_only_mining",
        "manifest": (ROOT / "data/p0a25/code_mining_manifest.jsonl").resolve(),
        "gate": "P0-A25-TRAIN-ONLY-FAILURE-MINING",
        "version": "p0a25-train-only-mining-v1",
    },
    "p0a26_gate": {
        "audit_root": (ROOT / "reports/audit/p0a26").resolve(),
        "split_role": "p0a25_frozen_gate",
        "manifest": (ROOT / "data/p0a25/code_gate100.jsonl").resolve(),
        "gate": "P0-A26-FRESH-CODE100-EVAL",
        "version": "p0a26-fresh-code100-v1",
    },
    "p0a27_quantized_gate": {
        "audit_root": (ROOT / "reports/audit/p0a27").resolve(),
        "split_role": "p0a25_frozen_gate",
        "manifest": (ROOT / "data/p0a25/code_gate100.jsonl").resolve(),
        "gate": "P0-A27-QUANTIZED-CODE100-EVAL",
        "version": "p0a27-quantized-code100-v1",
    },
    "p0a30": {
        "audit_root": (ROOT / "reports/audit/p0a30").resolve(),
        "split_role": "p0a30_external_validation",
        "gate": "P0-A30-NLP-SCALE-VALIDATION",
        "version": "p0a30-nlp-scale-v1",
    },
    "p0a31_nlp_gate": {
        "audit_root": (ROOT / "reports/audit/p0a31").resolve(),
        "split_role": "p0a31_frozen_gate",
        "manifest": (ROOT / "data/p0a31/nlp_gate100.jsonl").resolve(),
        "gate": "P0-A31-NLP100-EVAL",
        "version": "p0a31-nlp100-v1",
    },
    "p0a32": {
        "audit_root": (ROOT / "reports/audit/p0a32").resolve(),
        "split_role": "p0a32_external_validation",
        "manifest": (ROOT / "data/p0a32/nlp_internal_validation.jsonl").resolve(),
        "gate": "P0-A32-NLP-CONTINUATION-VALIDATION",
        "version": "p0a32-nlp-continuation-v1",
    },
    "p0a33_nlp_gate": {
        "audit_root": (ROOT / "reports/audit/p0a33").resolve(),
        "split_role": "p0a31_frozen_gate",
        "manifest": (ROOT / "data/p0a31/nlp_gate100.jsonl").resolve(),
        "gate": "P0-A33-TRAINED-NLP100-EVAL",
        "version": "p0a33-trained-nlp100-v1",
    },
    "p0a34": {
        "audit_root": (ROOT / "reports/audit/p0a34").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A34-CHINESE-EXAM-VALIDATION",
        "version": "p0a34-chinese-exam-v1",
    },
    "p0a35": {
        "audit_root": (ROOT / "reports/audit/p0a35").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A35-NLP-RUNTIME-VALIDATION",
        "version": "p0a35-nlp-runtime-v1",
    },
    "p0a36": {
        "audit_root": (ROOT / "reports/audit/p0a36").resolve(),
        "split_role": "p0a36_external_validation",
        "manifest": (ROOT / "data/p0a36/nlp_validation.jsonl").resolve(),
        "gate": "P0-A36-BALANCED-MCQ-VALIDATION",
        "version": "p0a36-balanced-mcq-v1",
    },
    "p0a37": {
        "audit_root": (ROOT / "reports/audit/p0a37").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A37-NLP-TRANSFER-VALIDATION",
        "version": "p0a37-nlp-transfer-v1",
    },
    "p0a38_nlp_long_gate": {
        "audit_root": (ROOT / "reports/audit/p0a38").resolve(),
        "split_role": "p0a31_frozen_gate",
        "manifest": (ROOT / "data/p0a31/nlp_gate100.jsonl").resolve(),
        "gate": "P0-A38-NLP-LONG-OUTPUT-GATE",
        "version": "p0a38-nlp-long-output-v1",
    },
    "p0a39_synth": {
        "audit_root": (ROOT / "reports/audit/p0a39").resolve(),
        "split_role": "p0a36_external_validation",
        "manifest": (ROOT / "data/p0a36/nlp_validation.jsonl").resolve(),
        "gate": "P0-A39-ORIGINAL-QWEN3-SYNTH-VALIDATION",
        "version": "p0a39-original-qwen3-synth-v1",
    },
    "p0a39_ceval": {
        "audit_root": (ROOT / "reports/audit/p0a39").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A39-ORIGINAL-QWEN3-CEVAL-VALIDATION",
        "version": "p0a39-original-qwen3-ceval-v1",
    },
    "p0a40_synth": {
        "audit_root": (ROOT / "reports/audit/p0a40").resolve(),
        "split_role": "p0a36_external_validation",
        "manifest": (ROOT / "data/p0a36/nlp_validation.jsonl").resolve(),
        "gate": "P0-A40-ORIGINAL-PLUS-NLP-SYNTH-VALIDATION",
        "version": "p0a40-original-plus-nlp-synth-v1",
    },
    "p0a40_ceval": {
        "audit_root": (ROOT / "reports/audit/p0a40").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A40-ORIGINAL-PLUS-NLP-CEVAL-VALIDATION",
        "version": "p0a40-original-plus-nlp-ceval-v1",
    },
    "p0a42_math": {
        "audit_root": (ROOT / "reports/audit/p0a42").resolve(),
        "split_role": "p0a10_internal_validation",
        "manifest": (ROOT / "data/p0a10/math_validation.jsonl").resolve(),
        "gate": "P0-A42-MATH-TRAIN-ONLY-VALIDATION",
        "version": "p0a42-math-v1",
    },
    "p0a42_code": {
        "audit_root": (ROOT / "reports/audit/p0a42").resolve(),
        "split_role": "p0a25_external_validation",
        "manifest": (ROOT / "data/p0a25/code_validation.jsonl").resolve(),
        "gate": "P0-A42-CODE-TRAIN-ONLY-VALIDATION",
        "version": "p0a42-code-v1",
    },
    "p0a42_nlp": {
        "audit_root": (ROOT / "reports/audit/p0a42").resolve(),
        "split_role": "p0a34_external_validation",
        "manifest": (ROOT / "data/p0a34/nlp_validation.jsonl").resolve(),
        "gate": "P0-A42-NLP-TRAIN-ONLY-VALIDATION",
        "version": "p0a42-nlp-v1",
    },
}


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="p0a11")
    parser.add_argument("--domain", required=True, choices=("math", "code", "nlp"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument(
        "--thinking",
        choices=("off", "on"),
        default="off",
        help="P0-A13 only: toggle Qwen3 native thinking for the Math request.",
    )
    return parser.parse_args()


def read_manifest(
    path: Path, domain: str, expected: int, protocol: str
) -> list[dict[str, Any]]:
    allowed = PROTOCOLS[protocol].get(
        "manifest", (ROOT / f"data/{protocol}/{domain}_validation.jsonl").resolve()
    )
    if path != allowed or not path.is_file():
        raise EvaluationError(f"Unexpected {protocol.upper()} manifest: {display(path)}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise EvaluationError(f"Duplicate/missing id on row {line_number}")
            if row.get("domain") != domain:
                raise EvaluationError(f"Wrong domain on row {line_number}")
            if row.get("split_role") != PROTOCOLS[protocol]["split_role"]:
                raise EvaluationError(f"Wrong split on row {line_number}")
            if domain == "code":
                test_count = len(row.get("unit_tests") or [])
                if protocol in {"p0a19", "p0a20"}:
                    upper = 6 if protocol == "p0a20" else 5
                    tests_valid = 3 <= test_count <= upper
                    expected_text = f"3 to {upper}"
                else:
                    expected_tests = 3 if protocol == "p0a18" else 10
                    tests_valid = test_count == expected_tests
                    expected_text = str(expected_tests)
                if not tests_valid:
                    raise EvaluationError(
                        f"Code row lacks {expected_text} tests on row {line_number}"
                    )
            seen.add(sample_id)
            rows.append(row)
    if expected <= 0 or len(rows) != expected:
        raise EvaluationError(f"Expected {expected} rows, found {len(rows)}")
    return rows


def validate_output(path: Path, suffix: str, protocol: str) -> None:
    audit_root = PROTOCOLS[protocol]["audit_root"]
    if path.suffix != suffix or audit_root not in path.parents:
        raise EvaluationError(
            f"{protocol.upper()} output must remain in reports/audit/{protocol}: {display(path)}"
        )


def evaluate_one(
    index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    model_id: str,
    created_ts: str,
) -> tuple[int, dict[str, Any]]:
    messages = (
        build_frozen_gate_messages(row)
        if args.protocol in {"p0a31_nlp_gate", "p0a32", "p0a33_nlp_gate", "p0a34", "p0a35", "p0a36", "p0a37", "p0a38_nlp_long_gate", "p0a39_synth", "p0a39_ceval", "p0a40_synth", "p0a40_ceval", "p0a42_nlp"}
        else build_messages(row)
    )
    response = ""
    generation_error = ""
    started = time.perf_counter()
    try:
        token_limit = args.max_tokens or (512 if args.domain == "math" else 768)
        response, latency_ms = generate(
            args.endpoint,
            model_id,
            messages,
            token_limit,
            args.timeout_sec,
            enable_thinking=args.thinking == "on",
        )
        correct, prediction, detail, canonical = score_row(
            row, response, args.code_timeout_sec
        )
    except (EvaluationError, OSError, ValueError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        correct, prediction, detail, canonical = False, "", str(exc), False
        generation_error = f"{type(exc).__name__}: {exc}"
    item = {
            "capability_eval_version": PROTOCOLS[args.protocol]["version"],
            "created_ts": created_ts,
            "candidate_name": args.candidate_name,
            "served_model_id": model_id,
            "domain": args.domain,
            "sample_id": row["sample_id"],
            "prompt_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
            "correct": bool(correct),
            "canonical_format": bool(canonical),
            "prediction": prediction,
            "score_detail": detail,
            "latency_ms": latency_ms,
            "generation_error": generation_error,
            "response_text": response,
    }
    item["row_hash"] = sha256_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return index, item


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise EvaluationError("--workers must be in [1, 16]")
    if args.max_tokens < 0 or args.max_tokens > 1536:
        raise EvaluationError("--max-tokens must be in [0, 1536]")
    approved_thinking = (
        args.protocol in {"p0a13", "p0a15", "p0a16"} and args.domain == "math"
    ) or (args.protocol == "p0a20" and args.domain == "code") or (
        args.protocol == "p0a35" and args.domain == "nlp"
    )
    if args.thinking == "on" and not approved_thinking:
        raise EvaluationError("Thinking mode is restricted to approved Math runtime validation")
    manifest = resolve(args.manifest)
    trace_path = resolve(args.output_trace)
    audit_path = resolve(args.audit)
    validate_output(trace_path, ".jsonl", args.protocol)
    validate_output(audit_path, ".json", args.protocol)
    rows = read_manifest(manifest, args.domain, args.expected_rows, args.protocol)
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    created_ts = datetime.now(timezone.utc).isoformat()
    progress_path = trace_path.with_name(trace_path.stem + "_progress.jsonl")
    trace_slots: list[dict[str, Any] | None] = [None] * len(rows)
    row_indices = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    if progress_path.is_file():
        for line in progress_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", ""))
            if sample_id not in row_indices or item.get("served_model_id") != model_id:
                raise EvaluationError("Stale or incompatible validation progress")
            index = row_indices[sample_id]
            if trace_slots[index] is not None:
                raise EvaluationError("Duplicate id in validation progress")
            trace_slots[index] = item
    completed = sum(item is not None for item in trace_slots)
    pending = [(index, row) for index, row in enumerate(rows) if trace_slots[index] is None]
    if completed:
        print(f"[{args.domain}] resumed {completed}/{len(rows)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(evaluate_one, index, row, args, model_id, created_ts)
            for index, row in pending
        ]
        for future in as_completed(futures):
            index, item = future.result()
            trace_slots[index] = item
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(rows):
                write_jsonl_atomic(
                    progress_path, [item for item in trace_slots if item is not None]
                )
                correct_so_far = sum(
                    int(bool(item and item["correct"])) for item in trace_slots
                )
                print(
                    f"[{args.domain}] {completed}/{len(rows)} "
                    f"correct_so_far={correct_so_far}",
                    flush=True,
                )
    trace = [item for item in trace_slots if item is not None]
    if len(trace) != len(rows):
        raise EvaluationError(f"Incomplete {args.protocol.upper()} validation trace")
    write_jsonl_atomic(trace_path, trace)
    progress_path.unlink(missing_ok=True)
    correct_count = sum(int(row["correct"]) for row in trace)
    canonical_count = sum(int(row["canonical_format"]) for row in trace)
    error_count = sum(int(bool(row["generation_error"])) for row in trace)
    audit = {
        "gate": PROTOCOLS[args.protocol]["gate"],
        "check_version": "1.0",
        "created_by": "scripts/evaluate_p0a11_domain.py",
        "created_ts": created_ts,
        "status": "passed" if error_count == 0 else "failed",
        "domain": args.domain,
        "candidate_name": args.candidate_name,
        "served_model_id": model_id,
        "manifest": display(manifest),
        "manifest_hash": sha256_file(manifest),
        "sample_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
        "canonical_format_rate": canonical_count / len(rows),
        "generation_error_count": error_count,
        "workers": args.workers,
        "max_tokens": args.max_tokens or (512 if args.domain == "math" else 768),
        "thinking": args.thinking,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace) / len(rows),
        "trace": display(trace_path),
        "trace_hash": sha256_file(trace_path),
        "gate300_loaded": False,
        "formal_full_loaded": False,
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json_atomic(audit_path, audit)
    print(f"Wrote {display(audit_path)}")
    print(f"status={audit['status']} accuracy={audit['accuracy']:.6f}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Domain evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
