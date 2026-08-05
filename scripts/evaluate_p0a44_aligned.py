#!/usr/bin/env python3
"""Evaluate P0-A44 candidates on train-only, formal-shaped holdouts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_chapter2_capability import run_assert_tests_check, strip_reasoning_envelope
from evaluate_p0a6_internal import (
    EvaluationError,
    discover_model,
    generate,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested."
ALLOWED = {
    "code": (ROOT / "data/p0a44/code_validation.jsonl", 1000),
    "ceval": (ROOT / "data/p0a44/nlp_ceval_dev.jsonl", 260),
    "cmmlu": (ROOT / "data/p0a44/nlp_cmmlu_dev.jsonl", 335),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(ALLOWED), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=180)
    parser.add_argument("--code-timeout-sec", type=float, default=10)
    parser.add_argument("--max-tokens", type=int, default=0)
    return parser.parse_args()


def resolved(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def extract_choice(value: str) -> str:
    stripped = strip_reasoning_envelope(value).strip().upper()
    if re.fullmatch(r"[A-D]", stripped):
        return stripped
    matches = re.findall(r"(?:最终答案|答案)\s*[:：]?\s*([A-D])", value, re.I)
    return matches[-1].upper() if matches else ""


def read_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    expected_path, expected_rows = ALLOWED[dataset]
    if path != expected_path.resolve():
        raise EvaluationError(f"Unexpected manifest: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != expected_rows or len({str(row.get('sample_id')) for row in rows}) != expected_rows:
        raise EvaluationError(f"Unexpected {dataset} rows: {len(rows)}")
    if any(row.get("split_role") != "p0a44_internal_validation" for row in rows):
        raise EvaluationError("P0-A44 split identity changed")
    return rows


def evaluate_one(index: int, row: dict[str, Any], args: argparse.Namespace, model_id: str, created: str) -> tuple[int, dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": str(row["prompt"])}]
    response = ""
    error = ""
    started = time.perf_counter()
    try:
        limit = args.max_tokens or (512 if args.dataset == "code" else 16)
        response, latency = generate(args.endpoint, model_id, messages, limit, args.timeout_sec, enable_thinking=False)
        scored_response = strip_reasoning_envelope(response)
        if args.dataset == "code":
            correct, detail = run_assert_tests_check(
                str(row["prompt_source"]), str(row["entry_point"]),
                [str(x) for x in row["unit_tests"]], scored_response, args.code_timeout_sec,
            )
            prediction = "pass" if correct else "fail"
        else:
            prediction = extract_choice(scored_response)
            correct = prediction == str(row["reference"]).strip().upper()
            detail = "choice_match" if correct else "choice_mismatch"
    except (EvaluationError, OSError, ValueError) as exc:
        latency = (time.perf_counter() - started) * 1000
        correct, prediction, detail = False, "", str(exc)
        error = f"{type(exc).__name__}: {exc}"
    item = {
        "capability_eval_version": "p0a44-aligned-v1", "created_ts": created,
        "candidate_name": args.candidate_name, "served_model_id": model_id,
        "dataset": args.dataset, "sample_id": row["sample_id"],
        "prompt_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "correct": bool(correct), "prediction": prediction, "score_detail": detail,
        "latency_ms": latency, "generation_error": error, "response_text": response,
    }
    item["row_hash"] = sha256_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return index, item


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise EvaluationError("workers must be in [1,16]")
    manifest = resolved(args.manifest)
    trace_path, audit_path = resolved(args.output_trace), resolved(args.audit)
    audit_root = (ROOT / "reports/audit/p0a44").resolve()
    if audit_root not in trace_path.parents or audit_root not in audit_path.parents:
        raise EvaluationError("P0-A44 outputs must stay in reports/audit/p0a44")
    rows = read_rows(manifest, args.dataset)
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    created = datetime.now(timezone.utc).isoformat()
    progress = trace_path.with_name(trace_path.stem + "_progress.jsonl")
    slots: list[dict[str, Any] | None] = [None] * len(rows)
    indices = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    if progress.is_file():
        for line in progress.open(encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line); sample_id = str(item.get("sample_id", ""))
            if sample_id not in indices or item.get("served_model_id") != model_id:
                raise EvaluationError("Stale P0-A44 progress")
            slots[indices[sample_id]] = item
    pending = [(i, row) for i, row in enumerate(rows) if slots[i] is None]
    completed = len(rows) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_one, i, row, args, model_id, created) for i, row in pending]
        for future in as_completed(futures):
            index, item = future.result(); slots[index] = item; completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(rows):
                write_jsonl_atomic(progress, [item for item in slots if item is not None])
                print(f"[{args.dataset}] {completed}/{len(rows)} correct_so_far={sum(int(bool(x and x['correct'])) for x in slots)}", flush=True)
    trace = [item for item in slots if item is not None]
    if len(trace) != len(rows):
        raise EvaluationError("Incomplete P0-A44 trace")
    write_jsonl_atomic(trace_path, trace); progress.unlink(missing_ok=True)
    correct = sum(int(row["correct"]) for row in trace)
    errors = sum(int(bool(row["generation_error"])) for row in trace)
    audit = {
        "gate": "P0-A44-ALIGNED-INTERNAL-VALIDATION", "check_version": "1.0",
        "created_by": "scripts/evaluate_p0a44_aligned.py", "created_ts": created,
        "status": "passed" if errors == 0 else "failed", "dataset": args.dataset,
        "candidate_name": args.candidate_name, "served_model_id": model_id,
        "manifest": str(manifest.relative_to(ROOT)), "manifest_hash": sha256_file(manifest),
        "sample_count": len(trace), "correct_count": correct, "accuracy": correct / len(trace),
        "generation_error_count": errors, "workers": args.workers,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace) / len(trace),
        "trace": str(trace_path.relative_to(ROOT)), "trace_hash": sha256_file(trace_path),
        "formal_full_loaded": False,
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json_atomic(audit_path, audit)
    print(f"Wrote {audit_path.relative_to(ROOT)} status={audit['status']} accuracy={audit['accuracy']:.6f}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A44 aligned evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
