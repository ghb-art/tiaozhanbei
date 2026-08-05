#!/usr/bin/env python3
"""Evaluate single-thinking vs fixed three-sample self-consistency on Calc-MAWPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_p0a6_internal import (
    EvaluationError,
    build_messages,
    discover_model,
    endpoint_root,
    extract_number,
    generate,
    request_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (ROOT / "data/p0a14/math_validation.jsonl").resolve()
AUDIT_ROOT = (ROOT / "reports/audit/p0a14").resolve()
EXPECTED_ROWS = 727


def resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def stable_seed(sample_id: str) -> int:
    return int(hashlib.sha256(f"p0a14:{sample_id}".encode()).hexdigest()[:8], 16)


def numeric_vote(predictions: list[str]) -> tuple[str, int, bool]:
    """Return majority prediction without accepting or reading a reference answer."""
    nonempty = [value for value in predictions if value]
    if not nonempty:
        return "", 0, True
    counts = Counter(nonempty)
    maximum = max(counts.values())
    winners = {value for value, count in counts.items() if count == maximum}
    for index, value in enumerate(predictions):
        if value in winners:
            return value, index, len(winners) > 1
    raise AssertionError("numeric vote has no selected index")


def generate_vote(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout: float,
    seed: int,
) -> tuple[list[str], list[str], str, int, bool, float]:
    started = time.perf_counter()
    payload = request_json(
        endpoint_root(endpoint) + "/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 3,
            "seed": int(seed),
            "max_tokens": 768,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 3:
        raise EvaluationError(f"Expected 3 completion choices, got: {payload}")
    responses: list[str] = []
    predictions: list[str] = []
    canonical: list[bool] = []
    for choice in choices:
        try:
            content = choice["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise EvaluationError(f"Malformed self-consistency choice: {choice}") from exc
        if content is None:
            raise EvaluationError("Self-consistency completion content is null")
        response = str(content)
        prediction, is_canonical = extract_number(response)
        responses.append(response)
        predictions.append(prediction)
        canonical.append(is_canonical)
    selected, selected_index, tied = numeric_vote(predictions)
    selected_canonical = canonical[selected_index] if selected_index < len(canonical) else False
    return responses, predictions, selected, selected_index, tied, latency_ms if selected_canonical else -latency_ms


def read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise EvaluationError("Missing P0-A14 manifest")
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8") if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise EvaluationError(f"Expected {EXPECTED_ROWS} P0-A14 rows, got {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if (
            not sample_id
            or sample_id in seen
            or row.get("domain") != "math"
            or row.get("split_role") != "p0a14_external_validation"
        ):
            raise EvaluationError(f"Invalid P0-A14 row: {sample_id!r}")
        seen.add(sample_id)
    return rows


def evaluate_one(
    index: int,
    row: dict[str, Any],
    mode: str,
    endpoint: str,
    model_id: str,
    timeout: float,
    created_ts: str,
) -> tuple[int, dict[str, Any]]:
    messages = build_messages(row)
    generation_error = ""
    responses: list[str] = []
    predictions: list[str] = []
    selected_index = 0
    vote_tied = False
    canonical = False
    started = time.perf_counter()
    try:
        if mode == "single":
            response, latency_ms = generate(
                endpoint,
                model_id,
                messages,
                768,
                timeout,
                enable_thinking=True,
            )
            prediction, canonical = extract_number(response)
            responses = [response]
            predictions = [prediction]
        else:
            responses, predictions, prediction, selected_index, vote_tied, signed_latency = generate_vote(
                endpoint, model_id, messages, timeout, stable_seed(str(row["sample_id"]))
            )
            canonical = signed_latency >= 0
            latency_ms = abs(signed_latency)
        reference, _ = extract_number(str(row["reference"]))
        correct = bool(prediction) and prediction == reference
        detail = f"reference={reference}"
    except (EvaluationError, OSError, ValueError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        prediction, reference, correct, detail = "", str(row.get("reference", "")), False, str(exc)
        generation_error = f"{type(exc).__name__}: {exc}"
    item = {
        "capability_eval_version": "p0a14-self-consistency-v1",
        "created_ts": created_ts,
        "mode": mode,
        "served_model_id": model_id,
        "domain": "math",
        "sample_id": row["sample_id"],
        "prompt_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "sample_count": 1 if mode == "single" else 3,
        "seed": None if mode == "single" else stable_seed(str(row["sample_id"])),
        "predictions": predictions,
        "prediction": prediction,
        "selected_index": selected_index,
        "vote_tied": vote_tied,
        "correct": bool(correct),
        "canonical_format": bool(canonical),
        "score_detail": detail,
        "latency_ms": latency_ms,
        "generation_error": generation_error,
        "response_texts": responses,
    }
    item["row_hash"] = sha256_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return index, item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", choices=("single", "vote3"), required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=180)
    args = parser.parse_args()
    try:
        if not 1 <= args.workers <= 16:
            raise EvaluationError("--workers must be in [1, 16]")
        trace_path = resolve(args.output_trace)
        audit_path = resolve(args.audit)
        if AUDIT_ROOT not in trace_path.parents or AUDIT_ROOT not in audit_path.parents:
            raise EvaluationError("P0-A14 outputs must stay in reports/audit/p0a14")
        if trace_path.suffix != ".jsonl" or audit_path.suffix != ".json":
            raise EvaluationError("Invalid P0-A14 output suffix")
        rows = read_manifest()
        model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
        created_ts = datetime.now(timezone.utc).isoformat()
        progress = trace_path.with_name(trace_path.stem + "_progress.jsonl")
        slots: list[dict[str, Any] | None] = [None] * len(rows)
        positions = {str(row["sample_id"]): index for index, row in enumerate(rows)}
        if progress.is_file():
            for line in progress.open(encoding="utf-8"):
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_id = str(item.get("sample_id", ""))
                if (
                    sample_id not in positions
                    or item.get("mode") != args.mode
                    or item.get("served_model_id") != model_id
                ):
                    raise EvaluationError("Stale P0-A14 progress")
                slots[positions[sample_id]] = item
        completed = sum(item is not None for item in slots)
        pending = [(index, row) for index, row in enumerate(rows) if slots[index] is None]
        if completed:
            print(f"[{args.mode}] resumed {completed}/{len(rows)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    evaluate_one,
                    index,
                    row,
                    args.mode,
                    args.endpoint,
                    model_id,
                    args.timeout_sec,
                    created_ts,
                )
                for index, row in pending
            ]
            for future in as_completed(futures):
                index, item = future.result()
                slots[index] = item
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(rows):
                    write_jsonl_atomic(progress, [item for item in slots if item is not None])
                    correct = sum(int(bool(item and item["correct"])) for item in slots)
                    print(
                        f"[{args.mode}] {completed}/{len(rows)} correct_so_far={correct}",
                        flush=True,
                    )
        trace = [item for item in slots if item is not None]
        if len(trace) != len(rows):
            raise EvaluationError("Incomplete P0-A14 trace")
        write_jsonl_atomic(trace_path, trace)
        progress.unlink(missing_ok=True)
        correct_count = sum(int(row["correct"]) for row in trace)
        canonical_count = sum(int(row["canonical_format"]) for row in trace)
        error_count = sum(int(bool(row["generation_error"])) for row in trace)
        tied_count = sum(int(bool(row["vote_tied"])) for row in trace)
        audit = {
            "gate": "P0-A14-MATH-SELF-CONSISTENCY-VALIDATION",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a14_math.py",
            "created_ts": created_ts,
            "status": "passed" if error_count == 0 else "failed",
            "mode": args.mode,
            "served_model_id": model_id,
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_hash": sha256_file(MANIFEST),
            "sample_count": len(rows),
            "responses_per_sample": 1 if args.mode == "single" else 3,
            "thinking": True,
            "temperature": 0.0 if args.mode == "single" else 0.6,
            "top_p": 1.0 if args.mode == "single" else 0.95,
            "max_tokens": 768,
            "vote": None if args.mode == "single" else "numeric_majority_then_first",
            "correct_count": correct_count,
            "accuracy": correct_count / len(rows),
            "canonical_format_rate": canonical_count / len(rows),
            "generation_error_count": error_count,
            "vote_tied_count": tied_count,
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace) / len(rows),
            "trace": trace_path.relative_to(ROOT).as_posix(),
            "trace_hash": sha256_file(trace_path),
            "gate300_loaded": False,
            "formal_full_loaded": False,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(audit_path, audit)
        print(f"Wrote {audit_path.relative_to(ROOT)}")
        print(f"status={audit['status']} accuracy={audit['accuracy']:.6f}")
        return 0 if error_count == 0 else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A14 evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
