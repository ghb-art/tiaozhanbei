#!/usr/bin/env python3
"""Mine train-only GSM8K errors for P0-A11 with the frozen Student base."""

from __future__ import annotations

import argparse
import json
import os
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


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (ROOT / "data/p0a11/math_mining_pool.jsonl").resolve()
TRACE = (ROOT / "reports/audit/p0a11/math_mining_trace.jsonl").resolve()
AUDIT = (ROOT / "reports/audit/gate_p0a11_math_mining.json").resolve()
PROGRESS = (ROOT / "reports/audit/p0a11/math_mining_progress.jsonl").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=120)
    return parser.parse_args()


def read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise EvaluationError("Run P0-A11 prepare before Math mining")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with MANIFEST.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise EvaluationError(f"Duplicate/missing id on row {line_number}")
            if row.get("domain") != "math" or row.get("dataset_key") != "gsm8k":
                raise EvaluationError(f"Non-GSM8K Math row {line_number}")
            if row.get("split_role") != "p0a11_internal_validation":
                raise EvaluationError(f"Unexpected split marker on row {line_number}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise EvaluationError("Math mining pool is empty")
    return rows


def evaluate_one(
    index: int,
    row: dict[str, Any],
    endpoint: str,
    model_id: str,
    timeout_sec: float,
    created_ts: str,
) -> tuple[int, dict[str, Any]]:
    messages = build_messages(row)
    response = ""
    generation_error = ""
    started = time.perf_counter()
    try:
        response, latency_ms = generate(endpoint, model_id, messages, 512, timeout_sec)
        correct, prediction, detail, canonical = score_row(row, response, 5.0)
    except (EvaluationError, OSError, ValueError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        correct, prediction, detail, canonical = False, "", str(exc), False
        generation_error = f"{type(exc).__name__}: {exc}"
    item = {
        "capability_eval_version": "p0a11-math-mining-v1",
        "created_ts": created_ts,
        "candidate_name": "p0a11-frozen-base",
        "served_model_id": model_id,
        "domain": "math",
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
    if not 1 <= args.workers <= 32:
        raise EvaluationError("--workers must be in [1, 32]")
    rows = read_manifest()
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    created_ts = datetime.now(timezone.utc).isoformat()
    trace: list[dict[str, Any] | None] = [None] * len(rows)
    by_id = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    if PROGRESS.is_file():
        with PROGRESS.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_id = str(item.get("sample_id", ""))
                if sample_id not in by_id or item.get("served_model_id") != model_id:
                    raise EvaluationError("Stale or incompatible Math mining progress")
                if trace[by_id[sample_id]] is not None:
                    raise EvaluationError("Duplicate id in Math mining progress")
                trace[by_id[sample_id]] = item
    completed = sum(item is not None for item in trace)
    correct_count = sum(int(bool(item and item["correct"])) for item in trace)
    if completed:
        print(f"[math-mine] resumed {completed}/{len(rows)}", flush=True)
    pending = [
        (index, row) for index, row in enumerate(rows) if trace[index] is None
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                evaluate_one, index, row, args.endpoint, model_id,
                args.timeout_sec, created_ts,
            )
            for index, row in pending
        ]
        for future in as_completed(futures):
            index, item = future.result()
            trace[index] = item
            completed += 1
            correct_count += int(item["correct"])
            if completed == 1 or completed % 100 == 0 or completed == len(rows):
                write_jsonl_atomic(PROGRESS, [item for item in trace if item is not None])
                print(
                    f"[math-mine] {completed}/{len(rows)} correct_so_far={correct_count}",
                    flush=True,
                )
    final_trace = [item for item in trace if item is not None]
    if len(final_trace) != len(rows):
        raise EvaluationError("Incomplete Math mining trace")
    write_jsonl_atomic(TRACE, final_trace)
    PROGRESS.unlink(missing_ok=True)
    errors = sum(int(bool(item["generation_error"])) for item in final_trace)
    audit = {
        "gate": "P0-A11-MATH-TRAIN-ONLY-MINING",
        "check_version": "1.0",
        "created_by": "scripts/mine_p0a11_math.py",
        "created_ts": created_ts,
        "status": "passed" if errors == 0 else "failed",
        "served_model_id": model_id,
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_hash": sha256_file(MANIFEST),
        "sample_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
        "generation_error_count": errors,
        "workers": args.workers,
        "trace": TRACE.relative_to(ROOT).as_posix(),
        "trace_hash": sha256_file(TRACE),
        "gate300_loaded": False,
        "formal_full_loaded": False,
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json_atomic(AUDIT, audit)
    print(f"Wrote {AUDIT.relative_to(ROOT)} status={audit['status']}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A11 Math mining failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
