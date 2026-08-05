#!/usr/bin/env python3
"""Mine train-only errors of the bare Student on CMMLU-shaped NLP rows."""

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

from evaluate_chapter2_capability import strip_reasoning_envelope
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
SOURCE = ROOT / "data/p0a44/nlp_train.jsonl"
TRACE = ROOT / "reports/audit/p0a48/mining_trace.jsonl"
AUDIT = ROOT / "reports/audit/p0a48/mining.json"


def extract_choice(value: str) -> str:
    cleaned = strip_reasoning_envelope(value).strip().upper()
    if re.fullmatch(r"[A-D]", cleaned):
        return cleaned
    matches = re.findall(r"(?:最终答案|答案)\s*[:：]?\s*([A-D])", cleaned, re.I)
    return matches[-1].upper() if matches else ""


def read_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    if len(rows) != 3939 or len({str(row.get("sample_id")) for row in rows}) != 3939:
        raise EvaluationError(f"Unexpected P0-A48 mining rows: {len(rows)}")
    for row in rows:
        metadata = row.get("metadata", {})
        if row.get("split_role") != "train" or metadata.get("output_contract") != "cmmlu_v15_single_letter":
            raise EvaluationError("P0-A48 source contract changed")
        if str(row.get("answer", "")).strip().upper() not in set("ABCD"):
            raise EvaluationError("P0-A48 source contains a non-choice target")
    return rows


def evaluate_one(index: int, row: dict[str, Any], args: argparse.Namespace, model_id: str, created: str) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    response = ""
    error = ""
    try:
        response, latency = generate(args.endpoint, model_id, row["messages"], 16, args.timeout_sec, enable_thinking=False)
        prediction = extract_choice(response)
    except (EvaluationError, OSError, ValueError) as exc:
        latency = (time.perf_counter() - started) * 1000
        prediction = ""
        error = f"{type(exc).__name__}: {exc}"
    item = {
        "stage": "p0a48-train-only-mining-v1",
        "created_ts": created,
        "served_model_id": model_id,
        "sample_id": row["sample_id"],
        "correct": prediction == str(row["answer"]).strip().upper(),
        "prediction": prediction,
        "latency_ms": latency,
        "generation_error": error,
        "response_text": response,
    }
    item["row_hash"] = sha256_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return index, item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise EvaluationError("workers must be in [1,16]")
    rows = read_rows()
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    created = datetime.now(timezone.utc).isoformat()
    progress = TRACE.with_name("mining_progress.jsonl")
    slots: list[dict[str, Any] | None] = [None] * len(rows)
    positions = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    if progress.is_file():
        for line in progress.open(encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", ""))
            if sample_id not in positions or item.get("served_model_id") != model_id:
                raise EvaluationError("Stale P0-A48 mining progress")
            slots[positions[sample_id]] = item
    pending = [(index, row) for index, row in enumerate(rows) if slots[index] is None]
    completed = len(rows) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_one, index, row, args, model_id, created) for index, row in pending]
        for future in as_completed(futures):
            index, item = future.result()
            slots[index] = item
            completed += 1
            if completed == 1 or completed % 250 == 0 or completed == len(rows):
                write_jsonl_atomic(progress, [value for value in slots if value is not None])
                correct = sum(int(bool(value and value["correct"])) for value in slots)
                print(f"[p0a48-mining] {completed}/{len(rows)} correct_so_far={correct}", flush=True)
    trace = [value for value in slots if value is not None]
    if len(trace) != len(rows):
        raise EvaluationError("Incomplete P0-A48 mining trace")
    write_jsonl_atomic(TRACE, trace)
    progress.unlink(missing_ok=True)
    correct = sum(int(row["correct"]) for row in trace)
    errors = sum(int(bool(row["generation_error"])) for row in trace)
    audit = {
        "gate": "P0-A48-TRAIN-ONLY-MINING",
        "check_version": "1.0",
        "created_by": "scripts/mine_p0a48_nlp.py",
        "created_ts": created,
        "status": "passed" if errors == 0 else "failed",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_hash": sha256_file(SOURCE),
        "served_model_id": model_id,
        "sample_count": len(trace),
        "correct_count": correct,
        "failure_count": len(trace) - correct,
        "accuracy": correct / len(trace),
        "generation_error_count": errors,
        "trace": TRACE.relative_to(ROOT).as_posix(),
        "trace_hash": sha256_file(TRACE),
        "formal_test_items_loaded": 0,
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json_atomic(AUDIT, audit)
    print(f"Wrote {AUDIT.relative_to(ROOT)} status={audit['status']} correct={correct}/{len(trace)}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A48 mining failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
