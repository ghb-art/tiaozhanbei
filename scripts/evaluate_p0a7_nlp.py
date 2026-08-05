#!/usr/bin/env python3
"""Evaluate an NLP Adapter on the untouched MMLU-aux train-only holdout."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a6_internal import (
    EvaluationError,
    discover_model,
    extract_choice,
    generate,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "请简要分析这道中文选择题。最后一行必须严格使用“最终答案：A”的格式，"
    "并将A替换为实际的A、B、C或D选项。"
)


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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=120)
    return parser.parse_args()


def read_manifest(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or path.name != "nlp_mmlu_aux_validation.jsonl":
        raise EvaluationError(f"Unexpected P0-A7 validation manifest: {display(path)}")
    if path.parent != (ROOT / "data/p0a7").resolve():
        raise EvaluationError("P0-A7 validation must remain inside data/p0a7")
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
            seen.add(sample_id)
            if row.get("dataset_key") != "mmlu_aux_chinese":
                raise EvaluationError(f"Unapproved dataset on row {line_number}")
            if row.get("split_role") != "p0a7_internal_validation":
                raise EvaluationError(f"Unapproved split role on row {line_number}")
            if str(row.get("reference", "")) not in {"A", "B", "C", "D"}:
                raise EvaluationError(f"Invalid label on row {line_number}")
            searchable = " ".join(
                str(row.get(key, ""))
                for key in ("sample_id", "source", "split_role", "dataset_key")
            ).casefold()
            if any(marker in searchable for marker in ("cmmlu/test", "mmlu/test", "formal_test")):
                raise EvaluationError(f"Formal-test reference on row {line_number}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise EvaluationError(f"Expected {expected_rows} rows, found {len(rows)}")
    return rows


def validate_output(path: Path, suffix: str) -> None:
    root = (ROOT / "reports/audit/p0a7").resolve()
    if path.suffix != suffix or root not in path.parents:
        raise EvaluationError(f"Invalid P0-A7 output path: {display(path)}")


def main() -> int:
    args = parse_args()
    manifest = resolve(args.manifest)
    trace_path = resolve(args.output_trace)
    audit_path = resolve(args.audit)
    validate_output(trace_path, ".jsonl")
    validate_output(audit_path, ".json")
    rows = read_manifest(manifest, args.expected_rows)
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    created_ts = datetime.now(timezone.utc).isoformat()
    trace: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["prompt"])},
        ]
        started = time.perf_counter()
        response = ""
        generation_error = ""
        try:
            response, latency_ms = generate(
                args.endpoint, model_id, messages, args.max_tokens, args.timeout_sec
            )
            prediction, canonical = extract_choice(response)
        except (EvaluationError, OSError, ValueError) as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            prediction, canonical = "", False
            generation_error = f"{type(exc).__name__}: {exc}"
        correct = prediction == str(row["reference"])
        item = {
            "capability_eval_version": "p0a7-nlp-internal-v1",
            "created_ts": created_ts,
            "candidate_name": args.candidate_name,
            "served_model_id": model_id,
            "sample_id": row["sample_id"],
            "prompt_hash": sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ),
            "prediction": prediction,
            "correct": correct,
            "canonical_format": canonical,
            "latency_ms": latency_ms,
            "generation_error": generation_error,
            "response_text": response,
        }
        item["row_hash"] = sha256_text(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
        )
        trace.append(item)
        print(
            f"[{index}/{len(rows)}] nlp correct={correct} canonical={canonical} "
            f"latency_ms={latency_ms:.1f}",
            flush=True,
        )
    write_jsonl_atomic(trace_path, trace)
    correct_count = sum(int(row["correct"]) for row in trace)
    canonical_count = sum(int(row["canonical_format"]) for row in trace)
    error_count = sum(int(bool(row["generation_error"])) for row in trace)
    audit = {
        "gate": "P0-A7-NLP-TRAIN-ONLY-VALIDATION",
        "check_version": "1.0",
        "created_by": "scripts/evaluate_p0a7_nlp.py",
        "created_ts": created_ts,
        "status": "passed" if error_count == 0 else "failed",
        "candidate_name": args.candidate_name,
        "served_model_id": model_id,
        "manifest": display(manifest),
        "manifest_hash": sha256_file(manifest),
        "sample_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
        "canonical_format_rate": canonical_count / len(rows),
        "generation_error_count": error_count,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace) / len(rows),
        "trace": display(trace_path),
        "trace_hash": sha256_file(trace_path),
        "formal_test_loaded": False,
        "cmmlu_test_loaded": False,
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    write_json_atomic(audit_path, audit)
    print(f"Wrote {display(audit_path)}")
    print(
        f"status={audit['status']} accuracy={audit['accuracy']:.6f} "
        f"canonical={audit['canonical_format_rate']:.6f}"
    )
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
