#!/usr/bin/env python3
"""Evaluate one P0-A10 Adapter on its untouched train-only holdout."""

from __future__ import annotations

import argparse
import json
import time
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
DOMAINS = ("math", "code", "nlp")


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
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    return parser.parse_args()


def read_manifest(path: Path, domain: str, expected: int) -> list[dict[str, Any]]:
    expected_path = (ROOT / f"data/p0a10/{domain}_validation.jsonl").resolve()
    if path != expected_path or not path.is_file():
        raise EvaluationError(f"Unexpected P0-A10 manifest: {display(path)}")
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
            if row.get("domain") != domain:
                raise EvaluationError(f"Wrong domain on row {line_number}")
            if row.get("split_role") != "p0a10_internal_validation":
                raise EvaluationError(f"Wrong split on row {line_number}")
            if domain == "code" and not row.get("unit_tests"):
                raise EvaluationError(f"Missing unit tests on row {line_number}")
            rows.append(row)
    if len(rows) != expected:
        raise EvaluationError(f"Expected {expected} rows, found {len(rows)}")
    return rows


def validate_output(path: Path, suffix: str) -> None:
    root = (ROOT / "reports/audit/p0a10").resolve()
    if path.suffix != suffix or root not in path.parents:
        raise EvaluationError(f"Invalid output: {display(path)}")


def main() -> int:
    args = parse_args()
    manifest = resolve(args.manifest)
    trace_path = resolve(args.output_trace)
    audit_path = resolve(args.audit)
    validate_output(trace_path, ".jsonl")
    validate_output(audit_path, ".json")
    rows = read_manifest(manifest, args.domain, args.expected_rows)
    model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
    token_limits = {"math": 512, "code": 768, "nlp": 256}
    created_ts = datetime.now(timezone.utc).isoformat()
    trace: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        messages = build_messages(row)
        response = ""
        generation_error = ""
        started = time.perf_counter()
        try:
            response, latency_ms = generate(
                args.endpoint, model_id, messages, token_limits[args.domain], args.timeout_sec
            )
            correct, prediction, detail, canonical = score_row(
                row, response, args.code_timeout_sec
            )
        except (EvaluationError, OSError, ValueError) as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            correct, prediction, detail, canonical = False, "", str(exc), False
            generation_error = f"{type(exc).__name__}: {exc}"
        item = {
            "capability_eval_version": "p0a10-domain-v1",
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
        trace.append(item)
        print(
            f"[{index}/{len(rows)}] {args.domain} correct={correct} "
            f"canonical={canonical} latency_ms={latency_ms:.1f}", flush=True
        )
    write_jsonl_atomic(trace_path, trace)
    correct_count = sum(int(row["correct"]) for row in trace)
    canonical_count = sum(int(row["canonical_format"]) for row in trace)
    error_count = sum(int(bool(row["generation_error"])) for row in trace)
    audit = {
        "gate": "P0-A10-TRAIN-ONLY-VALIDATION",
        "check_version": "1.0",
        "created_by": "scripts/evaluate_p0a10_domain.py",
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
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
