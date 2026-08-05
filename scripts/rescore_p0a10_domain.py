#!/usr/bin/env python3
"""Deterministically rescore a saved P0-A10 train-only response trace."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a10_domain import display, read_manifest, resolve, validate_output
from evaluate_p0a6_internal import (
    EvaluationError,
    score_row,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=("math", "code", "nlp"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-trace", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    return parser.parse_args()


def read_trace(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    allowed_root = (ROOT / "reports/audit/p0a10").resolve()
    if not path.is_file() or allowed_root not in path.parents:
        raise EvaluationError(f"Invalid rescore source: {display(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("sample_id") or "response_text" not in row:
                raise EvaluationError(f"Invalid trace row {line_number}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise EvaluationError(f"Expected {expected_rows} trace rows, found {len(rows)}")
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    input_path = resolve(args.input_trace)
    output_path = resolve(args.output_trace)
    audit_path = resolve(args.audit)
    validate_output(output_path, ".jsonl")
    validate_output(audit_path, ".json")
    manifest_rows = read_manifest(manifest_path, args.domain, args.expected_rows)
    source_rows = read_trace(input_path, args.expected_rows)
    created_ts = datetime.now(timezone.utc).isoformat()
    rescored: list[dict[str, Any]] = []
    for manifest_row, source_row in zip(manifest_rows, source_rows, strict=True):
        if source_row["sample_id"] != manifest_row["sample_id"]:
            raise EvaluationError(
                f"Trace/manifest mismatch: {source_row['sample_id']} != "
                f"{manifest_row['sample_id']}"
            )
        response = str(source_row["response_text"])
        correct, prediction, detail, canonical = score_row(
            manifest_row, response, args.code_timeout_sec
        )
        item = dict(source_row)
        item.update(
            {
                "capability_eval_version": "p0a10-domain-v2-rescore",
                "created_ts": created_ts,
                "correct": bool(correct),
                "canonical_format": bool(canonical),
                "prediction": prediction,
                "score_detail": detail,
            }
        )
        item.pop("row_hash", None)
        item["row_hash"] = sha256_text(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
        )
        rescored.append(item)

    write_jsonl_atomic(output_path, rescored)
    correct_count = sum(int(row["correct"]) for row in rescored)
    canonical_count = sum(int(row["canonical_format"]) for row in rescored)
    error_count = sum(int(bool(row.get("generation_error"))) for row in rescored)
    served_models = sorted({str(row.get("served_model_id", "")) for row in rescored})
    candidate_names = sorted({str(row.get("candidate_name", "")) for row in rescored})
    audit = {
        "gate": "P0-A10-TRAIN-ONLY-VALIDATION",
        "check_version": "2.0-rescore",
        "created_by": "scripts/rescore_p0a10_domain.py",
        "created_ts": created_ts,
        "status": "passed" if error_count == 0 else "failed",
        "domain": args.domain,
        "candidate_name": candidate_names[0] if len(candidate_names) == 1 else candidate_names,
        "served_model_id": served_models[0] if len(served_models) == 1 else served_models,
        "manifest": display(manifest_path),
        "manifest_hash": sha256_file(manifest_path),
        "sample_count": len(rescored),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rescored),
        "canonical_format_rate": canonical_count / len(rescored),
        "generation_error_count": error_count,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in rescored)
        / len(rescored),
        "trace": display(output_path),
        "trace_hash": sha256_file(output_path),
        "rescore_source": display(input_path),
        "rescore_source_hash": sha256_file(input_path),
        "gate300_loaded": False,
        "formal_full_loaded": False,
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
