#!/usr/bin/env python3
"""Run the selected P0-A23 Code adapter once on the frozen Code100 gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a5_gate import (
    EvaluationError,
    build_messages,
    discover_model,
    generate,
    read_jsonl,
    score,
    sha256_file,
    sha256_text,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/capability_v2/gate300.jsonl"
OUTPUT = ROOT / "data/eval/p0a24_code_gate100.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a24_code_gate100_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--selected-step", type=int, choices=(96, 192), required=True)
    parser.add_argument("--timeout-sec", type=float, default=180)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    args = parser.parse_args()
    try:
        if OUTPUT.exists() or AUDIT.exists():
            raise EvaluationError("P0-A24 gate artifacts already exist; repeated run refused")
        rows = [row for row in read_jsonl(MANIFEST) if row.get("domain") == "code"]
        if len(rows) != 100:
            raise EvaluationError(f"Unexpected P0-A24 Code count: {len(rows)}")
        served_model = discover_model(args.endpoint, args.model_id, args.timeout_sec)
        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            messages = build_messages(row)
            response = ""
            generation_error = ""
            started = time.perf_counter()
            try:
                response, latency_ms = generate(
                    args.endpoint,
                    served_model,
                    messages,
                    768,
                    args.timeout_sec,
                    enable_thinking=False,
                )
                correct, prediction, detail = score(row, response, args.code_timeout_sec)
            except (EvaluationError, OSError, ValueError) as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct, prediction, detail = False, "", str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            item = {
                "capability_eval_version": "p0a24-code-gate-v1",
                "created_ts": created_ts,
                "served_model_id": served_model,
                "selected_step": args.selected_step,
                "domain": "code",
                "dataset_key": row["dataset_key"],
                "sample_id": row["sample_id"],
                "prompt_hash": sha256_text(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ),
                "prediction": prediction,
                "correct": bool(correct),
                "score_detail": detail,
                "latency_ms": latency_ms,
                "generation_error": generation_error,
                "response_text": response,
            }
            item["row_hash"] = sha256_text(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
            )
            trace.append(item)
            print(
                f"[{index}/100] code correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(OUTPUT, trace)
        correct_count = sum(row["correct"] is True for row in trace)
        errors = sum(bool(row["generation_error"]) for row in trace)
        audit = {
            "gate": "P0-A24-CODE-GATE100-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a24_code_gate.py",
            "created_ts": created_ts,
            "status": "passed" if errors == 0 else "failed",
            "served_model_id": served_model,
            "selected_step": args.selected_step,
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_hash": sha256_file(MANIFEST),
            "code_rows_loaded": 100,
            "math_rows_loaded": 0,
            "nlp_rows_loaded": 0,
            "correct_count": correct_count,
            "accuracy": correct_count / 100,
            "generation_error_count": errors,
            "thinking": "off",
            "max_tokens": 768,
            "output_trace": OUTPUT.relative_to(ROOT).as_posix(),
            "output_trace_hash": sha256_file(OUTPUT),
            "formal_full_loaded": False,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        temporary = AUDIT.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(AUDIT)
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        print(f"Wrote {AUDIT.relative_to(ROOT)}")
        print(f"accuracy={correct_count / 100:.6f} generation_errors={errors}")
        return 0 if errors == 0 else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A24 evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
