#!/usr/bin/env python3
"""Evaluate the frozen best Math/Code/NLP routes on the 300-item gate once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
OUTPUT = ROOT / "data/eval/p0a41_best_router_hf_gate300.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a41_best_router_hf_gate300_eval.json"
DOMAINS = ("math", "code", "nlp")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id-math", required=True)
    parser.add_argument("--model-id-code", required=True)
    parser.add_argument("--model-id-nlp", required=True)
    parser.add_argument("--timeout-sec", type=float, default=180)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    args = parser.parse_args()
    try:
        if OUTPUT.exists() or AUDIT.exists():
            raise EvaluationError("P0-A41 gate artifacts already exist; repeated run refused")
        rows = read_jsonl(MANIFEST)
        counts = Counter(str(row.get("domain", "")) for row in rows)
        if counts != Counter({"math": 100, "code": 100, "nlp": 100}):
            raise EvaluationError(f"Unexpected P0-A41 counts: {counts}")
        requested = {
            "math": args.model_id_math,
            "code": args.model_id_code,
            "nlp": args.model_id_nlp,
        }
        models = {
            domain: discover_model(args.endpoint, requested[domain], args.timeout_sec)
            for domain in DOMAINS
        }
        token_limits = {"math": 512, "code": 768, "nlp": 256}
        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            domain = str(row["domain"])
            messages = build_messages(row)
            response = ""
            generation_error = ""
            started = time.perf_counter()
            try:
                response, latency_ms = generate(
                    args.endpoint,
                    models[domain],
                    messages,
                    token_limits[domain],
                    args.timeout_sec,
                    enable_thinking=False,
                )
                correct, prediction, detail = score(row, response, args.code_timeout_sec)
            except (EvaluationError, OSError, ValueError) as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct, prediction, detail = False, "", str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            item = {
                "capability_eval_version": "p0a41-best-router-gate300-v1",
                "created_ts": created_ts,
                "candidate_name": "p0a41-best-router-hf",
                "served_model_id": models[domain],
                "domain": domain,
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
                f"[{index}/300] {domain} correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(OUTPUT, trace)
        correct_counts = Counter(
            str(row["domain"]) for row in trace if row["correct"] is True
        )
        errors = sum(bool(row["generation_error"]) for row in trace)
        accuracy = {domain: correct_counts[domain] / 100 for domain in DOMAINS}
        audit = {
            "gate": "P0-A41-BEST-ROUTER-GATE300-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a41_best_router_gate.py",
            "created_ts": created_ts,
            "status": "passed" if errors == 0 else "failed",
            "candidate_name": "p0a41-best-router-hf",
            "served_model_id_by_domain": models,
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_hash": sha256_file(MANIFEST),
            "counts": dict(sorted(counts.items())),
            "correct_counts": dict(sorted(correct_counts.items())),
            "accuracy_by_domain": accuracy,
            "generation_error_count": errors,
            "max_tokens": token_limits,
            "thinking": "off",
            "code_timeout_sec": args.code_timeout_sec,
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
        print(f"accuracy={accuracy} generation_errors={errors}")
        return 0 if errors == 0 else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A41 evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
