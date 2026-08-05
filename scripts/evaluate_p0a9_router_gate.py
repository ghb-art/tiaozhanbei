#!/usr/bin/env python3
"""Evaluate the frozen P0-A8 Top-1 router on the P0-A5 300-item gate."""

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
    display_path,
    generate,
    read_jsonl,
    resolve_path,
    score,
    sha256_file,
    sha256_text,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST = (ROOT / "data/capability_v2/gate300.jsonl").resolve()
DOMAINS = ("math", "code", "nlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/capability_v2/gate300.jsonl")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id-math", required=True)
    parser.add_argument("--model-id-code", required=True)
    parser.add_argument("--model-id-nlp", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    parser.add_argument("--max-tokens-math", type=int, default=512)
    parser.add_argument("--max-tokens-code", type=int, default=768)
    parser.add_argument("--max-tokens-nlp", type=int, default=256)
    return parser.parse_args()


def validate_output(path: Path, parent: Path, suffix: str) -> None:
    resolved = path.resolve()
    if resolved.suffix != suffix or parent.resolve() not in resolved.parents:
        raise EvaluationError(f"Unexpected P0-A9 output path: {display_path(resolved)}")


def main() -> int:
    args = parse_args()
    try:
        manifest_path = resolve_path(args.manifest).resolve()
        if manifest_path != EXPECTED_MANIFEST or not manifest_path.is_file():
            raise EvaluationError(f"Unexpected gate manifest: {display_path(manifest_path)}")
        rows = read_jsonl(manifest_path)
        counts = Counter(str(row.get("domain", "")) for row in rows)
        if counts != Counter({"math": 100, "code": 100, "nlp": 100}):
            raise EvaluationError(f"Gate manifest counts are not 100/100/100: {counts}")
        requested = {
            "math": args.model_id_math,
            "code": args.model_id_code,
            "nlp": args.model_id_nlp,
        }
        model_ids = {
            domain: discover_model(args.endpoint, requested[domain], args.timeout_sec)
            for domain in DOMAINS
        }
        token_limits = {
            "math": args.max_tokens_math,
            "code": args.max_tokens_code,
            "nlp": args.max_tokens_nlp,
        }
        output_path = resolve_path(args.output_trace)
        audit_path = resolve_path(args.audit)
        validate_output(output_path, ROOT / "data/eval", ".jsonl")
        validate_output(audit_path, ROOT / "reports/audit", ".json")
        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            domain = str(row["domain"])
            model_id = model_ids[domain]
            messages = build_messages(row)
            prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            generation_error = ""
            response = ""
            started = time.perf_counter()
            try:
                response, latency_ms = generate(
                    args.endpoint,
                    model_id,
                    messages,
                    token_limits[domain],
                    args.timeout_sec,
                )
                correct, prediction, detail = score(row, response, args.code_timeout_sec)
            except EvaluationError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct, prediction, detail = False, "", str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            result = {
                "capability_eval_version": "p0a5-gate300-v2",
                "created_ts": created_ts,
                "candidate_name": args.candidate_name,
                "served_model_id": model_id,
                "domain": domain,
                "dataset_key": row["dataset_key"],
                "sample_id": row["sample_id"],
                "prompt_hash": prompt_hash,
                "prediction": prediction,
                "correct": bool(correct),
                "score_detail": detail,
                "latency_ms": latency_ms,
                "generation_error": generation_error,
                "response_text": response,
            }
            result["row_hash"] = sha256_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
            trace.append(result)
            print(
                f"[{index}/300] {domain} correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(output_path, trace)
        correct_counts = Counter(
            str(row["domain"]) for row in trace if row["correct"] is True
        )
        generation_errors = sum(bool(row["generation_error"]) for row in trace)
        accuracy = {domain: correct_counts[domain] / 100 for domain in DOMAINS}
        audit = {
            "gate": "P0-A9-ROUTER-GATE300-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a9_router_gate.py",
            "created_ts": created_ts,
            "status": "passed" if generation_errors == 0 else "failed",
            "candidate_name": args.candidate_name,
            "endpoint": args.endpoint,
            "served_model_id_by_domain": model_ids,
            "manifest": display_path(manifest_path),
            "manifest_hash": sha256_file(manifest_path),
            "output_trace": display_path(output_path),
            "output_trace_hash": sha256_file(output_path),
            "counts": dict(sorted(counts.items())),
            "correct_counts": dict(sorted(correct_counts.items())),
            "accuracy_by_domain": accuracy,
            "generation_error_count": generation_errors,
            "max_tokens": token_limits,
            "code_timeout_sec": args.code_timeout_sec,
            "formal_full_loaded": False,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(audit_path)
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"accuracy={accuracy} generation_errors={generation_errors}")
        return 0 if audit["status"] == "passed" else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A9 router gate evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
