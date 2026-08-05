#!/usr/bin/env python3
"""Evaluate the frozen P0-A13 router with thinking enabled only for Math."""

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
EXPECTED_SELECTION = ROOT / "reports/audit/p0a13/runtime_selection.json"
DOMAINS = ("math", "code", "nlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id-math", required=True)
    parser.add_argument("--model-id-code", required=True)
    parser.add_argument("--model-id-nlp", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    return parser.parse_args()


def require_runtime_selection() -> dict[str, Any]:
    if not EXPECTED_SELECTION.is_file():
        raise EvaluationError("Missing P0-A13 runtime selection")
    data = json.loads(EXPECTED_SELECTION.read_text(encoding="utf-8"))
    if data.get("status") != "passed" or data.get("selected_runtime") != "thinking_on":
        raise EvaluationError("P0-A13 runtime selection did not pass")
    return data


def main() -> int:
    args = parse_args()
    try:
        selection = require_runtime_selection()
        manifest_path = EXPECTED_MANIFEST
        rows = read_jsonl(manifest_path)
        counts = Counter(str(row.get("domain", "")) for row in rows)
        if counts != Counter({"math": 100, "code": 100, "nlp": 100}):
            raise EvaluationError(f"Gate manifest counts are not 100/100/100: {counts}")
        requested = {
            "math": args.model_id_math,
            "code": args.model_id_code,
            "nlp": args.model_id_nlp,
        }
        models = {
            domain: discover_model(args.endpoint, requested[domain], args.timeout_sec)
            for domain in DOMAINS
        }
        token_limits = {"math": 768, "code": 768, "nlp": 256}
        output_path = resolve_path(args.output_trace).resolve()
        audit_path = resolve_path(args.audit).resolve()
        if output_path != (ROOT / "data/eval/p0a13_router_hf_gate300.jsonl").resolve():
            raise EvaluationError(f"Unexpected P0-A13 trace: {display_path(output_path)}")
        if audit_path != (ROOT / "reports/audit/gate_p0a13_router_hf_gate300_eval.json").resolve():
            raise EvaluationError(f"Unexpected P0-A13 audit: {display_path(audit_path)}")
        if output_path.exists() or audit_path.exists():
            raise EvaluationError("P0-A13 gate artifacts already exist; repeated run refused")

        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            domain = str(row["domain"])
            thinking = domain == "math"
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
                    enable_thinking=thinking,
                )
                correct, prediction, detail = score(row, response, args.code_timeout_sec)
            except EvaluationError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct, prediction, detail = False, "", str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            result = {
                "capability_eval_version": "p0a13-gate300-v1",
                "created_ts": created_ts,
                "candidate_name": args.candidate_name,
                "served_model_id": models[domain],
                "domain": domain,
                "dataset_key": row["dataset_key"],
                "sample_id": row["sample_id"],
                "prompt_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                "thinking": thinking,
                "max_tokens": token_limits[domain],
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
                f"[{index}/300] {domain} thinking={thinking} "
                f"correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(output_path, trace)
        correct_counts = Counter(
            str(row["domain"]) for row in trace if row["correct"] is True
        )
        generation_errors = sum(bool(row["generation_error"]) for row in trace)
        accuracy = {domain: correct_counts[domain] / 100 for domain in DOMAINS}
        audit = {
            "gate": "P0-A13-ROUTER-GATE300-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a13_router_gate.py",
            "created_ts": created_ts,
            "status": "passed" if generation_errors == 0 else "failed",
            "candidate_name": args.candidate_name,
            "served_model_id_by_domain": models,
            "runtime_by_domain": {
                "math": {"thinking": True, "max_tokens": 768},
                "code": {"thinking": False, "max_tokens": 768},
                "nlp": {"thinking": False, "max_tokens": 256},
            },
            "runtime_selection": EXPECTED_SELECTION.relative_to(ROOT).as_posix(),
            "runtime_selection_hash": sha256_file(EXPECTED_SELECTION),
            "runtime_selection_gain": selection["gain"],
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_hash": sha256_file(manifest_path),
            "output_trace": output_path.relative_to(ROOT).as_posix(),
            "output_trace_hash": sha256_file(output_path),
            "counts": dict(sorted(counts.items())),
            "correct_counts": dict(sorted(correct_counts.items())),
            "accuracy_by_domain": accuracy,
            "generation_error_count": generation_errors,
            "formal_full_loaded": False,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"accuracy={accuracy} generation_errors={generation_errors}")
        return 0 if audit["status"] == "passed" else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A13 gate evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
