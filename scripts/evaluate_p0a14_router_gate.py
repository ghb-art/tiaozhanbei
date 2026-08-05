#!/usr/bin/env python3
"""Run the frozen P0-A14 gate with vote3 only on Math."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_p0a14_math import generate_vote, stable_seed
from evaluate_p0a5_gate import (
    EvaluationError,
    build_messages,
    discover_model,
    display_path,
    generate,
    read_jsonl,
    score,
    sha256_file,
    sha256_text,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/capability_v2/gate300.jsonl"
SELECTION = ROOT / "reports/audit/p0a14/runtime_selection.json"
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
    parser.add_argument("--timeout-sec", type=float, default=180)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not SELECTION.is_file():
            raise EvaluationError("Missing P0-A14 runtime selection")
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        if selection.get("status") != "passed" or selection.get("selected_runtime") != "thinking_vote3":
            raise EvaluationError("P0-A14 runtime selection did not pass")
        rows = read_jsonl(MANIFEST)
        counts = Counter(str(row.get("domain", "")) for row in rows)
        if counts != Counter({"math": 100, "code": 100, "nlp": 100}):
            raise EvaluationError(f"Unexpected gate counts: {counts}")
        requested = {
            "math": args.model_id_math,
            "code": args.model_id_code,
            "nlp": args.model_id_nlp,
        }
        models = {
            domain: discover_model(args.endpoint, requested[domain], args.timeout_sec)
            for domain in DOMAINS
        }
        output = Path(args.output_trace)
        output = (output if output.is_absolute() else ROOT / output).resolve()
        audit_path = Path(args.audit)
        audit_path = (audit_path if audit_path.is_absolute() else ROOT / audit_path).resolve()
        expected_output = (ROOT / "data/eval/p0a14_router_hf_gate300.jsonl").resolve()
        expected_audit = (ROOT / "reports/audit/gate_p0a14_router_hf_gate300_eval.json").resolve()
        if output != expected_output or audit_path != expected_audit:
            raise EvaluationError("Unexpected P0-A14 gate output path")
        if output.exists() or audit_path.exists():
            raise EvaluationError("P0-A14 gate artifacts already exist; repeated run refused")

        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            domain = str(row["domain"])
            messages = build_messages(row)
            responses: list[str] = []
            predictions: list[str] = []
            selected_index = 0
            vote_tied = False
            generation_error = ""
            started = time.perf_counter()
            try:
                if domain == "math":
                    responses, predictions, _, selected_index, vote_tied, signed_latency = generate_vote(
                        args.endpoint,
                        models[domain],
                        messages,
                        args.timeout_sec,
                        stable_seed(str(row["sample_id"])),
                    )
                    latency_ms = abs(signed_latency)
                    selected_response = responses[selected_index]
                else:
                    selected_response, latency_ms = generate(
                        args.endpoint,
                        models[domain],
                        messages,
                        768 if domain == "code" else 256,
                        args.timeout_sec,
                        enable_thinking=False,
                    )
                    responses = [selected_response]
                correct, prediction, detail = score(row, selected_response, args.code_timeout_sec)
            except (EvaluationError, OSError, ValueError) as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct, prediction, detail = False, "", str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            item = {
                "capability_eval_version": "p0a14-gate300-v1",
                "created_ts": created_ts,
                "candidate_name": args.candidate_name,
                "served_model_id": models[domain],
                "domain": domain,
                "dataset_key": row["dataset_key"],
                "sample_id": row["sample_id"],
                "prompt_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                "runtime": "thinking_vote3" if domain == "math" else "single_thinking_off",
                "vote_predictions": predictions,
                "selected_index": selected_index,
                "vote_tied": vote_tied,
                "prediction": prediction,
                "correct": bool(correct),
                "score_detail": detail,
                "latency_ms": latency_ms,
                "generation_error": generation_error,
                "response_texts": responses,
            }
            item["row_hash"] = sha256_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
            trace.append(item)
            print(
                f"[{index}/300] {domain} runtime={item['runtime']} "
                f"correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        write_jsonl(output, trace)
        correct_counts = Counter(str(row["domain"]) for row in trace if row["correct"])
        generation_errors = sum(bool(row["generation_error"]) for row in trace)
        accuracy = {domain: correct_counts[domain] / 100 for domain in DOMAINS}
        audit = {
            "gate": "P0-A14-ROUTER-GATE300-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a14_router_gate.py",
            "created_ts": created_ts,
            "status": "passed" if generation_errors == 0 else "failed",
            "candidate_name": args.candidate_name,
            "served_model_id_by_domain": models,
            "runtime_by_domain": {
                "math": "thinking_vote3",
                "code": "single_thinking_off",
                "nlp": "single_thinking_off",
            },
            "runtime_selection": SELECTION.relative_to(ROOT).as_posix(),
            "runtime_selection_hash": sha256_file(SELECTION),
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_hash": sha256_file(MANIFEST),
            "output_trace": output.relative_to(ROOT).as_posix(),
            "output_trace_hash": sha256_file(output),
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
        temporary = audit_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(audit_path)
        print(f"Wrote {display_path(output)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"accuracy={accuracy} generation_errors={generation_errors}")
        return 0 if generation_errors == 0 else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A14 gate evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
