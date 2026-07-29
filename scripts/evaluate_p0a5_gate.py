#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPRESSION = ROOT / "model_compression"
if str(MODEL_COMPRESSION) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPRESSION))

from build_p0a5_data import extract_code, safe_python, sandbox_limits, sha256_file, sha256_text


class EvaluationError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise EvaluationError(f"Non-object manifest row {line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Endpoint request failed: {url}: {exc}") from exc


def discover_model(endpoint: str, requested: str, timeout: float) -> str:
    payload = request_json(endpoint.rstrip("/") + "/v1/models", None, timeout)
    ids = [
        str(item.get("id", ""))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if requested and requested in ids:
        return requested
    if len(ids) == 1:
        return ids[0]
    if requested:
        return requested
    raise EvaluationError(f"Could not infer endpoint model id: {ids}")


def generate(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> tuple[str, float]:
    started = time.perf_counter()
    payload = request_json(
        endpoint.rstrip("/") + "/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        text = str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise EvaluationError(f"Malformed completion response: {payload}") from exc
    return text, latency_ms


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    return cleaned[:-2] if cleaned.endswith(".0") else cleaned


def extract_number(value: str) -> str:
    if "####" in value:
        value = value.rsplit("####", 1)[1]
    matches = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_choice(value: str) -> str:
    upper = value.strip().upper()
    matches = re.findall(
        r"(?:最终答案|答案|ANSWER|OPTION|选项)\s*[:：]?\s*([ABCD])\b",
        upper,
    )
    if matches:
        return matches[-1]
    fallback = re.findall(r"\b([ABCD])\b", upper)
    return fallback[-1] if fallback else ""


def score_code(response: str, tests: list[str], timeout: float) -> tuple[bool, str]:
    code = extract_code(response)
    source = code + "\n\n" + "\n".join(str(test) for test in tests) + "\n"
    if not safe_python(source):
        return False, "unsafe_or_invalid_python"
    with tempfile.TemporaryDirectory(prefix="p0a5-gate-") as temp_dir:
        program = Path(temp_dir) / "main.py"
        program.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(program)],
                cwd=temp_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=sandbox_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    detail = "passed" if completed.returncode == 0 else (
        completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip()
        else f"returncode={completed.returncode}"
    )
    return completed.returncode == 0, detail[:500]


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    domain = str(row["domain"])
    if domain == "math":
        instruction = "Solve the problem concisely. End with `#### <number>`."
    elif domain == "code":
        instruction = (
            "Return only a complete Python function implementation in one python code block. "
            "Do not use files, network access, third-party packages, or explanatory prose."
        )
    elif domain == "nlp":
        instruction = (
            "请简要分析这道中文选择题，并在最后一行严格输出“最终答案：X”，"
            "其中X只能是A、B、C或D。"
        )
    else:
        raise EvaluationError(f"Unsupported domain: {domain}")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": str(row["prompt"])},
    ]


def score(row: dict[str, Any], response: str, code_timeout: float) -> tuple[bool, str, str]:
    domain = str(row["domain"])
    if domain == "math":
        prediction = extract_number(response)
        reference = normalize_number(str(row["reference"]))
        return prediction == reference, prediction, f"reference={reference}"
    if domain == "nlp":
        prediction = extract_choice(response)
        reference = str(row["reference"]).strip().upper()
        return prediction == reference, prediction, f"reference={reference}"
    if domain == "code":
        passed, detail = score_code(response, list(row["unit_tests"]), code_timeout)
        return passed, "pass" if passed else "fail", detail
    raise EvaluationError(f"Unsupported domain: {domain}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the single P0-A5 300-item gate.")
    parser.add_argument("--manifest", default="data/capability_v2/gate300.jsonl")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    parser.add_argument("--max-tokens-math", type=int, default=512)
    parser.add_argument("--max-tokens-code", type=int, default=768)
    parser.add_argument("--max-tokens-nlp", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = resolve_path(args.manifest)
        rows = read_jsonl(manifest_path)
        counts = Counter(str(row.get("domain", "")) for row in rows)
        if counts != Counter({"math": 100, "code": 100, "nlp": 100}):
            raise EvaluationError(f"Gate manifest counts are not 100/100/100: {counts}")
        model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
        token_limits = {
            "math": args.max_tokens_math,
            "code": args.max_tokens_code,
            "nlp": args.max_tokens_nlp,
        }
        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
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
                    token_limits[str(row["domain"])],
                    args.timeout_sec,
                )
                correct, prediction, detail = score(row, response, args.code_timeout_sec)
            except EvaluationError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct = False
                prediction = ""
                detail = str(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
            result = {
                "capability_eval_version": "p0a5-gate300-v1",
                "created_ts": created_ts,
                "candidate_name": args.candidate_name,
                "served_model_id": model_id,
                "domain": row["domain"],
                "dataset_key": row["dataset_key"],
                "sample_id": row["sample_id"],
                "prompt_hash": prompt_hash,
                "prediction": prediction,
                "correct": correct,
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
                f"[{index}/300] {row['domain']} correct={correct} "
                f"latency_ms={latency_ms:.1f}",
                flush=True,
            )
        output_path = resolve_path(args.output_trace)
        write_jsonl(output_path, trace)
        correct_counts = Counter(
            str(row["domain"]) for row in trace if row["correct"] is True
        )
        generation_errors = sum(bool(row["generation_error"]) for row in trace)
        accuracy = {domain: correct_counts[domain] / 100 for domain in ("math", "code", "nlp")}
        audit = {
            "gate": "P0-A5-GATE300-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a5_gate.py",
            "created_ts": created_ts,
            "status": "passed" if generation_errors == 0 else "failed",
            "candidate_name": args.candidate_name,
            "endpoint": args.endpoint,
            "served_model_id": model_id,
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
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path = resolve_path(args.audit)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"accuracy={accuracy} generation_errors={generation_errors}")
        return 0 if audit["status"] == "passed" else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 gate evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
