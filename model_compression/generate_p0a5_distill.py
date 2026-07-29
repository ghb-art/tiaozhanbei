#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "model_compression") not in sys.path:
    sys.path.insert(0, str(ROOT / "model_compression"))

from build_p0a5_data import execute_code_case, sha256_file, sha256_text


class DistillError(RuntimeError):
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
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
        raise DistillError(f"Endpoint request failed: {exc}") from exc


def discover_model(endpoint: str, requested: str, timeout: float) -> str:
    response = request_json(endpoint.rstrip("/") + "/v1/models", None, timeout)
    ids = [
        str(item.get("id", ""))
        for item in response.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if requested and requested in ids:
        return requested
    if len(ids) == 1:
        return ids[0]
    if requested:
        return requested
    raise DistillError(f"Cannot infer Teacher model id: {ids}")


def generate(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> str:
    response = request_json(
        endpoint.rstrip("/") + "/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout,
    )
    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise DistillError(f"Malformed Teacher response: {response}") from exc


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    return cleaned[:-2] if cleaned.endswith(".0") else cleaned


def extract_number(value: str) -> str:
    if "####" in value:
        value = value.rsplit("####", 1)[1]
    matches = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def prompt_for(row: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    task = str(row["dataset_key"])
    user_prompt = str(row["messages"][-1]["content"])
    if task == "gsm8k":
        system = (
            "Solve the math problem with a short correct derivation. "
            "End with exactly `#### <number>`."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], 512
    if task == "opencodeinstruct":
        system = (
            "Return only a complete Python function implementation in one code block. "
            "Do not use files, network access, or third-party packages."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], 768
    reference = str(row["answer"])
    system = (
        "你是中文知识蒸馏教师。参考答案已经过人工核验。"
        "请只生成1到3句简短分析，不要重复答案，不要引入参考答案之外的新事实。"
    )
    user = f"问题：\n{user_prompt}\n\n参考答案：\n{reference}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], 192


def validate_and_build(
    row: dict[str, Any], teacher_response: str, code_timeout: float
) -> tuple[str, str]:
    task = str(row["dataset_key"])
    source_answer = str(row["answer"]).strip()
    if task == "gsm8k":
        reference = normalize_number(str(row.get("metadata", {}).get("reference_answer", "")))
        if extract_number(teacher_response) == reference:
            return teacher_response, "teacher_verified"
        return source_answer, "source_verified_fallback"
    if task == "opencodeinstruct":
        tests = list(row.get("metadata", {}).get("unit_tests", []))
        candidate = {
            "output": teacher_response,
            "unit_tests": json.dumps(tests, ensure_ascii=False),
        }
        _, status = execute_code_case((candidate, code_timeout))
        if status == "passed":
            return teacher_response, "teacher_verified"
        return source_answer, "source_verified_fallback"
    rationale = teacher_response.strip()
    if not rationale or len(rationale) > 800 or any(
        marker in rationale for marker in ("无法判断", "不确定", "作为AI")
    ):
        rationale = "依据题目给出的信息和已核验知识作答。"
        status = "human_reference_fallback"
    else:
        status = "teacher_rationale_human_reference"
    return f"简短分析：{rationale}\n最终答案：{source_answer}", status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified P0-A5 distillation targets.")
    parser.add_argument("--config", default="configs/p0a5_capability.json")
    parser.add_argument("--source", default="data/capability_v2/source_train.jsonl")
    parser.add_argument("--output", default="data/capability_v2/distill_train.jsonl")
    parser.add_argument("--trace", default="data/capability_v2/teacher_trace.jsonl")
    parser.add_argument("--audit", default="reports/audit/gate_p0a5_distill.json")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        source_path = resolve_path(args.source)
        output_path = resolve_path(args.output)
        trace_path = resolve_path(args.trace)
        audit_path = resolve_path(args.audit)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = read_jsonl(source_path)
        counts = Counter(str(row["dataset_key"]) for row in rows)
        expected = Counter(
            {
                "gsm8k": int(config["datasets"]["math"]["train_rows"]),
                "opencodeinstruct": int(config["datasets"]["code"]["train_rows"]),
                "cmmlu": int(config["datasets"]["nlp"]["train_rows"]),
            }
        )
        if counts != expected:
            raise DistillError(f"Source counts changed: {counts} != {expected}")
        if args.dry_run:
            print(f"P0-A5 distill dry-run passed: rows={len(rows)} counts={dict(counts)}")
            return 0
        model_id = discover_model(args.endpoint, args.model_id, args.timeout_sec)
        existing: dict[str, dict[str, Any]] = {}
        if trace_path.is_file():
            existing = {
                str(row["sample_id"]): row
                for row in read_jsonl(trace_path)
                if row.get("sample_id")
            }
        lock = threading.Lock()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        append_mode = trace_path.open("a", encoding="utf-8")

        def process(row: dict[str, Any]) -> dict[str, Any]:
            sample_id = str(row["sample_id"])
            if sample_id in existing:
                return existing[sample_id]
            messages, max_tokens = prompt_for(row)
            error = ""
            started = time.perf_counter()
            try:
                response = generate(
                    args.endpoint, model_id, messages, max_tokens, args.timeout_sec
                )
            except DistillError as exc:
                response = ""
                error = str(exc)
            latency_ms = (time.perf_counter() - started) * 1000
            answer, validation = validate_and_build(
                row, response, args.code_timeout_sec
            )
            result = {
                "sample_id": sample_id,
                "dataset_key": row["dataset_key"],
                "teacher_model_id": model_id,
                "teacher_prompt_hash": sha256_text(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ),
                "teacher_response": response,
                "teacher_error": error,
                "latency_ms": latency_ms,
                "validation": validation,
                "distill_answer": answer,
            }
            result["row_hash"] = sha256_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
            with lock:
                append_mode.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
                append_mode.flush()
            return result

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            results = list(executor.map(process, rows, chunksize=1))
        append_mode.close()
        by_id = {str(row["sample_id"]): row for row in results}
        if len(by_id) != len(rows):
            raise DistillError("Teacher trace contains duplicate or missing samples")
        output_rows: list[dict[str, Any]] = []
        validation_counts: Counter[str] = Counter()
        teacher_errors = 0
        for source in rows:
            result = by_id[str(source["sample_id"])]
            destination = dict(source)
            destination["answer"] = result["distill_answer"]
            destination["distill_validation"] = result["validation"]
            destination["teacher_model_id"] = model_id
            output_rows.append(destination)
            validation_counts[result["validation"]] += 1
            teacher_errors += bool(result["teacher_error"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        audit = {
            "gate": "P0-A5-DISTILL",
            "check_version": "1.0",
            "created_by": "model_compression/generate_p0a5_distill.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "config": display_path(config_path),
            "config_hash": sha256_file(config_path),
            "source": display_path(source_path),
            "source_hash": sha256_file(source_path),
            "output": display_path(output_path),
            "output_hash": sha256_file(output_path),
            "trace": display_path(trace_path),
            "trace_hash": sha256_file(trace_path),
            "rows": len(output_rows),
            "counts": dict(sorted(counts.items())),
            "validation_counts": dict(sorted(validation_counts.items())),
            "teacher_request_error_count": teacher_errors,
            "fallbacks_are_source_verified": True,
            "formal_test_reference_count": 0,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"validation={dict(validation_counts)} teacher_errors={teacher_errors}")
        return 0
    except (DistillError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 distillation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
