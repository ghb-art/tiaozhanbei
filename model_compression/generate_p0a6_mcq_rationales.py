#!/usr/bin/env python3
"""Generate label-locked C-Eval rationale targets with a 14B Teacher.

Only the labelled C-Eval training partition already present in
``data/p0a6/train.jsonl`` is eligible.  The Teacher is given the human label
and may add a short explanation, but it can never replace that label.  A
response whose final answer letter differs from the human label is retried and
then rejected.

The append-only trace is flushed after every completed sample, so an
interrupted job can be continued with ``--resume``.  The final trace and
training output are rewritten atomically in source order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/p0a6/train.jsonl"
DEFAULT_OUTPUT = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
DEFAULT_TRACE = ROOT / "data/p0a6/nlp_mcq_rationale_trace.jsonl"
DEFAULT_AUDIT = ROOT / "reports/audit/gate_p0a6_mcq_rationales.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
DEFAULT_MODEL_ID = "p0a5-teacher"
EXPECTED_CEVAL_ROWS = 1335
CHOICES = ("A", "B", "C", "D")

FORBIDDEN_SOURCE_PARTS = {"eval", "formal", "sealed", "cmmlu"}
FINAL_ANSWER_RE = re.compile(
    r"最终答案\s*[:：]\s*[*_`]*([A-D])[*_`]*\s*[。.!！]?\s*$", re.I
)
ANY_ANSWER_RE = re.compile(r"(?:最终答案|答案)\s*[:：]\s*([A-D])", re.I)
ANSWER_ONLY_RE = re.compile(r"^\s*最终答案\s*[:：]\s*([A-D])\s*$", re.I)
BAD_RATIONALE_MARKERS = (
    "无法判断",
    "无法确定",
    "我不确定答案",
    "作为ai",
    "作为 AI",
    "参考答案给出",
    "人工标注",
)

TeacherCallable = Callable[[list[dict[str, str]], int, float], str]


class RationaleGenerationError(RuntimeError):
    """Raised when the source protocol or generation gate is violated."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def guard_source_path(path: Path) -> None:
    resolved_parts = {part.casefold() for part in path.resolve().parts}
    forbidden = sorted(resolved_parts & FORBIDDEN_SOURCE_PARTS)
    if forbidden:
        raise RationaleGenerationError(
            "Forbidden evaluation/formal/sealed/CMMLU source path: "
            f"{display_path(path)} ({', '.join(forbidden)})"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RationaleGenerationError(f"Missing source: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RationaleGenerationError(
                    f"Invalid JSON at {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RationaleGenerationError(
                    f"Non-object row at {display_path(path)}:{line_number}"
                )
            rows.append(row)
    return rows


def _last_assistant_free_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RationaleGenerationError(
            f"Missing messages for {row.get('sample_id', '<missing>')}"
        )
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RationaleGenerationError("Malformed source message")
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user"} or not content:
            raise RationaleGenerationError(
                "C-Eval rationale source may contain only non-empty system/user messages"
            )
        normalized.append({"role": role, "content": content})
    if not any(item["role"] == "user" for item in normalized):
        raise RationaleGenerationError("C-Eval source has no user question")
    return normalized


def extract_human_label(row: dict[str, Any]) -> str:
    """Return the immutable human label after two-source agreement checks."""

    source_answer = str(row.get("answer", "")).strip()
    answer_match = ANSWER_ONLY_RE.fullmatch(source_answer)
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise RationaleGenerationError("C-Eval row has no metadata object")
    reference = str(metadata.get("reference_answer", "")).strip().upper()
    if answer_match is None or reference not in CHOICES:
        raise RationaleGenerationError(
            f"Missing normalized human label for {row.get('sample_id', '<missing>')}"
        )
    answer_letter = answer_match.group(1).upper()
    if answer_letter != reference:
        raise RationaleGenerationError(
            f"Human-label disagreement for {row.get('sample_id', '<missing>')}: "
            f"answer={answer_letter}, metadata={reference}"
        )
    if metadata.get("human_labelled") is not True:
        raise RationaleGenerationError(
            f"C-Eval row is not marked human_labelled: {row.get('sample_id')}"
        )
    options = metadata.get("options")
    if not isinstance(options, dict) or any(
        not str(options.get(letter, "")).strip() for letter in CHOICES
    ):
        raise RationaleGenerationError(
            f"C-Eval row lacks A-D options: {row.get('sample_id')}"
        )
    return reference


def select_ceval_training_rows(
    source_path: Path, expected_rows: int = EXPECTED_CEVAL_ROWS
) -> list[dict[str, Any]]:
    guard_source_path(source_path)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(source_path):
        dataset_key = str(row.get("dataset_key", "")).strip().casefold()
        if dataset_key == "cmmlu":
            raise RationaleGenerationError(
                "CMMLU data is forbidden in the rationale-generation source"
            )
        if dataset_key != "ceval" or str(row.get("split_role", "")) != "train":
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise RationaleGenerationError(
                f"Missing or duplicate C-Eval sample_id: {sample_id!r}"
            )
        identity = " ".join(
            (sample_id, str(row.get("source", "")), str(row.get("dataset_key", "")))
        ).casefold()
        if any(marker in identity for marker in ("cmmlu", "formal", "sealed")):
            raise RationaleGenerationError(
                f"Forbidden source identity in {sample_id}"
            )
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("ceval_split") != "val":
            raise RationaleGenerationError(
                f"Only labelled C-Eval val repurposed as train is allowed: {sample_id}"
            )
        _last_assistant_free_messages(row)
        extract_human_label(row)
        seen.add(sample_id)
        selected.append(row)
    if len(selected) != expected_rows:
        raise RationaleGenerationError(
            f"C-Eval train-row count changed: {len(selected)} != {expected_rows}"
        )
    return selected


def build_teacher_messages(row: dict[str, Any], label: str, attempt: int) -> list[dict[str, str]]:
    source_messages = _last_assistant_free_messages(row)
    question = next(
        item["content"] for item in reversed(source_messages) if item["role"] == "user"
    )
    option_text = str(row["metadata"]["options"][label]).strip()
    system = (
        "你是中文学科选择题知识蒸馏教师。正确选项已经由人工标注并锁定，"
        "你只能解释该选项为何正确，绝对不得修改、质疑或猜测答案。"
        "请给出1到3句、8到400字的简短中文理由，理由应包含有助于学生学习的知识依据，"
        "不能只说‘该选项正确’。最后一行必须严格写成“最终答案：X”，X替换为锁定字母，"
        "字母后不得有标点或其他内容。"
    )
    user = (
        f"题目如下：\n{question}\n\n"
        f"人工锁定正确选项：{label}. {option_text}\n"
        f"请解释并在末行原样输出：最终答案：{label}"
    )
    if attempt == 2:
        user += (
            "\n这是长度与格式修复重试。务必保留人工锁定答案；"
            "禁止展开完整公式推导或分点，理由不超过120个汉字，然后输出指定末行。"
        )
    elif attempt > 2:
        user += (
            "\n这是最后一次格式修复。只写一条核心定义、定理或关键计算和结论，"
            "理由不超过80个汉字；最后单独一行输出人工锁定答案。"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_and_validate_response(response: str, expected_label: str) -> tuple[str, str]:
    """Validate a Teacher response and return normalized rationale/answer."""

    text = str(response).strip()
    if not text:
        raise RationaleGenerationError("empty_teacher_response")
    final_match = FINAL_ANSWER_RE.search(text)
    if final_match is None:
        raise RationaleGenerationError("missing_strict_final_answer")
    final_label = final_match.group(1).upper()
    if final_label != expected_label:
        raise RationaleGenerationError(
            f"teacher_label_mismatch:{final_label}!={expected_label}"
        )
    answer_markers = [match.group(1).upper() for match in ANY_ANSWER_RE.finditer(text)]
    if any(letter != expected_label for letter in answer_markers):
        raise RationaleGenerationError("conflicting_answer_marker")
    rationale = text[: final_match.start()].strip()
    # Markdown wrappers around the terminal answer belong to formatting, not
    # to the teaching rationale (for example: ``**最终答案：A**``).
    rationale = re.sub(r"[\s*_`#：:,，;；-]+$", "", rationale).strip()
    rationale = re.sub(r"^(?:简短分析|简短理由|理由|分析)\s*[:：]\s*", "", rationale)
    # The prompt asks for at most 400 Chinese characters.  Allow a small
    # tokenizer/formatting tolerance so a complete, label-locked explanation
    # is not discarded for being only a few characters over that target.
    if not 8 <= len(rationale) <= 512:
        raise RationaleGenerationError(
            f"rationale_length_out_of_range:{len(rationale)}"
        )
    rationale_casefold = rationale.casefold()
    if any(marker.casefold() in rationale_casefold for marker in BAD_RATIONALE_MARKERS):
        raise RationaleGenerationError("low_information_or_refusal_rationale")
    answer = f"简短理由：{rationale}\n最终答案：{expected_label}"
    if not answer.endswith(expected_label):
        raise AssertionError("Normalized answer no longer ends with locked label")
    return rationale, answer


def rationale_sample_id(source_sample_id: str) -> str:
    suffix = source_sample_id
    if suffix.startswith("ceval/"):
        suffix = suffix[len("ceval/") :]
    return f"ceval_rationale_train/{suffix}"


def build_training_row(
    source: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    label = extract_human_label(source)
    if str(trace.get("manual_label", "")).upper() != label:
        raise RationaleGenerationError("Trace attempted to replace the human label")
    _, normalized_answer = parse_and_validate_response(
        str(trace.get("teacher_response", "")), label
    )
    destination = copy.deepcopy(source)
    destination.update(
        {
            "sample_id": rationale_sample_id(str(source["sample_id"])),
            "dataset_key": "ceval_rationale_train",
            "domain": "nlp",
            "task_id": "nlp",
            "source": "C-Eval-labelled+14B-rationale",
            "split_role": "train",
            "answer": normalized_answer,
            "answer_token_weight": 2.0,
            "training_weight": 1.0,
            "quality_weight": 1.0,
            "kl_weight": 0.2,
            "distill_validation": "teacher_rationale_human_label_locked",
            "teacher_model_id": str(trace["teacher_model_id"]),
        }
    )
    metadata = dict(destination.get("metadata", {}))
    metadata.update(
        {
            "source_sample_id": str(source["sample_id"]),
            "source_dataset_key": "ceval",
            "reference_answer": label,
            "human_labelled": True,
            "teacher_added_rationale_only": True,
            "teacher_response_hash": sha256_text(
                str(trace.get("teacher_response", ""))
            ),
        }
    )
    destination["metadata"] = metadata
    return destination


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    timeout_sec: float,
    api_key: str = "",
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RationaleGenerationError(f"HTTP {exc.code}: {body[:400]}") from exc
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise RationaleGenerationError(f"Teacher request failed: {exc}") from exc


def discover_model(
    endpoint: str, requested_model_id: str, timeout_sec: float, api_key: str
) -> str:
    response = request_json(
        endpoint.rstrip("/") + "/v1/models", None, timeout_sec, api_key
    )
    model_ids = [
        str(item["id"])
        for item in response.get("data", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    if requested_model_id:
        if requested_model_id not in model_ids:
            raise RationaleGenerationError(
                f"Requested 14B Teacher id is not served: {requested_model_id}; "
                f"available={model_ids}"
            )
        return requested_model_id
    if len(model_ids) != 1:
        raise RationaleGenerationError(
            f"Cannot infer one Teacher model id: {model_ids}"
        )
    return model_ids[0]


def openai_teacher_callable(
    endpoint: str,
    model_id: str,
    temperature: float,
    api_key: str,
) -> TeacherCallable:
    def generate(
        messages: list[dict[str, str]], max_tokens: int, timeout_sec: float
    ) -> str:
        response = request_json(
            endpoint.rstrip("/") + "/v1/chat/completions",
            {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout_sec,
            api_key,
        )
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RationaleGenerationError(
                f"Malformed Teacher response: {str(response)[:400]}"
            ) from exc

    return generate


def load_resume_trace(
    path: Path,
    sources_by_id: dict[str, dict[str, Any]],
    requested_model_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return all terminal rows and only reusable accepted trace rows."""

    if not path.is_file():
        return {}, {}
    terminal: dict[str, dict[str, Any]] = {}
    reusable: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        source = sources_by_id.get(sample_id)
        if source is None:
            continue
        terminal[sample_id] = row
        if row.get("status") != "accepted":
            continue
        if requested_model_id and row.get("teacher_model_id") != requested_model_id:
            continue
        label = extract_human_label(source)
        prompt_hash = canonical_hash(build_teacher_messages(source, label, 1))
        if row.get("source_row_hash") != canonical_hash(source):
            continue
        if row.get("teacher_prompt_hash") != prompt_hash:
            continue
        if str(row.get("manual_label", "")).upper() != label:
            continue
        try:
            parse_and_validate_response(str(row.get("teacher_response", "")), label)
        except RationaleGenerationError:
            continue
        reusable[sample_id] = row
    return terminal, reusable


def _trace_hash(row: dict[str, Any]) -> str:
    return canonical_hash({key: value for key, value in row.items() if key != "row_hash"})


def generate_one(
    source: dict[str, Any],
    teacher_model_id: str,
    teacher: TeacherCallable,
    max_attempts: int,
    max_tokens: int,
    timeout_sec: float,
) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    label = extract_human_label(source)
    errors: list[str] = []
    final_response = ""
    total_latency_ms = 0.0
    prompt_hash = canonical_hash(build_teacher_messages(source, label, 1))
    for attempt in range(1, max_attempts + 1):
        messages = build_teacher_messages(source, label, attempt)
        started = time.perf_counter()
        try:
            final_response = teacher(messages, max_tokens, timeout_sec)
            total_latency_ms += (time.perf_counter() - started) * 1000
            rationale, normalized_answer = parse_and_validate_response(
                final_response, label
            )
        except Exception as exc:
            total_latency_ms += (time.perf_counter() - started) * 1000
            errors.append(f"attempt_{attempt}:{type(exc).__name__}:{exc}")
            continue
        result = {
            "sample_id": sample_id,
            "dataset_key": "ceval",
            "status": "accepted",
            "manual_label": label,
            "teacher_model_id": teacher_model_id,
            "teacher_prompt_hash": prompt_hash,
            "source_row_hash": canonical_hash(source),
            "teacher_response": final_response,
            "rationale": rationale,
            "distill_answer": normalized_answer,
            "attempt_count": attempt,
            "attempt_errors": errors,
            "latency_ms": total_latency_ms,
        }
        result["row_hash"] = _trace_hash(result)
        return result
    result = {
        "sample_id": sample_id,
        "dataset_key": "ceval",
        "status": "rejected",
        "manual_label": label,
        "teacher_model_id": teacher_model_id,
        "teacher_prompt_hash": prompt_hash,
        "source_row_hash": canonical_hash(source),
        "teacher_response": final_response,
        "rationale": "",
        "distill_answer": "",
        "attempt_count": max_attempts,
        "attempt_errors": errors,
        "latency_ms": total_latency_ms,
    }
    result["row_hash"] = _trace_hash(result)
    return result


def run_generation(
    args: argparse.Namespace,
    *,
    teacher: TeacherCallable | None = None,
    served_model_id: str | None = None,
) -> dict[str, Any]:
    source_path = resolve_path(args.source)
    output_path = resolve_path(args.output)
    trace_path = resolve_path(args.trace)
    audit_path = resolve_path(args.audit)
    rows = select_ceval_training_rows(source_path, args.expected_rows)
    sources_by_id = {str(row["sample_id"]): row for row in rows}
    source_ids = [str(row["sample_id"]) for row in rows]

    if args.dry_run:
        audit = {
            "gate": "P0-A6-NLP-MCQ-RATIONALE",
            "check_version": "1.0",
            "created_by": "model_compression/generate_p0a6_mcq_rationales.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "dry_run": True,
            "source": display_path(source_path),
            "source_hash": sha256_file(source_path),
            "selected_rows": len(rows),
            "human_label_lock_validated": True,
            "forbidden_formal_or_cmmlu_sources_read": False,
        }
        audit["report_hash"] = _trace_hash(audit)
        atomic_write_json(audit_path, audit)
        print(
            f"P0-A6 MCQ rationale dry-run passed: selected={len(rows)} "
            f"audit={display_path(audit_path)}"
        )
        return audit

    terminal: dict[str, dict[str, Any]] = {}
    reusable: dict[str, dict[str, Any]] = {}
    if args.resume:
        terminal, reusable = load_resume_trace(
            trace_path, sources_by_id, args.model_id
        )
    pending = [row for row in rows if str(row["sample_id"]) not in reusable]

    if served_model_id is None:
        if pending:
            served_model_id = discover_model(
                args.endpoint, args.model_id, args.timeout_sec, args.api_key
            )
        else:
            model_ids = {
                str(row.get("teacher_model_id", "")) for row in reusable.values()
            }
            if len(model_ids) != 1:
                raise RationaleGenerationError(
                    f"Completed trace has ambiguous Teacher ids: {sorted(model_ids)}"
                )
            served_model_id = next(iter(model_ids))
    if args.model_id and served_model_id != args.model_id:
        raise RationaleGenerationError(
            f"Teacher identity mismatch: served={served_model_id}, requested={args.model_id}"
        )
    if teacher is None:
        teacher = openai_teacher_callable(
            args.endpoint, served_model_id, args.temperature, args.api_key
        )

    results = dict(reusable)
    terminal.update(reusable)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_mode = "a" if args.resume else "w"
    append_lock = threading.Lock()
    completed = 0
    with trace_path.open(trace_mode, encoding="utf-8") as trace_handle:
        def record(result: dict[str, Any]) -> None:
            nonlocal completed
            sample_id = str(result["sample_id"])
            terminal[sample_id] = result
            if result["status"] == "accepted":
                results[sample_id] = result
            else:
                results.pop(sample_id, None)
            with append_lock:
                trace_handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
                trace_handle.flush()
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {sample_id} "
                    f"status={result['status']} attempts={result['attempt_count']}",
                    flush=True,
                )

        if args.workers == 1:
            for source in pending:
                record(
                    generate_one(
                        source,
                        served_model_id,
                        teacher,
                        args.max_attempts,
                        args.max_tokens,
                        args.timeout_sec,
                    )
                )
        elif pending:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        generate_one,
                        source,
                        served_model_id,
                        teacher,
                        args.max_attempts,
                        args.max_tokens,
                        args.timeout_sec,
                    ): str(source["sample_id"])
                    for source in pending
                }
                for future in as_completed(futures):
                    record(future.result())

    canonical_trace = [terminal[sample_id] for sample_id in source_ids]
    output_rows = [
        build_training_row(source, results[str(source["sample_id"])])
        for source in rows
        if str(source["sample_id"]) in results
    ]
    atomic_write_jsonl(trace_path, canonical_trace)
    atomic_write_jsonl(output_path, output_rows)

    status_counts = Counter(str(row["status"]) for row in canonical_trace)
    retry_count = sum(max(0, int(row["attempt_count"]) - 1) for row in canonical_trace)
    status = "passed" if len(output_rows) == len(rows) else "failed"
    audit = {
        "gate": "P0-A6-NLP-MCQ-RATIONALE",
        "check_version": "1.0",
        "created_by": "model_compression/generate_p0a6_mcq_rationales.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dry_run": False,
        "source": display_path(source_path),
        "source_hash": sha256_file(source_path),
        "source_dataset_key": "ceval",
        "source_split_role": "train",
        "source_partition": "C-Eval labelled val repurposed as auxiliary train",
        "selected_rows": len(rows),
        "teacher_endpoint": args.endpoint,
        "teacher_model_id": served_model_id,
        "workers": args.workers,
        "max_attempts": args.max_attempts,
        "resumed_accepted_rows": len(reusable),
        "newly_processed_rows": len(pending),
        "status_counts": dict(sorted(status_counts.items())),
        "retry_count": retry_count,
        "human_label_lock_validated": True,
        "teacher_may_replace_human_label": False,
        "formal_test_loaded": False,
        "cmmlu_loaded": False,
        "forbidden_formal_or_cmmlu_sources_read": False,
        "output": display_path(output_path),
        "output_rows": len(output_rows),
        "output_hash": sha256_file(output_path),
        "trace": display_path(trace_path),
        "trace_rows": len(canonical_trace),
        "trace_hash": sha256_file(trace_path),
        "output_schema": {
            "dataset_key": "ceval_rationale_train",
            "domain": "nlp",
            "answer_token_weight": 2.0,
            "training_weight": 1.0,
            "kl_weight": 0.2,
        },
    }
    audit["report_hash"] = _trace_hash(audit)
    atomic_write_json(audit_path, audit)
    print(f"Wrote {display_path(trace_path)}")
    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(
        f"status={status} accepted={len(output_rows)}/{len(rows)} "
        f"retries={retry_count}"
    )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-label-locked P0-A6 C-Eval rationale training rows."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_CEVAL_ROWS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.expected_rows <= 0:
        parser.error("--expected-rows must be positive")
    if not 0.0 <= args.temperature <= 1.0:
        parser.error("--temperature must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = run_generation(args)
    except RationaleGenerationError as exc:
        print(f"P0-A6 MCQ rationale generation failed: {exc}", file=sys.stderr)
        return 1
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
