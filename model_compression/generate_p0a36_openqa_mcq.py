#!/usr/bin/env python3
"""Generate balanced, deterministically labelled MCQs from trusted COIG open QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("stem", "humanities", "social_law", "general")
CHOICES = ("A", "B", "C", "D")
TRAIN_PER_GROUP = 256
VALIDATION_PER_GROUP = 64
TOTAL_PER_GROUP = TRAIN_PER_GROUP + VALIDATION_PER_GROUP
MINIMUM_PER_LABEL_PER_GROUP = 64
SEED = 20260802


class GenerateError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GenerateError(f"Missing input: {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latest_trace(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for row in read_jsonl(path):
            latest[str(row["request_id"])] = row
    return latest


def teacher_request(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": 384,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    choices = value.get("choices") or []
    if not choices:
        raise GenerateError("teacher_returned_no_choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise GenerateError("teacher_returned_empty_content")
    return content.strip()


def parse_response(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise GenerateError("teacher_response_not_json")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise GenerateError("teacher_response_not_object")
    return value


def normalized_option(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def render_prompt(question: str, options: list[str]) -> str:
    return "\n".join(
        [f"问题：{question}", *[f"{letter}. {option}" for letter, option in zip(CHOICES, options)]]
    )


def validate(
    request_row: dict[str, Any], response: str, model_id: str
) -> tuple[dict[str, Any] | None, str]:
    try:
        value = parse_response(response)
    except (GenerateError, json.JSONDecodeError) as exc:
        return None, str(exc)
    question = str(value.get("question", "")).strip()
    correct = str(value.get("correct", "")).strip()
    reason = str(value.get("reason_zh", "")).strip()
    distractors = value.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != 3:
        return None, "distractors_not_three"
    distractors = [str(item).strip() for item in distractors]
    if not (5 <= len(question) <= 600 and 1 <= len(correct) <= 160):
        return None, "question_or_correct_length"
    if any(not item or len(item) > 160 for item in distractors):
        return None, "distractor_length"
    if len(re.findall(r"[\u3400-\u9fff]", reason)) < 4 or len(reason) > 400:
        return None, "reason_invalid"
    normalized = [normalized_option(item) for item in [correct, *distractors]]
    if any(not item for item in normalized) or len(set(normalized)) != 4:
        return None, "options_not_distinct"
    if any(value in question for value in ("正确答案：", "答案是", "参考答案")):
        return None, "question_leaks_answer"
    distractors = sorted(
        distractors,
        key=lambda item: sha256_text(f"{SEED}:distractor:{request_row['request_id']}:{item}"),
    )
    correct_index = int(sha256_text(f"{SEED}:label:{request_row['request_id']}")[:8], 16) % 4
    options = list(distractors)
    options.insert(correct_index, correct)
    label = CHOICES[correct_index]
    prompt = render_prompt(question, options)
    return {
        "sample_id": str(request_row["request_id"]),
        "dataset_key": "coig_openqa_mcq_verified",
        "domain": "nlp",
        "task_id": "nlp",
        "source": "COIG-CQIA-open-QA-to-MCQ-Qwen2.5-14B-AWQ",
        "messages": [
            {
                "role": "system",
                "content": (
                    "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
                    "请将A替换为实际选项，只能使用A、B、C或D，禁止输出占位符。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "answer": f"简短分析：{reason.rstrip('。')}。\n最终答案：{label}",
        "answer_letter": label,
        "answer_token_position": "last",
        "answer_token_weight": 2.0,
        "quality_weight": 1.0,
        "training_weight": 2.0,
        "kl_weight": 0.10,
        "validation_group_id": str(request_row["validation_group_id"]),
        "teacher_model_id": model_id,
        "metadata": {
            "knowledge_group": str(request_row["group"]),
            "source_domains": list(request_row.get("domains") or []),
            "source_prompt_hash": str(request_row["source_prompt_hash"]),
            "source_seen_as_open_qa": bool(request_row["source_seen_as_open_qa"]),
            "label_assigned_by": "sha256_deterministic_shuffle",
        },
        "_evaluation_prompt": prompt,
        "_evaluation_reference": label,
    }, ""


def select_rows(
    accepted: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted.values():
        group = str((row.get("metadata") or {}).get("knowledge_group", ""))
        label = str(row.get("answer_letter", ""))
        buckets[(group, label)].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: sha256_text(f"{SEED}:select:{row['sample_id']}"))
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in GROUPS:
        remaining: list[dict[str, Any]] = []
        for label in CHOICES:
            rows = buckets[(group, label)]
            if len(rows) < MINIMUM_PER_LABEL_PER_GROUP:
                return None
            validation.extend(rows[: VALIDATION_PER_GROUP // 4])
            train.extend(rows[VALIDATION_PER_GROUP // 4 : MINIMUM_PER_LABEL_PER_GROUP])
            remaining.extend(rows[MINIMUM_PER_LABEL_PER_GROUP:])
        remaining.sort(
            key=lambda row: sha256_text(f"{SEED}:fill:{group}:{row['sample_id']}")
        )
        fill_count = TRAIN_PER_GROUP - 4 * (
            MINIMUM_PER_LABEL_PER_GROUP - VALIDATION_PER_GROUP // 4
        )
        if len(remaining) < fill_count:
            return None
        train.extend(remaining[:fill_count])
    return train, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="data/distill/p0a36_human_verified_openqa_mcq_requests.jsonl")
    parser.add_argument("--trace", default="data/distill/p0a36_openqa_mcq_trace.jsonl")
    parser.add_argument("--train-output", default="data/p0a36/nlp_train.jsonl")
    parser.add_argument("--validation-output", default="data/p0a36/nlp_validation.jsonl")
    parser.add_argument("--audit", default="reports/audit/gate_p0a36_teacher_data.json")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=180)
    args = parser.parse_args()
    requests_path = resolve(args.requests)
    trace_path = resolve(args.trace)
    train_path = resolve(args.train_output)
    validation_path = resolve(args.validation_output)
    audit_path = resolve(args.audit)
    requests = read_jsonl(requests_path)
    if len(requests) != 2000:
        raise GenerateError(f"Unexpected P0-A36 request count: {len(requests)}")
    previous = latest_trace(trace_path)
    accepted = {
        request_id: row["verified_row"]
        for request_id, row in previous.items()
        if isinstance(row.get("verified_row"), dict)
    }
    pending = [row for row in requests if str(row["request_id"]) not in previous]

    def process(row: dict[str, Any]) -> dict[str, Any]:
        response = ""
        try:
            response = teacher_request(
                args.endpoint, args.model_id, list(row["messages"]), args.timeout_sec
            )
            verified, reason = validate(row, response, args.model_id)
            return {
                "request_id": row["request_id"],
                "status": "accepted" if verified is not None else "rejected",
                "reason": reason,
                "model_id": args.model_id,
                "response": response,
                "verified_row": verified,
            }
        except (HTTPError, URLError, TimeoutError, GenerateError, json.JSONDecodeError) as exc:
            return {
                "request_id": row["request_id"],
                "status": "rejected",
                "reason": f"{type(exc).__name__}:{exc}",
                "model_id": args.model_id,
                "response": response,
                "verified_row": None,
            }

    selected = select_rows(accepted)
    processed = 0
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(0, len(pending), args.workers * 4):
                if selected is not None:
                    break
                batch = pending[offset : offset + args.workers * 4]
                futures = {executor.submit(process, row): row for row in batch}
                for future in as_completed(futures):
                    result = future.result()
                    stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    processed += 1
                    if isinstance(result.get("verified_row"), dict):
                        accepted[str(result["request_id"])] = result["verified_row"]
                    if processed % 100 == 0:
                        print(
                            f"[P0-A36 Teacher] processed={processed}/{len(pending)} "
                            f"accepted_total={len(accepted)}",
                            flush=True,
                        )
                selected = select_rows(accepted)
    if selected is None:
        counts = Counter(
            (
                str((row.get("metadata") or {}).get("knowledge_group", "")),
                str(row.get("answer_letter", "")),
            )
            for row in accepted.values()
        )
        raise GenerateError(f"Cannot satisfy balanced train/validation quotas: {dict(counts)}")
    train, validation = selected
    train_ids = {str(row["sample_id"]) for row in train}
    validation_ids = {str(row["sample_id"]) for row in validation}
    if train_ids & validation_ids or len(train_ids) != 1024 or len(validation_ids) != 256:
        raise GenerateError("P0-A36 split integrity failure")
    train_rows: list[dict[str, Any]] = []
    for row in train:
        copied = dict(row)
        copied.pop("_evaluation_prompt")
        copied.pop("_evaluation_reference")
        copied["split_role"] = "train"
        train_rows.append(copied)
    validation_rows = [
        {
            "sample_id": str(row["sample_id"]),
            "dataset_key": "coig_openqa_mcq_validation",
            "domain": "nlp",
            "subject": str((row.get("metadata") or {}).get("knowledge_group", "")),
            "prompt": str(row["_evaluation_prompt"]),
            "reference": str(row["_evaluation_reference"]),
            "validator": "choice_exact",
            "split_role": "p0a36_external_validation",
        }
        for row in validation
    ]
    train_rows.sort(key=lambda row: str(row["sample_id"]))
    validation_rows.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    latest = latest_trace(trace_path)
    train_groups = Counter(
        str((row.get("metadata") or {}).get("knowledge_group", "")) for row in train_rows
    )
    validation_groups = Counter(str(row["subject"]) for row in validation_rows)
    label_counts = Counter(str(row["answer_letter"]) for row in train_rows)
    validation_labels = Counter(str(row["reference"]) for row in validation_rows)
    report = {
        "gate": "P0-A36-BALANCED-OPENQA-MCQ-TEACHER-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/generate_p0a36_openqa_mcq.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "teacher_model_id": args.model_id,
        "request_count": len(requests),
        "trace_rows": len(latest),
        "accepted_count": len(accepted),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_group_counts": dict(sorted(train_groups.items())),
        "validation_group_counts": dict(sorted(validation_groups.items())),
        "train_label_counts": dict(sorted(label_counts.items())),
        "validation_label_counts": dict(sorted(validation_labels.items())),
        "train_validation_overlap": 0,
        "rejection_counts": dict(Counter(
            str(row.get("reason", "")) for row in latest.values()
            if not isinstance(row.get("verified_row"), dict)
        )),
        "policy": {
            "correct_content_origin": "COIG human_verified answer transformed by 14B",
            "correct_label_origin": "deterministic sha256 option placement",
            "group_and_label_balance": "exact",
            "formal_cmmlu_test_opened": False,
            "p0a34_validation_reused": False,
        },
        "inputs": {requests_path.relative_to(ROOT).as_posix(): sha256_file(requests_path)},
        "trace": trace_path.relative_to(ROOT).as_posix(),
        "trace_hash": sha256_file(trace_path),
        "train_output": train_path.relative_to(ROOT).as_posix(),
        "train_output_hash": sha256_file(train_path),
        "validation_output": validation_path.relative_to(ROOT).as_posix(),
        "validation_output_hash": sha256_file(validation_path),
        "errors": [],
    }
    write_json(audit_path, report)
    print(
        f"P0-A36 train={len(train_rows)} validation={len(validation_rows)} "
        f"accepted={len(accepted)} status=passed"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GenerateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A36 Teacher data failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
