from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .generate_teacher_capability_distill import validate_code_row
except ImportError:
    from generate_teacher_capability_distill import validate_code_row


class EdgeCandidateEvalError(RuntimeError):
    pass


def load_generation_validation_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EdgeCandidateEvalError(f"Missing generation validation JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EdgeCandidateEvalError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EdgeCandidateEvalError(f"Expected an object at {path}:{line_number}")
            if row.get("used_for_training") is not False:
                raise EdgeCandidateEvalError(
                    "External generation validation rows must declare "
                    f"used_for_training=false: {row.get('sample_id', '<missing>')}"
                )
            messages = row.get("messages")
            answer = str(row.get("answer", ""))
            if not isinstance(messages, list) or not messages or not answer:
                raise EdgeCandidateEvalError(
                    f"Incomplete validation row: {row.get('sample_id', '<missing>')}"
                )
            copied = {
                "source": str(row.get("source", "external_generation_validation")),
                "dataset_key": str(row.get("dataset_key", "")),
                "sample_id": str(row.get("sample_id", "")),
                "validation_group_id": str(row.get("validation_group_id", "")),
                "messages": messages,
                "answer": answer,
            }
            if isinstance(row.get("code_eval"), dict):
                copied["code_eval"] = row["code_eval"]
            rows.append(copied)
    if not rows:
        raise EdgeCandidateEvalError(f"No validation rows: {path}")
    return rows


def render_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    close_reasoning_prefix: bool,
    disable_thinking: bool = False,
) -> str:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    if close_reasoning_prefix and prompt.rstrip().endswith("<think>"):
        prompt += "</think>\n"
    return prompt


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    return cleaned[:-2] if cleaned.endswith(".0") else cleaned


def extract_gsm8k_reference(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.rsplit("####", 1)[1])
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_gsm8k_prediction(text: str) -> str:
    if "####" in text:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text.rsplit("####", 1)[1].replace(",", ""))
        if matches:
            return normalize_number(matches[0])
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_choice(text: str) -> str:
    stripped = text.strip().upper()
    match = re.search(r"(?:答案|ANSWER|选项|OPTION)?\s*[:：]?\s*([ABCD])\b", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", stripped)
    return match.group(1) if match else ""


def score_generation_validation(
    example: dict[str, Any],
    response: str,
    code_timeout_sec: float,
) -> float:
    dataset_key = str(example.get("dataset_key", ""))
    answer = str(example.get("answer", ""))
    if dataset_key == "gsm8k":
        expected = extract_gsm8k_reference(answer)
        return float(bool(expected) and extract_gsm8k_prediction(response) == expected)
    if dataset_key == "cmmlu":
        expected = extract_choice(answer) or answer.strip().upper()[:1]
        return float(bool(expected) and extract_choice(response) == expected)
    if dataset_key == "humaneval":
        accepted, _, _ = validate_code_row(example, response, code_timeout_sec)
        return float(accepted)
    return 0.0
