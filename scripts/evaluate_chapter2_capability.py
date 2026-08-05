from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPRESSION = ROOT / "model_compression"
if str(MODEL_COMPRESSION) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPRESSION))

from inference_utils import (
    infer_model_id,
    load_local_student,
    parse_comma_values,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL_DIR = ROOT / "models" / "checkpoints" / "p0a4" / "student-shared-merged"
DEFAULT_OUTPUT_TRACE = ROOT / "reports" / "audit" / "chapter2_capability_eval_smoke.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_chapter2_capability_eval_smoke.json"
SPLITS = ROOT / "data" / "splits"
SYSTEM_PROMPT = "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested."
HUMANEVAL_EXEC_PREAMBLE = """\
from typing import *
import math
import re
import itertools
import functools
import collections
from collections import *
import heapq
import bisect
import string
import operator
import copy
import decimal
import fractions
import statistics
from functools import *
"""


class CapabilityEvalError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_split_ids(dataset_key: str, split: str = "test", split_dir: Path = SPLITS) -> list[str]:
    path = split_dir / f"{dataset_key}_{split}.txt"
    if not path.is_file():
        raise CapabilityEvalError(f"Missing split file: {display_path(path)}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_shard(samples: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise CapabilityEvalError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise CapabilityEvalError("--shard-index must be in [0, num-shards)")
    if num_shards == 1:
        return samples
    return [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def extract_gsm8k_reference(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.rsplit("####", 1)[1])
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_gsm8k_prediction(text: str) -> str:
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        matches = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
        if matches:
            return normalize_number(matches[0])
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_choice(text: str) -> str:
    stripped = strip_reasoning_envelope(text).strip().upper()
    match = re.search(r"(?:答案|ANSWER|选项|OPTION)?\s*[:：]?\s*([ABCD])\b", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", stripped)
    return match.group(1) if match else ""


def clean_code_completion(text: str) -> str:
    stripped = strip_reasoning_envelope(text).strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    stripped = re.sub(r"^python\s+", "", stripped, flags=re.IGNORECASE).strip()
    return stripped


def strip_reasoning_envelope(text: str) -> str:
    """Remove llama.cpp's residual Qwen think wrapper when thinking is disabled.

    Some llama.cpp versions keep an empty ``<think>...</think>`` envelope in
    ``message.content`` even with ``--reasoning off``.  It is transport markup,
    not part of the requested Python body or multiple-choice answer.
    """

    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_function_block(source: str, entry_point: str) -> str:
    lines = source.splitlines()
    start_index = None
    def_pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    for index, line in enumerate(lines):
        if def_pattern.match(line):
            start_index = index
            break
    if start_index is None:
        return ""

    block = [lines[start_index]]
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")):
            if re.match(r"^(def|class)\s+", stripped) or stripped.startswith(("if __name__", "```")):
                break
            if not stripped.startswith(("#", "@")):
                break
        block.append(line)
    return "\n".join(block).rstrip() + "\n"


def count_leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def shift_later_indents_left(body: str) -> str:
    lines = body.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return body
    later_indents = [
        count_leading_spaces(line)
        for line in lines[first_index + 1 :]
        if line.strip() and count_leading_spaces(line) > 0
    ]
    if not later_indents:
        return body
    shift = min(later_indents)
    shifted = []
    for index, line in enumerate(lines):
        if index > first_index and line.startswith(" " * shift):
            shifted.append(line[shift:])
        else:
            shifted.append(line)
    return "\n".join(shifted)


def left_strip_body(body: str) -> str:
    return "\n".join(line.lstrip() if line.strip() else "" for line in body.splitlines())


def humaneval_body_variants(completion: str) -> list[str]:
    base = textwrap.dedent(completion).strip("\n")
    variants: list[str] = []
    stripped_lines = [
        line
        for line in base.splitlines()
        if not line.strip().lower().startswith(("here is", "sure,", "the code", "答案", "解释"))
    ]
    stripped_prose = "\n".join(stripped_lines).strip("\n")
    for candidate in [base, stripped_prose, shift_later_indents_left(base), left_strip_body(base)]:
        if candidate not in variants:
            variants.append(candidate)
    return variants


def indent_humaneval_body(body: str) -> str:
    if not body.strip():
        return "    pass\n"
    return "\n".join(("    " + line) if line.strip() else "" for line in body.splitlines()) + "\n"


def humaneval_candidate_sources(prompt_source: str, entry_point: str, completion: str) -> list[str]:
    """Build executable candidates using the exact formal HumanEval response protocol."""
    completion = clean_code_completion(completion)
    candidates: list[str] = []
    if f"def {entry_point}" in completion:
        raw_candidates = [
            completion.rstrip() + "\n",
            extract_function_block(completion, entry_point),
            textwrap.dedent(completion).strip() + "\n",
        ]
    else:
        raw_candidates = [
            prompt_source + indent_humaneval_body(body)
            for body in humaneval_body_variants(completion)
        ]
    for candidate in raw_candidates:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def run_python_check(source: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", source],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if completed.returncode == 0:
        return True, "passed"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1][:240] if detail else f"returncode={completed.returncode}"


def run_assert_source_check(
    candidate_source: str,
    tests: list[str],
    timeout_sec: float,
    setup_code: str = "",
) -> tuple[bool, str]:
    test_source = "\n".join(str(test).strip() for test in tests if str(test).strip())
    check_source = (
        f"{HUMANEVAL_EXEC_PREAMBLE}\n"
        f"{setup_code.rstrip()}\n"
        f"{candidate_source.rstrip()}\n"
        f"{test_source}\n"
    )
    return run_python_check(check_source, timeout_sec)


def run_assert_tests_check(
    prompt_source: str,
    entry_point: str,
    tests: list[str],
    completion: str,
    timeout_sec: float,
    setup_code: str = "",
) -> tuple[bool, str]:
    first_detail = ""
    for candidate_source in humaneval_candidate_sources(prompt_source, entry_point, completion):
        passed, detail = run_assert_source_check(candidate_source, tests, timeout_sec, setup_code)
        if passed:
            return True, detail
        if not first_detail:
            first_detail = detail
    return False, first_detail or "failed"


def run_humaneval_check(problem: dict[str, Any], completion: str, timeout_sec: float) -> tuple[bool, str]:
    entry_point = str(problem["entry_point"])
    candidate_sources = humaneval_candidate_sources(str(problem["prompt"]), entry_point, completion)
    first_detail = ""
    for candidate_source in candidate_sources:
        check_source = f"{HUMANEVAL_EXEC_PREAMBLE}\n{candidate_source}\n{problem['test']}\ncheck({entry_point})\n"
        passed, detail = run_python_check(check_source, timeout_sec)
        if passed:
            return True, detail
        if not first_detail:
            first_detail = detail
    return False, first_detail or "failed"


def generate_text(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int,
    disable_thinking: bool = False,
) -> tuple[str, float]:
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    chat_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True).to(device)
    input_length = int(inputs["input_ids"].shape[1])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    response_ids = output_ids[0, input_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip(), latency_ms


def request_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8", errors="replace")
            return json.loads(text)
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise CapabilityEvalError(f"HTTP {exc.code}: {body_text[:500]}") from exc
    except (URLError, OSError) as exc:
        raise CapabilityEvalError(f"Endpoint request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CapabilityEvalError(f"Endpoint response is not JSON: {exc}") from exc


def health_status(base_url: str, timeout_sec: float) -> int | None:
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/health", method="GET"), timeout=timeout_sec) as response:
            return int(response.status)
    except Exception:
        return None


def get_served_model(base_url: str, timeout_sec: float, fallback: str) -> str:
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/v1/models", method="GET"), timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    except Exception:
        pass
    return fallback


def generate_text_endpoint(
    base_url: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout_sec: float,
    max_new_tokens: int,
    disable_thinking: bool = False,
    request_extra: dict[str, Any] | None = None,
) -> tuple[str, float]:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if request_extra:
        forbidden = set(request_extra) & {"model", "messages", "temperature", "max_tokens", "stream"}
        if disable_thinking and "chat_template_kwargs" in request_extra:
            forbidden.add("chat_template_kwargs")
        if forbidden:
            raise CapabilityEvalError(f"Request-extra cannot override evaluator fields: {sorted(forbidden)}")
        payload.update(request_extra)
    started = time.perf_counter()
    response = request_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout_sec)
    latency_ms = (time.perf_counter() - started) * 1000
    choices = response.get("choices", [])
    if not choices:
        raise CapabilityEvalError("Endpoint response has no choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(message.get("content", "")).strip(), latency_ms


@lru_cache(maxsize=None)
def load_gsm8k_rows(split: str) -> tuple[dict[str, Any], ...]:
    path = ROOT / "data" / "datasets" / "gsm8k" / "grade_school_math" / "data" / f"{split}.jsonl"
    return tuple(read_jsonl(path))


def load_gsm8k_sample(sample_id: str) -> dict[str, Any]:
    _, split, index_text = sample_id.split("/")
    index = int(index_text)
    row = load_gsm8k_rows(split)[index]
    return {
        "dataset_key": "gsm8k",
        "sample_id": sample_id,
        "question": row["question"],
        "answer": row["answer"],
        "reference": extract_gsm8k_reference(row["answer"]),
    }


@lru_cache(maxsize=None)
def load_cmmlu_rows(split: str, subject: str) -> tuple[dict[str, str], ...]:
    path = ROOT / "data" / "datasets" / "cmmlu" / "data" / split / f"{subject}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def load_cmmlu_sample(sample_id: str) -> dict[str, Any]:
    _, split, subject, index_text = sample_id.split("/")
    index = int(index_text)
    row = load_cmmlu_rows(split, subject)[index]
    return {
        "dataset_key": "cmmlu",
        "sample_id": sample_id,
        "subject": subject,
        "question": row["Question"],
        "choices": {key: row[key] for key in ("A", "B", "C", "D")},
        "reference": row["Answer"].strip().upper(),
    }


@lru_cache(maxsize=1)
def load_humaneval_rows() -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
    samples: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["dataset_key"] = "humaneval"
            row["sample_id"] = row["task_id"]
            samples[row["task_id"]] = row
    return samples


def load_humaneval_sample(sample_id: str) -> dict[str, Any]:
    return dict(load_humaneval_rows()[sample_id])


SAMPLE_LOADERS = {
    "gsm8k": load_gsm8k_sample,
    "cmmlu": load_cmmlu_sample,
    "humaneval": load_humaneval_sample,
}


def load_samples(
    dataset_key: str,
    limit: int,
    use_frozen_final: bool,
    split_dir: Path = SPLITS,
) -> list[dict[str, Any]]:
    if dataset_key not in SAMPLE_LOADERS:
        raise CapabilityEvalError(f"Unsupported dataset: {dataset_key}")
    if use_frozen_final:
        sample_ids = read_split_ids(dataset_key, "test", split_dir)
    elif dataset_key == "humaneval":
        sample_ids = list(load_humaneval_rows())[:limit]
    else:
        sample_ids = read_split_ids(dataset_key, "test", split_dir)[:limit]
    if limit > 0:
        sample_ids = sample_ids[:limit]
    return [SAMPLE_LOADERS[dataset_key](sample_id) for sample_id in sample_ids]


def build_messages(sample: dict[str, Any], prompt_style: str = "default") -> tuple[list[dict[str, str]], str]:
    dataset_key = sample["dataset_key"]
    if dataset_key == "gsm8k":
        if prompt_style == "v15":
            prompt = (
                "Solve the math word problem carefully. Translate every stated relationship before calculating, "
                "check the arithmetic and units once, and keep the reasoning concise. End with exactly one final "
                "line in the form #### <number>. Do not write any numbers after that line.\n\n"
                f"Problem: {sample['question']}"
            )
        elif prompt_style == "v11":
            prompt = (
                "Solve the math problem with concise arithmetic. End with exactly one final line in this format: "
                "#### <number>. Do not write any numbers after that final line.\n\n"
                f"Problem: {sample['question']}"
            )
        else:
            prompt = (
                "Solve the math problem. Show concise reasoning and put the final answer in the form "
                "#### <number>.\n\n"
                f"Problem: {sample['question']}"
            )
    elif dataset_key == "cmmlu":
        choices = sample["choices"]
        if prompt_style in {"v11", "v15"}:
            prompt = (
                "以下是单项选择题。请先判断正确选项，但最终只输出一个大写字母 A、B、C 或 D，不要解释。\n\n"
                f"题目：{sample['question']}\n"
                f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}\n"
                "最终答案："
            )
        else:
            prompt = (
                "以下是单项选择题。只输出一个大写字母 A、B、C 或 D。\n\n"
                f"题目：{sample['question']}\n"
                f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}"
            )
    elif dataset_key == "humaneval":
        if prompt_style == "v15":
            prompt = (
                "Complete the Python function correctly. Return only the function body: do not repeat the def "
                "header or docstring, and do not use markdown or explanations. Start the first body statement at "
                "column 1; keep only the relative indentation required inside loops, conditions, and nested blocks. "
                "Handle the edge cases stated in the docstring.\n\n"
                f"{sample['prompt']}"
            )
        elif prompt_style == "v11":
            prompt = (
                "Complete the Python function. Return only valid Python code, no markdown and no explanation. "
                "If the prompt already contains the function header, return only the indented function body.\n\n"
                f"{sample['prompt']}"
            )
        else:
            prompt = (
                "Complete the Python function. Return only the code needed to complete or define the function.\n\n"
                f"{sample['prompt']}"
            )
    else:
        raise CapabilityEvalError(f"Unsupported dataset: {dataset_key}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return messages, sha256_text(prompt_style + "\n" + SYSTEM_PROMPT + "\n" + prompt)


def score_sample(sample: dict[str, Any], response_text: str, humaneval_timeout_sec: float) -> tuple[bool, str, str]:
    dataset_key = sample["dataset_key"]
    if dataset_key == "gsm8k":
        prediction = extract_gsm8k_prediction(response_text)
        return prediction == sample["reference"], prediction, "exact_numeric_match"
    if dataset_key == "cmmlu":
        prediction = extract_choice(response_text)
        return prediction == sample["reference"], prediction, "choice_match"
    if dataset_key == "humaneval":
        passed, detail = run_humaneval_check(sample, response_text, humaneval_timeout_sec)
        return passed, "pass" if passed else "fail", detail
    raise CapabilityEvalError(f"Unsupported dataset: {dataset_key}")


def parse_prompt_style_map(values: list[str]) -> dict[str, str]:
    prompt_style_map: dict[str, str] = {}
    for item in parse_comma_values(values):
        if "=" not in item:
            raise CapabilityEvalError(f"Invalid --prompt-style-map entry, expected dataset=style: {item}")
        dataset_key, style = item.split("=", 1)
        dataset_key = dataset_key.strip()
        style = style.strip()
        if dataset_key not in SAMPLE_LOADERS:
            raise CapabilityEvalError(f"Unsupported --prompt-style-map dataset: {dataset_key}")
        if style not in {"default", "v11", "v15"}:
            raise CapabilityEvalError(f"Unsupported prompt style for {dataset_key}: {style}")
        prompt_style_map[dataset_key] = style
    return prompt_style_map


def parse_max_new_tokens_map(values: list[str]) -> dict[str, int]:
    token_map: dict[str, int] = {}
    for item in parse_comma_values(values):
        if "=" not in item:
            raise CapabilityEvalError(f"Invalid --max-new-tokens-map entry, expected dataset=count: {item}")
        dataset_key, count_text = item.split("=", 1)
        dataset_key = dataset_key.strip()
        if dataset_key not in SAMPLE_LOADERS:
            raise CapabilityEvalError(f"Unsupported --max-new-tokens-map dataset: {dataset_key}")
        try:
            count = int(count_text.strip())
        except ValueError as exc:
            raise CapabilityEvalError(f"Invalid max token count for {dataset_key}: {count_text}") from exc
        if count <= 0:
            raise CapabilityEvalError(f"Max token count must be positive for {dataset_key}")
        token_map[dataset_key] = count
    return token_map


def parse_request_extra_map(values: list[str]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for item in values:
        if "=" not in item:
            raise CapabilityEvalError(f"Invalid request-extra map entry: {item}")
        dataset_key, payload_text = item.split("=", 1)
        dataset_key = dataset_key.strip()
        if dataset_key not in SAMPLE_LOADERS:
            raise CapabilityEvalError(f"Unsupported request-extra dataset: {dataset_key}")
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise CapabilityEvalError("Each request-extra map value must be a JSON object")
        parsed[dataset_key] = payload
    return parsed


def parse_min_accuracy_map(values: list[str]) -> dict[str, float]:
    accuracy_map: dict[str, float] = {}
    for item in parse_comma_values(values):
        if "=" not in item:
            raise CapabilityEvalError(f"Invalid --fail-fast-min-accuracy-map entry: {item}")
        dataset_key, value_text = item.split("=", 1)
        dataset_key = dataset_key.strip()
        if dataset_key not in SAMPLE_LOADERS:
            raise CapabilityEvalError(f"Unsupported fail-fast dataset: {dataset_key}")
        try:
            value = float(value_text.strip())
        except ValueError as exc:
            raise CapabilityEvalError(f"Invalid fail-fast accuracy for {dataset_key}: {value_text}") from exc
        if not 0 <= value <= 1:
            raise CapabilityEvalError(f"Fail-fast accuracy must be in [0, 1] for {dataset_key}")
        accuracy_map[dataset_key] = value
    return accuracy_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 2 capability evaluation for GSM8K, HumanEval and CMMLU.")
    parser.add_argument("--local-model-dir", "--local_model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--student-url", "--student_url", default="")
    parser.add_argument("--student-model-id", "--student_model_id", default="")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset key filter; repeat or comma-separate.")
    parser.add_argument("--sample-limit-per-dataset", "--sample_limit_per_dataset", type=int, default=1)
    parser.add_argument(
        "--use-frozen-final",
        "--use_frozen_final",
        action="store_true",
        help="Evaluate frozen final test ids from data/splits/*_test.txt. Use --sample-limit-per-dataset 0 for all.",
    )
    parser.add_argument(
        "--split-dir",
        default=str(SPLITS),
        help="Directory containing the frozen full dataset_test.txt files.",
    )
    parser.add_argument("--output-trace", "--output_trace", default=str(DEFAULT_OUTPUT_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking=false to Qwen3 local templates and HTTP endpoints.",
    )
    parser.add_argument(
        "--kv-cache-type",
        default="",
        help="Audit-only endpoint KV cache type, for example q8_0.",
    )
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=160)
    parser.add_argument(
        "--max-new-tokens-map",
        "--max_new_tokens_map",
        action="append",
        default=[],
        help="Dataset-specific generation limits, e.g. cmmlu=16,gsm8k=160,humaneval=256.",
    )
    parser.add_argument(
        "--request-extra-json-map",
        action="append",
        default=[],
        help="Dataset-specific endpoint extras as dataset=JSON; repeat once per task.",
    )
    parser.add_argument(
        "--fail-fast-min-accuracy-map",
        "--fail_fast_min_accuracy_map",
        action="append",
        default=[],
        help="Stop before later datasets when a completed dataset is already below its absolute smoke threshold.",
    )
    parser.add_argument("--humaneval-timeout-sec", "--humaneval_timeout_sec", type=float, default=3.0)
    parser.add_argument("--timeout-sec", "--timeout_sec", type=float, default=120.0)
    parser.add_argument("--num-shards", "--num_shards", type=int, default=1)
    parser.add_argument("--shard-index", "--shard_index", type=int, default=0)
    parser.add_argument("--min-accuracy", "--min_accuracy", type=float, default=0.0)
    parser.add_argument("--prompt-style", "--prompt_style", choices=["default", "v11", "v15"], default="default")
    parser.add_argument(
        "--prompt-style-map",
        "--prompt_style_map",
        action="append",
        default=[],
        help="Dataset-specific prompt style mapping, e.g. gsm8k=default,humaneval=v11.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit_per_dataset < 0:
        print("--sample-limit-per-dataset must be >= 0", file=sys.stderr)
        return 2
    if not 0 <= args.min_accuracy <= 1:
        print("--min-accuracy must be in [0, 1]", file=sys.stderr)
        return 2
    if args.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 2
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        print("--shard-index must be in [0, num-shards)", file=sys.stderr)
        return 2

    requested = set(parse_comma_values(args.dataset)) if args.dataset else {"gsm8k", "humaneval", "cmmlu"}
    unsupported = requested - set(SAMPLE_LOADERS)
    if unsupported:
        print(f"Unsupported datasets: {', '.join(sorted(unsupported))}", file=sys.stderr)
        return 2

    try:
        prompt_style_map = parse_prompt_style_map(args.prompt_style_map)
        max_new_tokens_map = parse_max_new_tokens_map(args.max_new_tokens_map)
        request_extra_map = parse_request_extra_map(args.request_extra_json_map)
        fail_fast_min_accuracy_map = parse_min_accuracy_map(args.fail_fast_min_accuracy_map)
    except CapabilityEvalError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    model_dir = resolve_path(args.local_model_dir)
    split_dir = resolve_path(args.split_dir)
    output_path = resolve_path(args.output_trace)
    audit_path = resolve_path(args.audit)
    created_ts = datetime.now(timezone.utc).isoformat()

    backend = "openai_compatible" if args.student_url else "local_transformers_base"
    health = None
    requested_model_id = args.student_model_id or infer_model_id(model_dir)
    served_model_id = requested_model_id
    tokenizer = None
    model = None
    if args.student_url:
        health = health_status(args.student_url, args.timeout_sec)
        if health != 200:
            print(f"Student endpoint health check failed: {args.student_url} status={health}", file=sys.stderr)
            return 1
        served_model_id = get_served_model(args.student_url, args.timeout_sec, requested_model_id)
    elif args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if not args.student_url:
        tokenizer, model = load_local_student(model_dir, args.device, args.dtype)

    trace_rows: list[dict[str, Any]] = []
    unsharded_sample_ids: list[str] = []
    fail_fast_reason = ""
    for dataset_key in sorted(requested):
        dataset_prompt_style = prompt_style_map.get(dataset_key, args.prompt_style)
        dataset_max_new_tokens = max_new_tokens_map.get(dataset_key, args.max_new_tokens)
        samples = load_samples(
            dataset_key,
            args.sample_limit_per_dataset,
            args.use_frozen_final,
            split_dir,
        )
        unsharded_sample_ids.extend(str(sample["sample_id"]) for sample in samples)
        samples = apply_shard(samples, args.num_shards, args.shard_index)
        for index, sample in enumerate(samples, start=1):
            messages, prompt_hash = build_messages(sample, dataset_prompt_style)
            generation_error = ""
            generation_started = time.perf_counter()
            try:
                if args.student_url:
                    response_text, latency_ms = generate_text_endpoint(
                        args.student_url,
                        served_model_id,
                        messages,
                        args.timeout_sec,
                        dataset_max_new_tokens,
                        args.disable_thinking,
                        request_extra_map.get(dataset_key),
                    )
                elif tokenizer is not None and model is not None:
                    response_text, latency_ms = generate_text(
                        tokenizer,
                        model,
                        messages,
                        args.device,
                        dataset_max_new_tokens,
                        args.disable_thinking,
                    )
                else:
                    raise CapabilityEvalError("No model backend initialized")
                correct, prediction, score_detail = score_sample(sample, response_text, args.humaneval_timeout_sec)
            except CapabilityEvalError as exc:
                response_text = ""
                latency_ms = (time.perf_counter() - generation_started) * 1000
                correct = False
                prediction = ""
                generation_error = f"{type(exc).__name__}: {exc}"
                score_detail = f"generation_error: {generation_error}"
            row = {
                "capability_eval_version": "1.1",
                "created_by": "scripts/evaluate_chapter2_capability.py",
                "created_ts": created_ts,
                "probe_backend": backend,
                "dataset_key": dataset_key,
                "sample_id": sample["sample_id"],
                "prompt_style": dataset_prompt_style,
                "prompt_hash": prompt_hash,
                "max_new_tokens": dataset_max_new_tokens,
                "reference": sample.get("reference", ""),
                "prediction": prediction,
                "correct": correct,
                "score_detail": score_detail,
                "latency_ms": latency_ms,
                "response_text": response_text,
                "response_char_count": len(response_text),
                "generation_error": generation_error,
            }
            row["capability_eval_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
            trace_rows.append(row)
            print(
                f"[{backend}:{dataset_key}] {index}/{len(samples)} {sample['sample_id']} "
                f"correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )
        dataset_rows = [row for row in trace_rows if row["dataset_key"] == dataset_key]
        dataset_accuracy = (
            sum(1 for row in dataset_rows if row["correct"]) / len(dataset_rows) if dataset_rows else 0.0
        )
        fail_fast_threshold = fail_fast_min_accuracy_map.get(dataset_key)
        if fail_fast_threshold is not None and dataset_accuracy < fail_fast_threshold:
            fail_fast_reason = (
                f"dataset_accuracy_below_threshold: {dataset_key} "
                f"{dataset_accuracy:.6f} < {fail_fast_threshold:.6f}"
            )
            print(f"[FAIL FAST] {fail_fast_reason}", flush=True)
            break

    write_jsonl(output_path, trace_rows)
    counts = Counter(row["dataset_key"] for row in trace_rows)
    correct_counts = Counter(row["dataset_key"] for row in trace_rows if row["correct"])
    accuracy_by_dataset = {
        key: correct_counts[key] / counts[key] if counts[key] else 0.0
        for key in sorted(counts)
    }
    overall_accuracy = sum(1 for row in trace_rows if row["correct"]) / len(trace_rows) if trace_rows else 0.0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    status = "passed" if trace_rows and overall_accuracy >= args.min_accuracy and not fail_fast_reason else "failed"
    full_final = args.use_frozen_final and args.sample_limit_per_dataset == 0
    if args.num_shards > 1:
        gate = "CH2-CAPABILITY-EVAL-shard"
    elif full_final:
        gate = "CH2-CAPABILITY-EVAL"
    else:
        gate = "CH2-CAPABILITY-EVAL-smoke"
    audit = {
        "gate": gate,
        "check_version": "1.4",
        "created_by": "scripts/evaluate_chapter2_capability.py",
        "created_ts": created_ts,
        "status": status,
        "probe_backend": backend,
        "student_url": args.student_url,
        "student_endpoint_health_status": health,
        "student_model_id": requested_model_id,
        "served_model_id": served_model_id,
        "local_model_dir": display_path(model_dir),
        "dtype": args.dtype,
        "disable_thinking": bool(args.disable_thinking),
        "kv_cache_type": str(args.kv_cache_type),
        "device": args.device,
        "output_trace_path": display_path(output_path),
        "capability_eval_trace_hash": sha256_file(output_path),
        "selected_dataset_keys": sorted(requested),
        "use_frozen_final": bool(args.use_frozen_final),
        "split_dir": display_path(split_dir),
        "split_manifest_hash": (
            sha256_file(split_dir / "manifest.json")
            if (split_dir / "manifest.json").is_file()
            else ""
        ),
        "sample_limit_per_dataset": args.sample_limit_per_dataset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "unsharded_sample_count": len(unsharded_sample_ids),
        "unsharded_sample_ids_hash": sha256_text("\n".join(unsharded_sample_ids) + "\n"),
        "sample_count": len(trace_rows),
        "dataset_counts": dict(sorted(counts.items())),
        "correct_counts": dict(sorted(correct_counts.items())),
        "accuracy_by_dataset": accuracy_by_dataset,
        "overall_accuracy": overall_accuracy,
        "min_accuracy": args.min_accuracy,
        "peak_memory_mb": peak_memory_mb,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_map": dict(sorted(max_new_tokens_map.items())),
        "request_extra_map_hash": sha256_text(
            json.dumps(request_extra_map, ensure_ascii=False, sort_keys=True)
        ),
        "fail_fast_min_accuracy_map": dict(sorted(fail_fast_min_accuracy_map.items())),
        "fail_fast_reason": fail_fast_reason,
        "humaneval_timeout_sec": args.humaneval_timeout_sec,
        "generation_config": {
            "do_sample": False,
            "temperature": 0.0,
            "num_candidates": 1,
        },
        "prompt_style": args.prompt_style,
        "prompt_style_map": dict(sorted(prompt_style_map.items())),
        "prompt_template_hash": sha256_text(SYSTEM_PROMPT),
        "humaneval_exec_preamble_hash": sha256_text(HUMANEVAL_EXEC_PREAMBLE),
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"capability_eval_trace_hash={audit['capability_eval_trace_hash']}")
    if status != "passed":
        print("Chapter 2 capability eval smoke failed.", file=sys.stderr)
        return 1
    print("Chapter 2 capability eval smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
