#!/usr/bin/env python3
"""Build the leak-safe train-only data for the final P0-A10 router candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/capability_v2/distill_train.jsonl"
MMLU_TRAIN = ROOT / "data/p0a7/nlp_mmlu_aux_train.jsonl"
CEVAL_TRAIN = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
MMLU_TRACE = ROOT / "data/distill/p0a7_nlp_teacher_trace.jsonl"
MMLU_USED = (
    ROOT / "data/distill/p0a7_nlp_verified_train.jsonl",
    ROOT / "data/distill/p0a7_nlp_verified_validation.jsonl",
)
OUTPUT_DIR = ROOT / "data/p0a10"
AUDIT = ROOT / "reports/audit/gate_p0a10_data.json"
TOKENIZER_DIR = ROOT / "models/checkpoints/p0a4/student-shared-merged"
SEED = 20260801
VALIDATION_ROWS = 256
MAX_SEQUENCE_LENGTH = 1536
MAX_GENERATION_TOKENS = {"math": 512, "code": 768, "nlp": 256}

PROMPTS = {
    "math": (
        "Solve the problem concisely. End with one line formatted as `#### 42`, "
        "where 42 is replaced by the actual numeric answer."
    ),
    "code": (
        "Return only a complete Python function implementation in one python code block. "
        "Do not use files, network access, third-party packages, or explanatory prose."
    ),
    "nlp": (
        "请简要分析这道中文选择题。最后一行必须严格使用“最终答案：A”的格式，"
        "并将A替换为实际的A、B、C或D选项。"
    ),
}


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(sample_id: str) -> str:
    return sha256_text(f"{SEED}:{sample_id}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def replace_system(messages: Any, domain: str) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise BuildError("messages is not a list")
    user = [item for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    if len(user) != 1:
        raise BuildError("expected exactly one user message")
    return [
        {"role": "system", "content": PROMPTS[domain]},
        {"role": "user", "content": str(user[0].get("content", ""))},
    ]


def training_row(row: dict[str, Any], domain: str) -> dict[str, Any]:
    copied = dict(row)
    copied["messages"] = replace_system(row.get("messages"), domain)
    copied["domain"] = domain
    copied["task_id"] = domain
    copied["split_role"] = "train"
    copied["answer_token_weight"] = float(row.get("answer_token_weight", 1.0))
    copied["quality_weight"] = float(row.get("quality_weight", 1.0))
    copied["training_weight"] = float(row.get("training_weight", 1.0))
    copied["kl_weight"] = {
        "math": 0.20,
        "code": 0.10,
        "nlp": float(row.get("kl_weight", 0.10)),
    }[domain]
    if domain == "nlp" and copied.get("dataset_key") == "cmmlu":
        copied["dataset_key"] = "mmlu_aux_chinese"
    return copied


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise BuildError("validation source has no messages")
    users = [item for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    if len(users) != 1:
        raise BuildError("validation source has invalid user messages")
    return str(users[0].get("content", ""))


def final_number(answer: str) -> str:
    suffix = answer.rsplit("####", 1)[-1]
    values = re.findall(r"-?\d+(?:\.\d+)?", suffix.replace(",", ""))
    if not values:
        raise BuildError("math answer has no final number")
    return values[-1]


def final_choice(answer: str) -> str:
    matches = re.findall(r"(?:FINAL|最终答案)\s*[:：]\s*([A-D])", answer, re.I)
    if not matches:
        raise BuildError("NLP answer has no final choice")
    return matches[-1].upper()


def token_lengths(tokenizer: Any, row: dict[str, Any]) -> tuple[int, int]:
    messages = row["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": str(row["answer"])}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(prompt_ids), len(full_ids)


def validation_row(row: dict[str, Any], domain: str) -> dict[str, Any]:
    result = {
        "sample_id": str(row["sample_id"]),
        "dataset_key": str(row["dataset_key"]),
        "domain": domain,
        "source": str(row.get("source", "")),
        "split_role": "p0a10_internal_validation",
        "prompt": user_prompt(row),
        "validator": {
            "math": "exact_numeric_answer",
            "code": "python_unit_tests",
            "nlp": "exact_choice",
        }[domain],
    }
    if domain == "math":
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        result["reference"] = str(metadata.get("reference_answer") or final_number(str(row["answer"])))
        result["unit_tests"] = []
    elif domain == "code":
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        tests = metadata.get("unit_tests")
        if not isinstance(tests, list) or not tests:
            raise BuildError(f"Code validation row has no unit tests: {row['sample_id']}")
        result["reference"] = "unit_tests"
        result["unit_tests"] = [str(item) for item in tests]
    else:
        result["reference"] = final_choice(str(row["answer"]))
        result["unit_tests"] = []
    return result


def main() -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("transformers is required for exact token filtering") from exc
    if not TOKENIZER_DIR.is_dir():
        raise BuildError(f"Missing tokenizer: {TOKENIZER_DIR.relative_to(ROOT)}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, trust_remote_code=True
    )
    source = read_jsonl(SOURCE)
    by_dataset = {key: [] for key in ("gsm8k", "opencodeinstruct")}
    for row in source:
        key = str(row.get("dataset_key", ""))
        if key in by_dataset:
            by_dataset[key].append(row)
    expected = {"gsm8k": 7173, "opencodeinstruct": 20000}
    if {key: len(value) for key, value in by_dataset.items()} != expected:
        raise BuildError(f"Unexpected source counts: { {k: len(v) for k,v in by_dataset.items()} }")

    train: list[dict[str, Any]] = []
    validation: dict[str, list[dict[str, Any]]] = {}
    token_rejections: Counter[str] = Counter()
    maximum_training_tokens: dict[str, int] = {domain: 0 for domain in ("math", "code", "nlp")}
    for dataset, domain in (("gsm8k", "math"), ("opencodeinstruct", "code")):
        ordered = sorted(by_dataset[dataset], key=lambda row: stable_key(str(row["sample_id"])))
        prepared = [(row, training_row(row, domain)) for row in ordered]
        validation_eligible: list[dict[str, Any]] = []
        training_eligible: list[dict[str, Any]] = []
        for source_row, prepared_row in prepared:
            prompt_tokens, full_tokens = token_lengths(tokenizer, prepared_row)
            maximum_training_tokens[domain] = max(maximum_training_tokens[domain], full_tokens)
            if prompt_tokens + MAX_GENERATION_TOKENS[domain] <= MAX_SEQUENCE_LENGTH:
                validation_eligible.append(source_row)
            else:
                token_rejections[f"{domain}_validation_prompt_budget"] += 1
            if full_tokens <= MAX_SEQUENCE_LENGTH:
                training_eligible.append(prepared_row)
            else:
                token_rejections[f"{domain}_training_sequence"] += 1
        held_out = validation_eligible[:VALIDATION_ROWS]
        held_out_ids = {str(row["sample_id"]) for row in held_out}
        selected = [row for row in training_eligible if str(row["sample_id"]) not in held_out_ids]
        if len(held_out) != VALIDATION_ROWS:
            raise BuildError(f"Insufficient {domain} token-safe validation rows")
        validation[domain] = [validation_row(row, domain) for row in held_out]
        train.extend(selected)

    mmlu_train = read_jsonl(MMLU_TRAIN)
    ceval_train = read_jsonl(CEVAL_TRAIN)
    if len(mmlu_train) != 3000 or len(ceval_train) != 1335:
        raise BuildError("Unexpected NLP training counts")
    for source_row in [*mmlu_train, *ceval_train]:
        row = training_row(source_row, "nlp")
        _prompt_tokens, full_tokens = token_lengths(tokenizer, row)
        maximum_training_tokens["nlp"] = max(maximum_training_tokens["nlp"], full_tokens)
        if full_tokens <= MAX_SEQUENCE_LENGTH:
            train.append(row)
        else:
            token_rejections["nlp_training_sequence"] += 1

    used = {
        str(row["sample_id"])
        for path in MMLU_USED
        for row in read_jsonl(path)
    }
    leftovers: dict[str, dict[str, Any]] = {}
    for trace_row in read_jsonl(MMLU_TRACE):
        row = trace_row.get("verified_row")
        if isinstance(row, dict) and str(row.get("sample_id", "")) not in used:
            leftovers[str(row["sample_id"])] = row
    if len(leftovers) != VALIDATION_ROWS:
        raise BuildError(f"Expected {VALIDATION_ROWS} unused NLP rows, found {len(leftovers)}")
    validation["nlp"] = [
        validation_row(row, "nlp")
        for row in sorted(leftovers.values(), key=lambda row: stable_key(str(row["sample_id"])))
    ]
    for row in leftovers.values():
        prepared = training_row(row, "nlp")
        prompt_tokens, _full_tokens = token_lengths(tokenizer, prepared)
        if prompt_tokens + MAX_GENERATION_TOKENS["nlp"] > MAX_SEQUENCE_LENGTH:
            raise BuildError(f"NLP validation prompt exceeds budget: {row['sample_id']}")

    train.sort(key=lambda row: stable_key(str(row["sample_id"])))
    train_ids = [str(row["sample_id"]) for row in train]
    if len(train_ids) != len(set(train_ids)):
        raise BuildError("Duplicate training sample ids")
    validation_ids = {
        str(row["sample_id"]) for rows in validation.values() for row in rows
    }
    overlap = len(set(train_ids) & validation_ids)
    if overlap:
        raise BuildError(f"Train-validation overlap: {overlap}")

    train_path = OUTPUT_DIR / "train.jsonl"
    write_jsonl(train_path, train)
    outputs: dict[str, Any] = {
        "train": {
            "path": train_path.relative_to(ROOT).as_posix(),
            "rows": len(train),
            "sha256": sha256_file(train_path),
        }
    }
    for domain, rows in validation.items():
        path = OUTPUT_DIR / f"{domain}_validation.jsonl"
        write_jsonl(path, rows)
        outputs[f"{domain}_validation"] = {
            "path": path.relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    audit = {
        "gate": "P0-A10-FINAL-CANDIDATE-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a10_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "policy": {
            "source_gate300_opened": False,
            "formal_full_opened": False,
            "p0a9_per_question_feedback_used": False,
            "selection_feedback": "P0-A9 aggregate domain deficits only",
            "validation_rows_per_domain": VALIDATION_ROWS,
        },
        "train_counts": dict(sorted(Counter(str(row["domain"]) for row in train).items())),
        "validation_counts": {key: len(value) for key, value in sorted(validation.items())},
        "token_filter": {
            "tokenizer": TOKENIZER_DIR.relative_to(ROOT).as_posix(),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "rejections": dict(sorted(token_rejections.items())),
            "maximum_observed_training_tokens_before_filter": maximum_training_tokens,
        },
        "train_validation_overlap": overlap,
        "outputs": outputs,
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (SOURCE, MMLU_TRAIN, CEVAL_TRAIN, MMLU_TRACE, *MMLU_USED)
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(f"Wrote {train_path.relative_to(ROOT)} rows={len(train)}")
    for domain, rows in validation.items():
        print(f"Wrote data/p0a10/{domain}_validation.jsonl rows={len(rows)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A10 data build failed: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
