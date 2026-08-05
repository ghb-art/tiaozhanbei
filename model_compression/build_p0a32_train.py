#!/usr/bin/env python3
"""Build the leak-safe P0-A32 NLP continuation corpus and validation manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NEW_TRAIN = ROOT / "data/p0a32/nlp_train.jsonl"
NEW_VALIDATION = ROOT / "data/p0a32/nlp_validation.jsonl"
MMLU_REPLAY = ROOT / "data/p0a7/nlp_mmlu_aux_train.jsonl"
CEVAL_REPLAY = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
OUTPUT_TRAIN = ROOT / "data/p0a32/train.jsonl"
OUTPUT_VALIDATION = ROOT / "data/p0a32/nlp_internal_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a32_train_data.json"
TEACHER_AUDIT = ROOT / "reports/audit/gate_p0a32_teacher_data.json"

EXPECTED = {
    "new_train": 4000,
    "new_validation": 500,
    "mmlu_replay": 3000,
    "ceval_replay": 1335,
}
SYSTEM_PROMPT = (
    "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
    "请将A替换为实际选项，只能使用A、B、C或D，禁止输出占位符。"
)
FINAL_RE = re.compile(r"(?:FINAL|最终答案)\s*[:：]\s*([A-D])", re.I)
REASON_RE = re.compile(r"^(?:理由|简短分析)\s*[:：]\s*", re.I)


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise BuildError(f"Non-object row in {path.relative_to(ROOT)}")
    return rows


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


def user_content(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise BuildError(f"Missing messages: {row.get('sample_id')}")
    users = [item for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    if len(users) != 1:
        raise BuildError(f"Expected one user message: {row.get('sample_id')}")
    return str(users[0].get("content", "")).strip()


def normalized_prompt(row: dict[str, Any]) -> str:
    content = user_content(row)
    content = re.sub(r"^以下是单项选择题。\s*", "", content)
    content = re.sub(r"^题目\s*[:：]\s*", "问题：", content)
    content = re.sub(
        r"\s*请给出一条简短理由，最后一行严格写成\s*FINAL\s*:\s*X。?\s*$",
        "",
        content,
        flags=re.I,
    )
    if not content.startswith("问题："):
        content = "问题：" + content
    if not all(f"{letter}." in content for letter in "ABCD"):
        raise BuildError(f"Malformed MCQ prompt: {row.get('sample_id')}")
    return content.strip()


def answer_parts(row: dict[str, Any]) -> tuple[str, str]:
    answer = str(row.get("answer", "")).strip()
    matches = FINAL_RE.findall(answer)
    if not matches:
        raise BuildError(f"Missing final answer: {row.get('sample_id')}")
    letter = matches[-1].upper()
    reason = FINAL_RE.sub("", answer).strip()
    reason = REASON_RE.sub("", reason).strip()
    reason = reason.rstrip("。；; \n")
    if not reason:
        raise BuildError(f"Missing rationale: {row.get('sample_id')}")
    return reason, letter


def new_training_row(row: dict[str, Any]) -> dict[str, Any]:
    reason, letter = answer_parts(row)
    return {
        "sample_id": str(row["sample_id"]),
        "dataset_key": "mmlu_aux_chinese",
        "domain": "nlp",
        "task_id": "nlp",
        "source": "P0-A32-MMLU-auxiliary_train-14B-verified-Chinese",
        "split_role": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized_prompt(row)},
        ],
        "answer": f"简短分析：{reason}。\n最终答案：{letter}",
        "answer_letter": letter,
        "answer_token_position": "last",
        "answer_token_weight": 2.0,
        "quality_weight": 1.0,
        "training_weight": 1.0,
        "kl_weight": 0.10,
        "validation_group_id": str(row["validation_group_id"]),
        "teacher_model_id": str(
            (row.get("teacher_verification") or {}).get("model_id", "p0a32-teacher14b")
        ),
        "distill_validation": "teacher_choice_matches_source_train_label",
    }


def replay_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    prompt = normalized_prompt(row) if str(row.get("dataset_key")) == "mmlu_aux_chinese" else user_content(row)
    reason, letter = answer_parts(row)
    copied.update(
        {
            "domain": "nlp",
            "task_id": "nlp",
            "split_role": "train",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "answer": f"简短分析：{reason}。\n最终答案：{letter}",
            "answer_letter": letter,
            "answer_token_position": "last",
            "answer_token_weight": 2.0,
            "training_weight": 1.0,
        }
    )
    copied["kl_weight"] = 0.10 if str(copied.get("dataset_key")) == "mmlu_aux_chinese" else 0.20
    return copied


def validation_row(row: dict[str, Any]) -> dict[str, Any]:
    _reason, letter = answer_parts(row)
    return {
        "sample_id": str(row["sample_id"]),
        "dataset_key": "mmlu_aux_chinese",
        "domain": "nlp",
        "subject": str(row.get("domain", "")),
        "prompt": normalized_prompt(row),
        "reference": letter,
        "validator": "choice_exact",
        "split_role": "p0a32_external_validation",
    }


def main() -> int:
    teacher_audit = json.loads(TEACHER_AUDIT.read_text(encoding="utf-8"))
    if teacher_audit.get("status") != "passed":
        raise BuildError("P0-A32 Teacher data audit has not passed")
    sources = {
        "new_train": read_jsonl(NEW_TRAIN),
        "new_validation": read_jsonl(NEW_VALIDATION),
        "mmlu_replay": read_jsonl(MMLU_REPLAY),
        "ceval_replay": read_jsonl(CEVAL_REPLAY),
    }
    counts = {name: len(rows) for name, rows in sources.items()}
    if counts != EXPECTED:
        raise BuildError(f"Unexpected source counts: {counts}")

    train = [new_training_row(row) for row in sources["new_train"]]
    train.extend(replay_row(row) for row in sources["mmlu_replay"])
    train.extend(replay_row(row) for row in sources["ceval_replay"])
    validation = [validation_row(row) for row in sources["new_validation"]]
    train.sort(key=lambda row: str(row["sample_id"]))
    validation.sort(key=lambda row: str(row["sample_id"]))

    train_ids = [str(row["sample_id"]) for row in train]
    validation_ids = [str(row["sample_id"]) for row in validation]
    if len(train_ids) != len(set(train_ids)):
        raise BuildError("Duplicate training sample ids")
    if len(validation_ids) != len(set(validation_ids)):
        raise BuildError("Duplicate validation sample ids")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise BuildError(f"Train-validation sample overlap: {len(overlap)}")

    write_jsonl(OUTPUT_TRAIN, train)
    write_jsonl(OUTPUT_VALIDATION, validation)
    report = {
        "gate": "P0-A32-NLP-CONTINUATION-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a32_train.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "policy": {
            "formal_test_opened": False,
            "frozen_nlp100_opened": False,
            "selection_source": "new_train_only_validation",
            "answer_protocol": "Chinese rationale followed by 最终答案：X",
        },
        "source_counts": counts,
        "train_rows": len(train),
        "train_dataset_counts": dict(Counter(str(row["dataset_key"]) for row in train)),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (NEW_TRAIN, NEW_VALIDATION, MMLU_REPLAY, CEVAL_REPLAY, TEACHER_AUDIT)
        },
        "outputs": {
            OUTPUT_TRAIN.relative_to(ROOT).as_posix(): sha256_file(OUTPUT_TRAIN),
            OUTPUT_VALIDATION.relative_to(ROOT).as_posix(): sha256_file(OUTPUT_VALIDATION),
        },
        "errors": [],
    }
    write_json(AUDIT, report)
    print(f"Wrote {OUTPUT_TRAIN.relative_to(ROOT)} rows={len(train)}")
    print(f"Wrote {OUTPUT_VALIDATION.relative_to(ROOT)} rows={len(validation)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A32 train data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
