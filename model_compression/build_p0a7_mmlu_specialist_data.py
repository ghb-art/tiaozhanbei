#!/usr/bin/env python3
"""Convert verified MMLU auxiliary_train translations into P0-A7 NLP data.

Only the Teacher-verified auxiliary training split is accepted.  The official
MMLU/CMMLU test splits and every formal evaluation artifact are outside this
builder's interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SOURCE = ROOT / "data/distill/p0a7_nlp_verified_train.jsonl"
DEFAULT_VALIDATION_SOURCE = ROOT / "data/distill/p0a7_nlp_verified_validation.jsonl"
DEFAULT_TRAIN_OUTPUT = ROOT / "data/p0a7/nlp_mmlu_aux_train.jsonl"
DEFAULT_VALIDATION_OUTPUT = ROOT / "data/p0a7/nlp_mmlu_aux_validation.jsonl"
DEFAULT_AUDIT = ROOT / "reports/audit/gate_p0a7_mmlu_specialist_data.json"
SYSTEM_PROMPT = (
    "请简要分析这道中文选择题。最后一行必须严格使用“最终答案：A”的格式，"
    "并将A替换为实际的A、B、C或D选项。"
)
ANSWER_RE = re.compile(r"^理由：(.+?)\nFINAL:\s*([A-D])\s*$", re.S)
PROMPT_RE = re.compile(r"^以下是单项选择题。\s*\n\s*题目:\s*(.+?)\n\s*\n请给出", re.S)


class DataError(RuntimeError):
    pass


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataError(f"Missing verified input: {display_path(path)}")
    lowered = {part.casefold() for part in path.resolve().parts}
    if lowered.intersection({"test", "eval", "formal", "sealed"}):
        raise DataError(f"Forbidden source path: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataError(f"Non-object row {line_number}")
            rows.append(value)
    if not rows:
        raise DataError(f"Empty verified input: {display_path(path)}")
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def convert_row(row: dict[str, Any], *, training: bool) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", "")).strip()
    group_id = str(row.get("validation_group_id", "")).strip()
    if not sample_id or not group_id:
        raise DataError("Verified row is missing sample/group identity")
    if row.get("origin") != "mmlu_auxiliary_train":
        raise DataError(f"Unapproved origin for {sample_id}: {row.get('origin')}")
    if row.get("used_for_final_test") is not False:
        raise DataError(f"Final-test marker is not false for {sample_id}")
    if bool(row.get("used_for_training")) != training:
        raise DataError(f"Split-role mismatch for {sample_id}")
    if bool(row.get("used_for_validation")) == training:
        raise DataError(f"Train/validation flags are inconsistent for {sample_id}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DataError(f"Missing verified messages for {sample_id}")
    user = str(messages[-1].get("content", ""))
    prompt_match = PROMPT_RE.search(user)
    if prompt_match is None:
        raise DataError(f"Cannot normalize verified prompt for {sample_id}")
    prompt = "问题：" + prompt_match.group(1).strip()
    answer_match = ANSWER_RE.fullmatch(str(row.get("answer", "")).strip())
    if answer_match is None:
        raise DataError(f"Cannot normalize verified answer for {sample_id}")
    reason = " ".join(answer_match.group(1).split())
    label = answer_match.group(2)
    if len(reason) > 300:
        raise DataError(f"Verified reason is too long for {sample_id}")
    common = {
        "sample_id": sample_id.replace("p0a4r3/", "p0a7/"),
        "validation_group_id": group_id,
        "dataset_key": "mmlu_aux_chinese",
        "domain": "nlp",
        "source": "MMLU-auxiliary_train-14B-verified-Chinese",
        "teacher_model_id": str(
            row.get("teacher_verification", {}).get("model_id", "")
        ),
    }
    if training:
        return {
            **common,
            "split_role": "train",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "answer": f"简短分析：{reason}\n最终答案：{label}",
            "answer_letter": label,
            "answer_token_position": "last",
            "answer_token_weight": 1.0,
            "training_weight": 1.0,
            "quality_weight": 1.0,
            "kl_weight": 0.05,
            "distill_validation": "teacher_choice_matches_source_train_label",
        }
    return {
        **common,
        "split_role": "p0a7_internal_validation",
        "prompt": prompt,
        "reference": label,
        "validator": "choice_exact",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--validation-source", default=str(DEFAULT_VALIDATION_SOURCE))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT))
    parser.add_argument("--validation-output", default=str(DEFAULT_VALIDATION_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--expected-train", type=int, default=3000)
    parser.add_argument("--expected-validation", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_source = resolve_path(args.train_source)
    validation_source = resolve_path(args.validation_source)
    train_output = resolve_path(args.train_output)
    validation_output = resolve_path(args.validation_output)
    audit_path = resolve_path(args.audit)
    train = [convert_row(row, training=True) for row in read_jsonl(train_source)]
    validation = [
        convert_row(row, training=False) for row in read_jsonl(validation_source)
    ]
    if len(train) != args.expected_train or len(validation) != args.expected_validation:
        raise DataError(
            f"Unexpected row counts: train={len(train)} validation={len(validation)}"
        )
    train_ids = {str(row["sample_id"]) for row in train}
    validation_ids = {str(row["sample_id"]) for row in validation}
    train_groups = {str(row["validation_group_id"]) for row in train}
    validation_groups = {str(row["validation_group_id"]) for row in validation}
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise DataError("Duplicate sample identity")
    if train_groups & validation_groups:
        raise DataError("Train/internal-validation overlap")
    atomic_write_jsonl(train_output, train)
    atomic_write_jsonl(validation_output, validation)
    audit = {
        "gate": "P0-A7-MMLU-AUX-CHINESE-SPECIALIST-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a7_mmlu_specialist_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source_split": "MMLU auxiliary_train only",
        "formal_test_loaded": False,
        "cmmlu_test_loaded": False,
        "train_source": display_path(train_source),
        "train_source_hash": sha256_file(train_source),
        "validation_source": display_path(validation_source),
        "validation_source_hash": sha256_file(validation_source),
        "train_output": display_path(train_output),
        "train_output_hash": sha256_file(train_output),
        "validation_output": display_path(validation_output),
        "validation_output_hash": sha256_file(validation_output),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "label_counts": dict(sorted(Counter(row["answer_letter"] for row in train).items())),
        "system_prompt_hash": sha256_text(SYSTEM_PROMPT),
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    atomic_write_json(audit_path, audit)
    print(f"Wrote {display_path(train_output)} rows={len(train)}")
    print(f"Wrote {display_path(validation_output)} rows={len(validation)}")
    print(f"Wrote {display_path(audit_path)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataError as exc:
        print(f"P0-A7 data build failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
