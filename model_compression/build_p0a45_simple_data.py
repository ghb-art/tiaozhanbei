#!/usr/bin/env python3
"""Build one simple, shared Math/Code/NLP Student training set."""

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
CONFIG = ROOT / "configs/p0a45_simple_shared.json"
SOURCE_DISTILL = ROOT / "data/capability_v2/distill_train.jsonl"
SOURCE_VALIDATION = ROOT / "data/capability_v2/internal_validation.jsonl"
NLP_SOURCES = (
    ROOT / "data/p0a32/train.jsonl",
    ROOT / "data/p0a34/train.jsonl",
    ROOT / "data/p0a36/train.jsonl",
)
NLP_VALIDATION = (
    ROOT / "data/p0a44/nlp_ceval_dev.jsonl",
    ROOT / "data/p0a44/nlp_cmmlu_dev.jsonl",
)
GATE = ROOT / "data/capability_v2/gate300.jsonl"
OUTPUT_DIR = ROOT / "data/p0a45"
TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VALIDATION_OUTPUT = OUTPUT_DIR / "internal_validation.jsonl"
AUDIT_OUTPUT = ROOT / "reports/audit/gate_p0a45_data.json"
DOMAINS = ("gsm8k", "opencodeinstruct", "cmmlu")
EXPECTED_SOURCE_COUNTS = {"gsm8k": 7173, "opencodeinstruct": 20000}
CHOICE_PATTERN = re.compile(r"(?:最终答案|FINAL)\s*[:：]\s*([ABCD])", re.IGNORECASE)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(
                    f"Invalid JSONL: {path.relative_to(ROOT)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise BuildError(f"Non-object row: {path.relative_to(ROOT)}:{line_number}")
            rows.append(row)
    if not rows:
        raise BuildError(f"Empty input: {path.relative_to(ROOT)}")
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def user_content(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    users = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return users[-1].strip() if users else ""


def minimal_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise BuildError(f"Missing messages: {row.get('sample_id')}")
    selected = [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "user"}
    ]
    if not selected or not any(message["role"] == "user" for message in selected):
        raise BuildError(f"Missing user prompt: {row.get('sample_id')}")
    return selected


def normalize_distill_row(row: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip()
    if not answer:
        raise BuildError(f"Missing answer: {row.get('sample_id')}")
    sample_id = str(row.get("sample_id", "")).strip()
    if not sample_id:
        raise BuildError("Distillation row has no sample_id")
    return {
        "sample_id": sample_id,
        "dataset_key": dataset_key,
        "domain": "math" if dataset_key == "gsm8k" else "code",
        "source": f"p0a5_verified_distill/{dataset_key}",
        "split_role": "train",
        "messages": minimal_messages(row),
        "answer": answer,
        "preserve_math": dataset_key == "gsm8k",
        "training_weight": 1.0,
    }


def normalize_nlp_answer(answer: str) -> tuple[str, str]:
    matches = CHOICE_PATTERN.findall(answer)
    if not matches:
        raise BuildError("NLP answer has no verified final option")
    letter = matches[-1].upper()
    rationale_lines = [
        line.rstrip()
        for line in answer.strip().splitlines()
        if not CHOICE_PATTERN.fullmatch(line.strip())
    ]
    rationale = "\n".join(line for line in rationale_lines if line.strip()).strip()
    if not rationale:
        rationale = "根据题目条件和各选项含义进行判断。"
    return f"{rationale}\n最终答案：{letter}", letter


def normalize_nlp_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt = user_content(row)
    if not prompt:
        raise BuildError(f"NLP row has no prompt: {row.get('sample_id')}")
    answer, letter = normalize_nlp_answer(str(row.get("answer", "")))
    identity = sha256_text(" ".join(prompt.split()) + "|" + letter)
    normalized = {
        "sample_id": f"p0a45/nlp/{identity[:24]}",
        "dataset_key": "cmmlu",
        "domain": "nlp",
        "source": f"p0a45_verified_mcq/{row.get('dataset_key', 'unknown')}",
        "split_role": "train",
        "messages": [
            {
                "role": "system",
                "content": (
                    "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
                    "请将A替换为实际选项，只能使用A、B、C或D。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "answer": answer,
        "preserve_math": False,
        "training_weight": 1.0,
        "source_dataset_key": str(row.get("dataset_key", "")),
    }
    return identity, normalized


def normalize_validation_row(row: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    normalized = normalize_distill_row(row, dataset_key)
    normalized["split_role"] = "internal_validation"
    normalized["source"] = f"p0a5_internal_validation/{dataset_key}"
    normalized["training_weight"] = 1.0
    return normalized


def normalize_nlp_validation(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", "")).strip()
    reference = str(row.get("reference", "")).strip().upper()
    if not prompt or reference not in {"A", "B", "C", "D"}:
        raise BuildError(f"Invalid NLP validation row: {row.get('sample_id')}")
    return {
        "sample_id": str(row["sample_id"]),
        "dataset_key": "cmmlu",
        "domain": "nlp",
        "source": "p0a44_labelled_mcq_dev",
        "split_role": "internal_validation",
        "messages": [
            {
                "role": "system",
                "content": (
                    "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
                    "请将A替换为实际选项，只能使用A、B、C或D。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "answer": f"最终答案：{reference}",
        "preserve_math": False,
        "training_weight": 1.0,
    }


def prompt_identity(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or user_content(row))
    return sha256_text(" ".join(prompt.split())) if prompt else ""


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    target_mass = {
        key: float(value)
        for key, value in config["student_training"]["task_loss_mass"].items()
    }
    if set(target_mass) != set(DOMAINS) or abs(sum(target_mass.values()) - 1.0) > 1e-9:
        raise BuildError(f"Invalid task loss mass: {target_mass}")

    source_rows = read_jsonl(SOURCE_DISTILL)
    train_by_domain: dict[str, list[dict[str, Any]]] = {
        "gsm8k": [],
        "opencodeinstruct": [],
        "cmmlu": [],
    }
    for row in source_rows:
        key = str(row.get("dataset_key", ""))
        if key in {"gsm8k", "opencodeinstruct"}:
            train_by_domain[key].append(normalize_distill_row(row, key))
    observed_source_counts = {
        key: len(train_by_domain[key]) for key in EXPECTED_SOURCE_COUNTS
    }
    if observed_source_counts != EXPECTED_SOURCE_COUNTS:
        raise BuildError(
            f"Verified Math/Code source changed: {observed_source_counts} "
            f"!= {EXPECTED_SOURCE_COUNTS}"
        )

    nlp_seen: set[str] = set()
    nlp_source_counts: Counter[str] = Counter()
    nlp_duplicate_count = 0
    for path in NLP_SOURCES:
        for row in read_jsonl(path):
            identity, normalized = normalize_nlp_row(row)
            if identity in nlp_seen:
                nlp_duplicate_count += 1
                continue
            nlp_seen.add(identity)
            train_by_domain["cmmlu"].append(normalized)
            nlp_source_counts[str(normalized["source_dataset_key"])] += 1
    if len(train_by_domain["cmmlu"]) < 10000:
        raise BuildError(f"Too few unique NLP MCQ rows: {len(train_by_domain['cmmlu'])}")

    total_rows = sum(len(rows) for rows in train_by_domain.values())
    for key, rows in train_by_domain.items():
        row_weight = target_mass[key] * total_rows / len(rows)
        for row in rows:
            row["training_weight"] = row_weight
    train_rows = [
        row for key in DOMAINS for row in train_by_domain[key]
    ]

    source_validation = read_jsonl(SOURCE_VALIDATION)
    validation_rows = [
        normalize_validation_row(row, str(row.get("dataset_key")))
        for row in source_validation
        if row.get("dataset_key") in {"gsm8k", "opencodeinstruct"}
    ]
    for path in NLP_VALIDATION:
        validation_rows.extend(normalize_nlp_validation(row) for row in read_jsonl(path))

    train_ids = {str(row["sample_id"]) for row in train_rows}
    validation_ids = {str(row["sample_id"]) for row in validation_rows}
    if len(train_ids) != len(train_rows):
        raise BuildError("Duplicate training sample_id")
    if len(validation_ids) != len(validation_rows):
        raise BuildError("Duplicate validation sample_id")
    if train_ids & validation_ids:
        raise BuildError(f"Train/validation identity overlap: {len(train_ids & validation_ids)}")

    gate_rows = read_jsonl(GATE)
    gate_ids = {str(row.get("sample_id", "")) for row in gate_rows}
    gate_prompt_ids = {prompt_identity(row) for row in gate_rows} - {""}
    train_prompt_ids = {prompt_identity(row) for row in train_rows} - {""}
    gate_id_overlap = len(train_ids & gate_ids)
    gate_prompt_overlap = len(train_prompt_ids & gate_prompt_ids)
    if gate_id_overlap or gate_prompt_overlap:
        raise BuildError(
            f"Training/gate overlap: identity={gate_id_overlap} prompt={gate_prompt_overlap}"
        )

    weighted_mass = Counter()
    for row in train_rows:
        weighted_mass[str(row["dataset_key"])] += float(row["training_weight"])
    mass_total = sum(weighted_mass.values())
    observed_mass = {key: weighted_mass[key] / mass_total for key in DOMAINS}
    if any(abs(observed_mass[key] - target_mass[key]) > 1e-9 for key in DOMAINS):
        raise BuildError(f"Weighted mass mismatch: {observed_mass}")

    atomic_jsonl(TRAIN_OUTPUT, train_rows)
    atomic_jsonl(VALIDATION_OUTPUT, validation_rows)
    validation_counts = Counter(str(row["dataset_key"]) for row in validation_rows)
    train_counts = {key: len(train_by_domain[key]) for key in DOMAINS}
    audit = {
        "gate": "P0-A45-SIMPLE-SHARED-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a45_simple_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "method": "one shared LoRA; supervised SFT only",
        "train_counts": train_counts,
        "validation_counts": dict(sorted(validation_counts.items())),
        "nlp_source_counts": dict(sorted(nlp_source_counts.items())),
        "nlp_duplicate_count": nlp_duplicate_count,
        "task_loss_mass": observed_mass,
        "train_validation_overlap": 0,
        "training_gate_identity_overlap": gate_id_overlap,
        "training_gate_prompt_overlap": gate_prompt_overlap,
        "formal_test_reference_count": 0,
        "outputs": {
            "train": {
                "path": TRAIN_OUTPUT.relative_to(ROOT).as_posix(),
                "rows": len(train_rows),
                "sha256": sha256_file(TRAIN_OUTPUT),
            },
            "validation": {
                "path": VALIDATION_OUTPUT.relative_to(ROOT).as_posix(),
                "rows": len(validation_rows),
                "sha256": sha256_file(VALIDATION_OUTPUT),
            },
        },
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                CONFIG,
                SOURCE_DISTILL,
                SOURCE_VALIDATION,
                *NLP_SOURCES,
                *NLP_VALIDATION,
                GATE,
            )
        },
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    atomic_json(AUDIT_OUTPUT, audit)
    print(f"Wrote {TRAIN_OUTPUT.relative_to(ROOT)} rows={len(train_rows)}")
    print(f"Wrote {VALIDATION_OUTPUT.relative_to(ROOT)} rows={len(validation_rows)}")
    print(f"Wrote {AUDIT_OUTPUT.relative_to(ROOT)}")
    print(
        f"status=passed train={train_counts} validation={dict(validation_counts)} "
        f"mass={observed_mass}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A45 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
