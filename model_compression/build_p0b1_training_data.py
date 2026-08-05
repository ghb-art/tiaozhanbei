#!/usr/bin/env python3
"""Assemble the frozen P0-B1 7,173/20,000/30,000 shared training corpus."""

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
CONFIG = ROOT / "configs/p0b1_converged_shared.json"
MATH_TRAIN = ROOT / "data/p0a45/train.jsonl"
MATH_VALIDATION = ROOT / "data/capability_v2/internal_validation.jsonl"
CODE_TRAIN = ROOT / "data/p0b1/code_train.jsonl"
CODE_VALIDATION = ROOT / "data/p0b1/code_validation.jsonl"
NLP_PRIOR = ROOT / "data/p0a45/train.jsonl"
NLP_NEW = ROOT / "data/p0b1/nlp_new_train.jsonl"
NLP_NEW_VALIDATION = ROOT / "data/p0b1/nlp_new_validation.jsonl"
GATE300 = ROOT / "data/capability_v2/gate300.jsonl"
TRAIN = ROOT / "data/p0b1/train.jsonl"
VALIDATION = ROOT / "data/p0b1/internal_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0b1_data.json"
SEED = 20260804
SYSTEM = "你是严谨的边缘推理助手。严格按照用户要求的格式作答。"
OPTION_RE = re.compile(r"(?:^|\n)\s*([A-D])[\.．、:：]\s*", re.I)
ANSWER_RE = re.compile(r"(?:FINAL|最终答案|答案)\s*[:：]\s*([A-D])", re.I)


class BuildError(RuntimeError):
    pass


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
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def user_prompt(row: dict[str, Any]) -> str:
    users = [
        str(item.get("content", ""))
        for item in row.get("messages", [])
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    return users[-1].strip() if users else str(row.get("prompt", "")).strip()


def mcq_parts(prompt: str) -> tuple[str, dict[str, str]] | None:
    matches = list(OPTION_RE.finditer(prompt))
    if [match.group(1).upper() for match in matches] != list("ABCD"):
        return None
    prefix = prompt[: matches[0].start()].strip()
    question_match = re.search(r"(?:题目|问题)\s*[:：]\s*(.*)", prefix, re.S)
    question = (question_match.group(1) if question_match else prefix).strip()
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        value = prompt[match.end() : end].strip()
        value = re.split(
            r"\n\s*(?:请(?:给出|选择|只输出|作答)|最终答案|FINAL)\s*[:：]?",
            value,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        options[match.group(1).upper()] = value
    if not question or any(not options[key] for key in "ABCD"):
        return None
    return question, options


def answer_letter(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    candidates = [
        str(row.get("answer_letter", "")),
        str(metadata.get("reference_answer", "")),
    ]
    for candidate in candidates:
        if candidate.strip().upper() in "ABCD" and len(candidate.strip()) == 1:
            return candidate.strip().upper()
    matches = ANSWER_RE.findall(str(row.get("answer", "")))
    return matches[-1].upper() if matches else None


def normalize_nlp(row: dict[str, Any], split_role: str) -> tuple[str, dict[str, Any]] | None:
    parsed = mcq_parts(user_prompt(row))
    label = answer_letter(row)
    if parsed is None or label not in "ABCD":
        return None
    question, options = parsed
    identity = sha256_text(
        " ".join(question.casefold().split())
        + json.dumps(options, ensure_ascii=False, sort_keys=True)
    )
    prompt = (
        "以下是单项选择题。请判断唯一正确选项，最终只输出一个大写字母 "
        "A、B、C 或 D，不要解释。\n\n"
        f"题目：{question}\n"
        + "\n".join(f"{key}. {options[key]}" for key in "ABCD")
        + "\n最终答案："
    )
    return identity, {
        "sample_id": f"p0b1/nlp/{identity[:24]}",
        "dataset_key": "cmmlu",
        "domain": "nlp",
        "source": f"p0b1_mcq/{row.get('source', row.get('dataset_key', 'unknown'))}",
        "split_role": split_role,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "answer": label,
        "preserve_math": False,
        "training_weight": 1.0,
        "kl_weight": 0.12,
        "metadata": {
            "output_contract": "cmmlu_v15_single_letter",
            "source_sample_id": str(row.get("sample_id", "")),
            "reference_answer": label,
            "source_domain": str(row.get("domain", "")),
        },
    }


def normalize_existing(row: dict[str, Any], dataset: str, split_role: str) -> dict[str, Any]:
    copied = dict(row)
    copied["dataset_key"] = dataset
    copied["domain"] = "math" if dataset == "gsm8k" else "code"
    copied["split_role"] = split_role
    copied["preserve_math"] = dataset == "gsm8k"
    copied["training_weight"] = 1.0
    copied["kl_weight"] = 0.25 if dataset == "gsm8k" else 0.12
    return copied


def prompt_identity(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or user_prompt(row))
    return sha256_text(" ".join(prompt.casefold().split())) if prompt else ""


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {key: int(value) for key, value in config["data"]["train_counts"].items()}
    math = [
        normalize_existing(row, "gsm8k", "train")
        for row in read_jsonl(MATH_TRAIN)
        if str(row.get("dataset_key")) == "gsm8k"
    ]
    code = [
        normalize_existing(row, "opencodeinstruct", "train")
        for row in read_jsonl(CODE_TRAIN)
    ]
    nlp_seen: dict[str, dict[str, Any]] = {}
    nlp_rejections: Counter[str] = Counter()
    nlp_source_counts: Counter[str] = Counter()
    for path in (NLP_NEW, NLP_PRIOR):
        for row in read_jsonl(path):
            if path == NLP_PRIOR and str(row.get("dataset_key")) != "cmmlu":
                continue
            normalized = normalize_nlp(row, "train")
            if normalized is None:
                nlp_rejections["unparseable_or_unlabelled"] += 1
                continue
            identity, value = normalized
            if identity in nlp_seen:
                nlp_rejections["semantic_duplicate"] += 1
                continue
            nlp_seen[identity] = value
            nlp_source_counts[str(value["source"])] += 1
    nlp = sorted(
        nlp_seen.values(),
        key=lambda row: sha256_text(f"{SEED}:nlp-train:{row['sample_id']}"),
    )[: expected["cmmlu"]]
    observed = {
        "gsm8k": len(math),
        "opencodeinstruct": len(code),
        "cmmlu": len(nlp),
    }
    if observed != expected:
        raise BuildError(f"Training counts changed: {observed} != {expected}")

    validation_math = [
        normalize_existing(row, "gsm8k", "internal_validation")
        for row in read_jsonl(MATH_VALIDATION)
        if str(row.get("dataset_key")) == "gsm8k"
    ]
    validation_code = sorted(
        (
            normalize_existing(row, "opencodeinstruct", "internal_validation")
            for row in read_jsonl(CODE_VALIDATION)
        ),
        key=lambda row: sha256_text(f"{SEED}:code-validation:{row['sample_id']}"),
    )[:300]
    validation_nlp_values: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(NLP_NEW_VALIDATION):
        normalized = normalize_nlp(row, "internal_validation")
        if normalized is not None:
            validation_nlp_values.setdefault(*normalized)
    validation_nlp = sorted(
        validation_nlp_values.values(),
        key=lambda row: sha256_text(f"{SEED}:nlp-validation:{row['sample_id']}"),
    )[:300]
    validation = validation_math + validation_code + validation_nlp
    validation_counts = Counter(str(row["dataset_key"]) for row in validation)
    expected_validation = {
        key: int(value) for key, value in config["data"]["validation_counts"].items()
    }
    if dict(validation_counts) != expected_validation:
        raise BuildError(
            f"Validation counts changed: {dict(validation_counts)} != {expected_validation}"
        )

    train = math + code + nlp
    target_mass = {
        key: float(value)
        for key, value in config["student_training"]["task_loss_mass"].items()
    }
    total = len(train)
    for key, rows in (
        ("gsm8k", math),
        ("opencodeinstruct", code),
        ("cmmlu", nlp),
    ):
        weight = target_mass[key] * total / len(rows)
        for row in rows:
            row["training_weight"] = weight

    train_ids = {str(row["sample_id"]) for row in train}
    validation_ids = {str(row["sample_id"]) for row in validation}
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise BuildError("Duplicate final train or validation sample_id")
    if train_ids & validation_ids:
        raise BuildError("Final train-validation id overlap")
    train_prompts = {prompt_identity(row) for row in train} - {""}
    validation_prompts = {prompt_identity(row) for row in validation} - {""}
    if train_prompts & validation_prompts:
        raise BuildError("Final train-validation prompt overlap")
    gate = read_jsonl(GATE300)
    gate_ids = {str(row.get("sample_id", "")) for row in gate}
    gate_prompts = {prompt_identity(row) for row in gate} - {""}
    if train_ids & gate_ids or train_prompts & gate_prompts:
        raise BuildError("P0-B1 training overlaps the frozen 300-item gate")

    weighted = Counter()
    for row in train:
        weighted[str(row["dataset_key"])] += float(row["training_weight"])
    weighted_total = sum(weighted.values())
    observed_mass = {key: weighted[key] / weighted_total for key in target_mass}
    if any(abs(observed_mass[key] - target_mass[key]) > 1e-9 for key in target_mass):
        raise BuildError(f"Weighted task mass mismatch: {observed_mass}")
    atomic_jsonl(TRAIN, train)
    atomic_jsonl(VALIDATION, validation)
    report = {
        "gate": "P0-B1-CONVERGED-SHARED-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0b1_training_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "train_counts": observed,
        "validation_counts": dict(validation_counts),
        "weighted_task_mass": observed_mass,
        "nlp_unique_candidates": len(nlp_seen),
        "nlp_source_counts": dict(sorted(nlp_source_counts.items())),
        "nlp_rejections": dict(sorted(nlp_rejections.items())),
        "output_contracts": {
            "math": "gsm8k_rationale_final_numeric",
            "code": "humaneval_v15_body_only",
            "nlp": "cmmlu_v15_single_letter",
        },
        "separation": {
            "train_validation_id_overlap": 0,
            "train_validation_prompt_overlap": 0,
            "train_gate_id_overlap": 0,
            "train_gate_prompt_overlap": 0,
            "formal_full_rows_loaded": 0,
            "formal_item_feedback_used": False,
        },
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                CONFIG,
                MATH_TRAIN,
                MATH_VALIDATION,
                CODE_TRAIN,
                CODE_VALIDATION,
                NLP_PRIOR,
                NLP_NEW,
                NLP_NEW_VALIDATION,
                GATE300,
            )
        },
        "outputs": {
            TRAIN.relative_to(ROOT).as_posix(): {"rows": len(train), "sha256": sha256_file(TRAIN)},
            VALIDATION.relative_to(ROOT).as_posix(): {"rows": len(validation), "sha256": sha256_file(VALIDATION)},
        },
        "errors": [],
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    atomic_json(AUDIT, report)
    print(f"Wrote {TRAIN.relative_to(ROOT)} rows={len(train)}")
    print(f"Wrote {VALIDATION.relative_to(ROOT)} rows={len(validation)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B1 data build failed: {exc}")
        raise SystemExit(1)
