#!/usr/bin/env python3
"""Train the P0-A6 shared Student LoRA with all-task base preservation.

This trainer intentionally has no evaluation-data argument.  Its only input is
the train-only JSONL corpus (``data/p0a6/train.jsonl`` by default).  Model
selection is performed by a separate, generation-based internal validator.

The important distributed invariant is that *every rank, for every batch*,
runs the adapter-disabled reference forward.  Do not make that forward
conditional on a task or on ``kl_weight``: different DDP ranks commonly see
different domains in the same step, and a conditional model call can deadlock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/checkpoints/p0a4/student-shared-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/p0a6/train.jsonl"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DOMAINS = ("math", "code", "nlp")
DOMAIN_TO_ID = {name: index for index, name in enumerate(DOMAINS)}
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math",
    "opencodeinstruct": "code",
    "open_code_instruct": "code",
    "code": "code",
    "coig_cqia": "nlp",
    "coig-cqia": "nlp",
    "coig_choice": "nlp",
    "ceval": "nlp",
    "c-eval": "nlp",
    "cmmlu": "nlp",
    "mmlu_aux_chinese": "nlp",
    "nlp": "nlp",
}
FORBIDDEN_INPUT_ROOTS = (
    ROOT / "data/eval",
    ROOT / "data/splits",
    ROOT / "data/formal",
    ROOT / "reports/sealed",
)
FORMAL_MARKERS = (
    "gsm8k/test/",
    "cmmlu/test/",
    "humaneval/",
    "official_full",
    "formal_test",
    "final_test",
)
FINAL_ANSWER_RE = re.compile(
    r"(?:最终答案|答案)\s*[:：]?\s*([A-D])(?=$|[\s。．.!！?？、）)\]])",
    re.IGNORECASE,
)

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-6
KL_TEMPERATURE = 2.0
MAX_SEQUENCE_LENGTH = 1536
DEFAULT_MAX_STEPS = 200
CHECKPOINT_STEPS = 100
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
EXPECTED_WORLD_SIZE = 4
DEFAULT_PER_DEVICE_BATCH = 1
DEFAULT_GRADIENT_ACCUMULATION = 8
SEED = 20260731


class TrainingError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in ("torch", "transformers", "accelerate", "peft"):
        if importlib.util.find_spec(module) is None:
            versions[module] = "missing"
            continue
        loaded = __import__(module)
        versions[module] = str(getattr(loaded, "__version__", "unknown"))
    return versions


def normalize_domain(row: dict[str, Any]) -> str:
    explicit = str(row.get("domain", "")).strip().casefold()
    if explicit in DOMAIN_TO_ID:
        return explicit
    dataset = str(row.get("dataset_key", "")).strip().casefold()
    if dataset in DATASET_TO_DOMAIN:
        return DATASET_TO_DOMAIN[dataset]
    raise TrainingError(
        f"Unknown task domain for sample {row.get('sample_id', '<missing>')}: "
        f"domain={explicit!r}, dataset_key={dataset!r}"
    )


def final_answer_letter(answer: str) -> str | None:
    matches = list(FINAL_ANSWER_RE.finditer(answer.strip()))
    return matches[-1].group(1).upper() if matches else None


def final_math_answer(answer: str) -> str | None:
    """Return the locked GSM8K value after the final ``####`` marker."""

    if "####" not in answer:
        return None
    value = answer.rsplit("####", 1)[-1].strip().splitlines()[0].strip()
    return value or None


def validate_batch_layout(
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    if world_size != EXPECTED_WORLD_SIZE:
        raise TrainingError(
            f"P0-A6 requires {EXPECTED_WORLD_SIZE}-GPU DDP; WORLD_SIZE={world_size}"
        )
    global_batch = world_size * per_device_batch_size * gradient_accumulation_steps
    if global_batch != 32:
        raise TrainingError(
            "P0-A6 effective global batch must remain 32: "
            f"{world_size} * {per_device_batch_size} * "
            f"{gradient_accumulation_steps} = {global_batch}"
        )
    return global_batch


def _validate_input_path(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_INPUT_ROOTS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            raise TrainingError(
                f"Evaluation/formal/sealed data cannot train a model: {display_path(path)}"
            )
    lowered_parts = {part.casefold() for part in resolved.parts}
    if lowered_parts.intersection({"eval", "formal", "sealed", "test"}):
        raise TrainingError(f"Forbidden training-data path: {display_path(path)}")


def read_rows(path: Path, focus_domain: str = "all") -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingError(f"Missing training data: {display_path(path)}")
    _validate_input_path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainingError(f"Training row {line_number} is not an object")
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen:
                raise TrainingError(
                    f"Missing or duplicate sample_id at line {line_number}: {sample_id!r}"
                )
            seen.add(sample_id)
            identity = " ".join(
                str(row.get(key, "")).casefold()
                for key in ("sample_id", "source", "split_role", "dataset_key")
            )
            if any(marker in identity for marker in FORMAL_MARKERS):
                raise TrainingError(f"Formal-test reference at line {line_number}")
            if str(row.get("split_role", "")) != "train":
                raise TrainingError(
                    f"Only split_role=train is accepted (line {line_number})"
                )
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise TrainingError(f"Missing messages at line {line_number}")
            if not str(row.get("answer", "")).strip():
                raise TrainingError(f"Missing answer at line {line_number}")
            row["domain"] = normalize_domain(row)
            for key, lower, upper in (
                ("kl_weight", 0.0, 1.0),
                ("answer_token_weight", 1.0, None),
                ("training_weight", 0.0, None),
            ):
                if key not in row:
                    raise TrainingError(f"Missing {key} at line {line_number}")
                try:
                    numeric = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise TrainingError(f"Invalid {key} at line {line_number}") from exc
                lower_ok = numeric >= lower if key == "answer_token_weight" else numeric > lower
                if key == "kl_weight":
                    lower_ok = numeric >= lower
                if not math.isfinite(numeric) or not lower_ok or (
                    upper is not None and numeric > upper
                ):
                    raise TrainingError(f"Invalid {key}={numeric} at line {line_number}")
                row[key] = numeric
            if row["answer_token_weight"] > 1.0:
                if row["domain"] == "nlp":
                    letter = final_answer_letter(str(row["answer"]))
                    if letter is None:
                        raise TrainingError(
                            "weighted NLP answer requires a normalized "
                            f"'最终答案：A-D' target at line {line_number}"
                        )
                    row["answer_letter"] = letter
                elif row["domain"] == "math":
                    value = final_math_answer(str(row["answer"]))
                    if value is None:
                        raise TrainingError(
                            "weighted Math answer requires a final '#### value' "
                            f"target at line {line_number}"
                        )
                    row["answer_value"] = value
                else:
                    raise TrainingError(
                        "answer_token_weight > 1 is only supported for Math and NLP "
                        f"(line {line_number})"
                    )
            rows.append(row)
    if not rows:
        raise TrainingError(f"Empty training data: {display_path(path)}")
    counts = Counter(str(row["domain"]) for row in rows)
    if focus_domain in DOMAINS:
        observed = set(counts)
        if observed not in ({focus_domain}, set(DOMAINS)):
            raise TrainingError(
                f"Focused {focus_domain} corpus must contain only that domain or "
                f"the complete three-domain corpus: {dict(counts)}"
            )
    elif focus_domain in {
        "nlp_rationale",
        "nlp_answer_first",
        "nlp_mmlu_aux",
        "nlp_mixed_mcq",
    }:
        if set(counts) != {"nlp"}:
            raise TrainingError(
                f"NLP specialist corpus must contain only NLP rows: {dict(counts)}"
            )
        if focus_domain == "nlp_rationale":
            allowed_datasets = {"ceval_rationale_train"}
        elif focus_domain == "nlp_answer_first":
            allowed_datasets = {
                "ceval_answer_first_train",
                "cmmlu_dev_answer_first_train",
            }
        elif focus_domain == "nlp_mmlu_aux":
            allowed_datasets = {"mmlu_aux_chinese"}
        else:
            allowed_datasets = {"mmlu_aux_chinese", "ceval_rationale_train"}
        invalid_datasets = sorted(
            {
                str(row.get("dataset_key", ""))
                for row in rows
                if str(row.get("dataset_key", "")) not in allowed_datasets
            }
        )
        if invalid_datasets:
            raise TrainingError(
                "NLP specialist corpus has an unapproved dataset_key: "
                f"{invalid_datasets}"
            )
        if focus_domain == "nlp_answer_first":
            for row in rows:
                answer = str(row.get("answer", "")).strip()
                first = re.match(r"^答案\s*[:：]\s*([A-D])(?:\s|$)", answer, re.I)
                final = final_answer_letter(answer)
                if (
                    row.get("answer_token_position") != "first"
                    or first is None
                    or first.group(1).upper() != final
                ):
                    raise TrainingError(
                        "Answer-first rows must repeat one locked label at the first "
                        f"and final answer markers: {row.get('sample_id')}"
                    )
    elif set(counts) != set(DOMAINS):
        raise TrainingError(f"All three domains are required: {dict(counts)}")
    return rows


def prepare_training_rows(
    rows: list[dict[str, Any]],
    focus_domain: str,
    mcq_answer_token_weight_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a compatible training view without mutating the source rows.

    The input has already passed the complete-corpus three-domain guard in
    :func:`read_rows`.  NLP focus is therefore a deterministic in-memory view
    of that same frozen train-only file, not a separately assembled corpus.
    """

    if focus_domain not in {
        "all", "math", "code", "nlp", "nlp_mcq", "nlp_rationale",
        "nlp_answer_first", "nlp_mmlu_aux", "nlp_mixed_mcq",
    }:
        raise TrainingError(f"Unsupported focus domain: {focus_domain!r}")
    multiplier = float(mcq_answer_token_weight_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise TrainingError(
            "--mcq-answer-token-weight-multiplier must be finite and >= 1"
        )
    if focus_domain == "all":
        selected = [dict(row) for row in rows]
    elif focus_domain == "math":
        selected = [dict(row) for row in rows if str(row["domain"]) == "math"]
    elif focus_domain == "code":
        selected = [dict(row) for row in rows if str(row["domain"]) == "code"]
    elif focus_domain == "nlp":
        selected = [dict(row) for row in rows if str(row["domain"]) == "nlp"]
    elif focus_domain == "nlp_mcq":
        selected = [
            dict(row)
            for row in rows
            if str(row["domain"]) == "nlp"
            and str(row.get("dataset_key", "")).casefold() in {"ceval", "c-eval"}
            and float(row.get("answer_token_weight", 1.0)) > 1.0
        ]
    elif focus_domain == "nlp_rationale":
        selected = [
            dict(row)
            for row in rows
            if str(row["domain"]) == "nlp"
            and str(row.get("dataset_key", "")) == "ceval_rationale_train"
            and float(row.get("answer_token_weight", 1.0)) > 1.0
        ]
    elif focus_domain == "nlp_answer_first":
        selected = [
            dict(row)
            for row in rows
            if str(row["domain"]) == "nlp"
            and str(row.get("dataset_key", ""))
            in {"ceval_answer_first_train", "cmmlu_dev_answer_first_train"}
            and row.get("answer_token_position") == "first"
            and float(row.get("answer_token_weight", 1.0)) > 1.0
        ]
    elif focus_domain == "nlp_mmlu_aux":
        selected = [
            dict(row)
            for row in rows
            if str(row["domain"]) == "nlp"
            and str(row.get("dataset_key", "")) == "mmlu_aux_chinese"
        ]
    else:
        selected = [
            dict(row)
            for row in rows
            if str(row["domain"]) == "nlp"
            and str(row.get("dataset_key", ""))
            in {"mmlu_aux_chinese", "ceval_rationale_train"}
        ]
    if not selected:
        raise TrainingError(f"No rows selected for focus_domain={focus_domain}")

    before_mean = sum(float(row["training_weight"]) for row in selected) / len(selected)
    normalization_divisor = 1.0
    if focus_domain in {
        "math", "code", "nlp", "nlp_mcq", "nlp_rationale", "nlp_answer_first",
        "nlp_mmlu_aux", "nlp_mixed_mcq",
    }:
        if not math.isfinite(before_mean) or before_mean <= 0:
            raise TrainingError("Cannot normalize invalid NLP training weights")
        normalization_divisor = before_mean
        for row in selected:
            row["training_weight"] = float(row["training_weight"]) / before_mean

    multiplied_rows = 0
    if multiplier > 1.0:
        for row in selected:
            existing = float(row["answer_token_weight"])
            # Only rows already declared as MCQ-answer-weighted by the frozen
            # data builder may receive the runtime multiplier.
            if existing > 1.0:
                row["answer_token_weight"] = existing * multiplier
                multiplied_rows += 1
    after_mean = sum(float(row["training_weight"]) for row in selected) / len(selected)
    stats = {
        "focus_domain": focus_domain,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "source_domain_counts": dict(
            sorted(Counter(str(row["domain"]) for row in rows).items())
        ),
        "selected_domain_counts": dict(
            sorted(Counter(str(row["domain"]) for row in selected).items())
        ),
        "selected_dataset_counts": dict(
            sorted(Counter(str(row.get("dataset_key", "")) for row in selected).items())
        ),
        "training_weight_mean_before": before_mean,
        "training_weight_normalization_divisor": normalization_divisor,
        "training_weight_mean_after": after_mean,
        "mcq_answer_token_weight_multiplier": multiplier,
        "mcq_answer_token_weight_multiplied_rows": multiplied_rows,
    }
    return selected, stats


def _as_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(value) for value in token_ids]


def _find_answer_letter_token(
    tokenizer: Any,
    full_ids: list[int],
    prompt_length: int,
    expected_letter: str,
    position: str = "last",
) -> int:
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    pattern = re.compile(
        rf"(?<![A-Za-z]){re.escape(expected_letter)}(?![A-Za-z])",
        re.IGNORECASE,
    )
    if position == "first":
        indices = range(prompt_length, min(len(full_ids), prompt_length + 64))
    elif position == "last":
        lower_bound = max(prompt_length, len(full_ids) - 64)
        indices = range(len(full_ids) - 1, lower_bound - 1, -1)
    else:
        raise TrainingError(f"Unsupported answer_token_position: {position!r}")
    for index in indices:
        token_id = int(full_ids[index])
        if token_id in special_ids:
            continue
        piece = str(tokenizer.decode([token_id], skip_special_tokens=True))
        if pattern.search(piece):
            return index
    raise TrainingError(
        f"Could not locate {position} answer token {expected_letter!r} in tokenized target"
    )


def _find_math_answer_tokens(
    tokenizer: Any,
    full_ids: list[int],
    prompt_length: int,
    expected_value: str,
) -> list[int]:
    """Locate the last encoded numeric answer without relying on test references."""

    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    candidates = (
        expected_value,
        " " + expected_value,
        "#### " + expected_value,
        "\n#### " + expected_value,
    )
    matches: list[tuple[int, int]] = []
    for candidate in candidates:
        encoded = _as_list(tokenizer.encode(candidate, add_special_tokens=False))
        if not encoded:
            continue
        width = len(encoded)
        for start in range(prompt_length, len(full_ids) - width + 1):
            if full_ids[start : start + width] == encoded:
                matches.append((start, start + width))
    if not matches:
        raise TrainingError(
            f"Could not locate final Math answer {expected_value!r} in tokenized target"
        )
    # Prefer the shortest/latest span when several candidates end at the same
    # position, so only the numeric value receives the extra weight.
    start, stop = max(matches, key=lambda item: (item[1], item[0]))
    weighted = [
        index for index in range(start, stop) if int(full_ids[index]) not in special_ids
    ]
    if not weighted:
        raise TrainingError(f"Math answer {expected_value!r} contains no trainable token")
    return weighted


def render(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in row["messages"]
        if isinstance(item, dict) and item.get("role") in {"system", "user"}
    ]
    if not messages or messages[-1]["role"] != "user":
        raise TrainingError(f"Invalid prompt messages: {row.get('sample_id')}")
    answer = str(row["answer"])
    full_messages = messages + [{"role": "assistant", "content": answer}]
    prompt_ids = _as_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = _as_list(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if len(full_ids) > max_length:
        raise TrainingError(
            f"Sequence exceeds max length: {row.get('sample_id')} "
            f"{len(full_ids)} > {max_length}"
        )
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(value != -100 for value in labels):
        raise TrainingError(f"Answer was fully truncated: {row.get('sample_id')}")
    token_weights = [1.0] * len(full_ids)
    answer_weight = float(row["answer_token_weight"])
    if answer_weight > 1.0:
        if str(row["domain"]) == "nlp":
            expected = str(row.get("answer_letter") or final_answer_letter(answer) or "")
            answer_indices = [
                _find_answer_letter_token(
                    tokenizer,
                    full_ids,
                    prompt_length,
                    expected,
                    str(row.get("answer_token_position", "last")),
                )
            ]
        elif str(row["domain"]) == "math":
            expected = str(row.get("answer_value") or final_math_answer(answer) or "")
            answer_indices = _find_math_answer_tokens(
                tokenizer, full_ids, prompt_length, expected
            )
        else:
            raise TrainingError("Weighted answer is unsupported for this domain")
        if any(labels[index] == -100 for index in answer_indices):
            raise TrainingError(f"Answer token was masked: {row.get('sample_id')}")
        for answer_index in answer_indices:
            token_weights[answer_index] = answer_weight
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "token_weights": token_weights,
        "sample_weight": float(row["training_weight"]),
        "kl_weight": float(row["kl_weight"]),
        "domain_id": DOMAIN_TO_ID[str(row["domain"])],
    }


def scan_token_budget(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    maximum = 0
    maximum_sample = ""
    weighted_answer_rows = 0
    for row in rows:
        rendered = render(tokenizer, row, max_length)
        length = len(rendered["input_ids"])
        if length > maximum:
            maximum = length
            maximum_sample = str(row["sample_id"])
        if float(row["answer_token_weight"]) > 1.0:
            weighted_answer_rows += 1
    return {
        "status": "passed",
        "scanned_rows": len(rows),
        "max_seq_length": max_length,
        "maximum_observed_tokens": maximum,
        "maximum_sample_id": maximum_sample,
        "answer_token_weight_rows": weighted_answer_rows,
    }


class TokenizedRows:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return render(self.tokenizer, self.rows[index], self.max_length)


class Collator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "token_weights": [],
        }
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["token_weights"].append(feature["token_weights"] + [1.0] * padding)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "token_weights": torch.tensor(batch["token_weights"], dtype=torch.float32),
            "sample_weight": torch.tensor(
                [feature["sample_weight"] for feature in features], dtype=torch.float32
            ),
            "kl_weight": torch.tensor(
                [feature["kl_weight"] for feature in features], dtype=torch.float32
            ),
            "domain_id": torch.tensor(
                [feature["domain_id"] for feature in features], dtype=torch.long
            ),
        }


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        if not (path / "trainer_state.json").is_file():
            continue
        if not (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        ):
            continue
        candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume(value: str, output_dir: Path) -> Path | None:
    if not value:
        return None
    checkpoint = latest_checkpoint(output_dir) if value == "auto" else resolve_path(value)
    if checkpoint is None or not checkpoint.is_dir():
        raise TrainingError(f"No complete resume checkpoint found for {value!r}")
    if not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        raise TrainingError(f"Invalid checkpoint name: {display_path(checkpoint)}")
    if not (checkpoint / "trainer_state.json").is_file():
        raise TrainingError(f"Checkpoint has no trainer state: {display_path(checkpoint)}")
    return checkpoint


def resolve_init_adapter(value: str) -> Path | None:
    if not value:
        return None
    adapter = resolve_path(value)
    if not adapter.is_dir():
        raise TrainingError(f"Initial adapter directory is missing: {display_path(adapter)}")
    if not (adapter / "adapter_config.json").is_file():
        raise TrainingError(f"Initial adapter has no config: {display_path(adapter)}")
    if not any(
        (adapter / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise TrainingError(f"Initial adapter has no weights: {display_path(adapter)}")
    return adapter


def publish_adapter(checkpoint: Path, output_dir: Path) -> list[str]:
    config_path = checkpoint / "adapter_config.json"
    weights = next(
        (
            path
            for path in (
                checkpoint / "adapter_model.safetensors",
                checkpoint / "adapter_model.bin",
            )
            if path.is_file()
        ),
        None,
    )
    if not config_path.is_file() or weights is None:
        raise TrainingError(f"Checkpoint has no complete adapter: {display_path(checkpoint)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for source in (config_path, weights, checkpoint / "README.md"):
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
            published.append(source.name)
    return published


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0-A6 all-task base-preserving Student LoRA trainer."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--focus-domain",
        choices=(
            "all",
            "math",
            "code",
            "nlp",
            "nlp_mcq",
            "nlp_rationale",
            "nlp_answer_first",
            "nlp_mmlu_aux",
            "nlp_mixed_mcq",
        ),
        default="all",
        help=(
            "Train the complete corpus, its normalized NLP-only view, or the "
            "labelled train-only Chinese MCQ subset."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument(
        "--mcq-answer-token-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied only to rows whose answer_token_weight is already > 1.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=CHECKPOINT_STEPS,
        help="Save interval; must divide --max-steps exactly.",
    )
    parser.add_argument(
        "--per-device-train-batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Checkpoint path, or 'auto' for the latest complete checkpoint.",
    )
    parser.add_argument(
        "--init-adapter",
        default="",
        help=(
            "Start a new optimizer/scheduler stage from an existing LoRA adapter. "
            "This is mutually exclusive with --resume-from-checkpoint."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank_from_env = int(os.environ.get("RANK", "0"))
    is_main = rank_from_env == 0
    audit_path = resolve_path(args.audit)
    audit: dict[str, Any] = {
        "gate": "P0-A6-STUDENT-TRAIN",
        "check_version": "1.0",
        "created_by": "model_compression/train_p0a6_student.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "errors": [],
    }
    try:
        model_dir = resolve_path(args.model_dir)
        train_path = resolve_path(args.train_data)
        output_dir = resolve_path(args.output_dir)
        if args.max_steps < 1:
            raise TrainingError("--max-steps must be positive")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise TrainingError("--learning-rate must be finite and positive")
        if args.lora_rank < 1 or args.lora_rank > 64:
            raise TrainingError("--lora-rank must be in [1, 64]")
        if args.lora_alpha < 1 or args.lora_alpha > 128:
            raise TrainingError("--lora-alpha must be in [1, 128]")
        if args.checkpoint_steps < 1:
            raise TrainingError("--checkpoint-steps must be positive")
        if args.max_steps % args.checkpoint_steps != 0:
            raise TrainingError(
                f"--max-steps must be divisible by {args.checkpoint_steps} so the final "
                "pilot state is checkpointed"
            )
        source_rows = read_rows(train_path, args.focus_domain)
        rows, focus_stats = prepare_training_rows(
            source_rows,
            args.focus_domain,
            args.mcq_answer_token_weight_multiplier,
        )
        counts = Counter(str(row["domain"]) for row in rows)
        kl_by_domain = {
            domain: {
                "minimum": min(
                    float(row["kl_weight"])
                    for row in rows
                    if row["domain"] == domain
                ),
                "maximum": max(
                    float(row["kl_weight"])
                    for row in rows
                    if row["domain"] == domain
                ),
                "mean": sum(
                    float(row["kl_weight"])
                    for row in rows
                    if row["domain"] == domain
                )
                / counts[domain],
            }
            for domain in counts
        }
        answer_weight_counts = Counter(
            str(row["domain"])
            for row in rows
            if float(row["answer_token_weight"]) > 1.0
        )
        dependencies = dependency_versions()
        resume_checkpoint = resolve_resume(args.resume_from_checkpoint, output_dir)
        init_adapter = resolve_init_adapter(args.init_adapter)
        if resume_checkpoint is not None and init_adapter is not None:
            raise TrainingError(
                "--init-adapter and --resume-from-checkpoint cannot be used together"
            )
        init_adapter_files: dict[str, str] = {}
        if init_adapter is not None:
            init_config = json.loads(
                (init_adapter / "adapter_config.json").read_text(encoding="utf-8")
            )
            if int(init_config.get("r", -1)) != args.lora_rank:
                raise TrainingError("Initial adapter rank does not match --lora-rank")
            if int(init_config.get("lora_alpha", -1)) != args.lora_alpha:
                raise TrainingError("Initial adapter alpha does not match --lora-alpha")
            for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                path = init_adapter / name
                if path.is_file():
                    init_adapter_files[name] = sha256_file(path)
        existing_checkpoints = list(output_dir.glob("checkpoint-*"))
        if existing_checkpoints and resume_checkpoint is None and not args.dry_run:
            raise TrainingError(
                "Output already contains checkpoints; use "
                "--resume-from-checkpoint auto or a new output directory"
            )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_batch = (
            EXPECTED_WORLD_SIZE
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        if not args.dry_run:
            global_batch = validate_batch_layout(
                world_size,
                args.per_device_train_batch_size,
                args.gradient_accumulation_steps,
            )
        token_budget_scan: dict[str, Any] = {}
        if args.dry_run:
            if dependencies.get("transformers") == "missing":
                raise TrainingError("transformers is required for the dry-run token scan")
            if not model_dir.is_dir():
                raise TrainingError(f"Missing model directory: {display_path(model_dir)}")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
            )
            token_budget_scan = scan_token_budget(
                tokenizer, rows, MAX_SEQUENCE_LENGTH
            )
        audit.update(
            {
                "status": "dry_run_passed" if args.dry_run else "running",
                "model_dir": display_path(model_dir),
                "train_data": display_path(train_path),
                "train_data_hash": sha256_file(train_path),
                "train_rows": len(rows),
                "focus": focus_stats,
                "domain_counts": dict(sorted(counts.items())),
                "kl_weight_by_domain": kl_by_domain,
                "answer_token_weight_rows": dict(sorted(answer_weight_counts.items())),
                "formal_test_reference_count": 0,
                "optimization": {
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": LORA_DROPOUT,
                    "target_modules": TARGET_MODULES,
                    "learning_rate": args.learning_rate,
                    "bf16": True,
                    "max_seq_length": MAX_SEQUENCE_LENGTH,
                    "max_steps": args.max_steps,
                    "checkpoint_steps": args.checkpoint_steps,
                    "save_total_limit": None,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "expected_world_size": EXPECTED_WORLD_SIZE,
                    "observed_world_size": world_size,
                    "effective_global_batch_size": global_batch,
                    "warmup_ratio": WARMUP_RATIO,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_use_reentrant": False,
                    "resume_from_checkpoint": (
                        display_path(resume_checkpoint) if resume_checkpoint else ""
                    ),
                    "init_adapter": (
                        display_path(init_adapter) if init_adapter else ""
                    ),
                    "init_adapter_file_hashes": init_adapter_files,
                },
                "preservation": {
                    "method": "all_task_adapter_disabled_base_kl",
                    "direction": "KL(base||student)",
                    "temperature": KL_TEMPERATURE,
                    "mask": "assistant_answer_tokens_only",
                    "reference_forward": "unconditional_on_every_rank_and_batch",
                    "supervised_weight": "1-kl_weight_per_row",
                },
                "dependencies": dependencies,
                "token_budget_scan": token_budget_scan,
            }
        )
        if is_main:
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str)
            )
            atomic_write_json(audit_path, audit)
        if args.dry_run:
            print(f"P0-A6 Student dry-run passed: {display_path(audit_path)}")
            return 0
        missing = [
            module
            for module in ("torch", "transformers", "accelerate", "peft")
            if dependencies.get(module) == "missing"
        ]
        if missing:
            raise TrainingError(f"Missing training dependencies: {missing}")
        if not model_dir.is_dir():
            raise TrainingError(f"Missing model directory: {display_path(model_dir)}")

        import torch
        import torch.distributed as distributed
        import torch.nn.functional as functional
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        class AllTaskKLTrainer(Trainer):
            """Trainer with unconditional base reference and domain metrics."""

            def __init__(self, *trainer_args: Any, **trainer_kwargs: Any):
                super().__init__(*trainer_args, **trainer_kwargs)
                self._domain_totals = torch.zeros(3, 4, dtype=torch.float64)
                self._domain_window = torch.zeros(3, 4, dtype=torch.float64)
                self.domain_log_history: list[dict[str, Any]] = []

            @staticmethod
            def _per_example_mean(
                token_values: Any,
                batch_indices: Any,
                token_weights: Any,
                batch_size: int,
            ) -> Any:
                numerator = torch.zeros(
                    batch_size, device=token_values.device, dtype=token_values.dtype
                )
                denominator = torch.zeros_like(numerator)
                numerator.scatter_add_(0, batch_indices, token_values * token_weights)
                denominator.scatter_add_(0, batch_indices, token_weights)
                return numerator / denominator.clamp_min(1.0)

            def _record_domain_metrics(
                self,
                domain_ids: Any,
                supervised: Any,
                kl: Any,
                combined: Any,
            ) -> None:
                metrics = torch.zeros(3, 4, device=supervised.device, dtype=torch.float64)
                for domain_id in range(3):
                    mask = domain_ids.eq(domain_id)
                    metrics[domain_id, 0] = supervised[mask].double().sum()
                    metrics[domain_id, 1] = kl[mask].double().sum()
                    metrics[domain_id, 2] = combined[mask].double().sum()
                    metrics[domain_id, 3] = mask.double().sum()
                # This collective is unconditional.  Every rank enters it once
                # per compute_loss call, independent of its local task mix.
                if distributed.is_available() and distributed.is_initialized():
                    distributed.all_reduce(metrics, op=distributed.ReduceOp.SUM)
                metrics = metrics.detach().cpu()
                self._domain_totals += metrics
                self._domain_window += metrics

            def domain_summary(self) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for domain_id, domain in enumerate(DOMAINS):
                    count = float(self._domain_totals[domain_id, 3].item())
                    result[domain] = {
                        "examples_seen": int(count),
                        "mean_sft_loss": (
                            float(self._domain_totals[domain_id, 0].item() / count)
                            if count
                            else None
                        ),
                        "mean_kl_base_to_student": (
                            float(self._domain_totals[domain_id, 1].item() / count)
                            if count
                            else None
                        ),
                        "mean_combined_loss": (
                            float(self._domain_totals[domain_id, 2].item() / count)
                            if count
                            else None
                        ),
                    }
                return result

            def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
                if "loss" in logs:
                    entry: dict[str, Any] = {"step": int(self.state.global_step)}
                    for domain_id, domain in enumerate(DOMAINS):
                        count = float(self._domain_window[domain_id, 3].item())
                        if count:
                            for column, suffix in (
                                (0, "sft_loss"),
                                (1, "kl_base_to_student"),
                                (2, "combined_loss"),
                            ):
                                value = float(
                                    self._domain_window[domain_id, column].item() / count
                                )
                                logs[f"domain/{domain}_{suffix}"] = value
                                entry[f"{domain}_{suffix}"] = value
                            entry[f"{domain}_examples"] = int(count)
                    if len(entry) > 1:
                        self.domain_log_history.append(entry)
                    self._domain_window.zero_()
                try:
                    super().log(logs, start_time=start_time)
                except TypeError:
                    # Compatibility with older Transformers releases.
                    super().log(logs)

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                del num_items_in_batch
                sample_weights = inputs.pop("sample_weight")
                kl_weights = inputs.pop("kl_weight")
                domain_ids = inputs.pop("domain_id")
                token_weights = inputs.pop("token_weights")
                labels = inputs.pop("labels")

                # Student forward goes through DDP so LoRA gradients synchronize.
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_token_weights = token_weights[..., 1:].contiguous()
                valid = shift_labels.ne(-100)
                if not bool(valid.any()):
                    raise TrainingError("Batch has no assistant answer tokens")
                batch_size = int(shift_labels.size(0))
                batch_indices = (
                    torch.arange(batch_size, device=shift_labels.device)
                    .unsqueeze(1)
                    .expand_as(shift_labels)[valid]
                )
                valid_student_logits = shift_logits[valid]
                valid_labels = shift_labels[valid]
                valid_token_weights = shift_token_weights[valid].to(
                    device=shift_logits.device, dtype=torch.float32
                )
                sft_tokens = functional.cross_entropy(
                    valid_student_logits.float(), valid_labels, reduction="none"
                )
                supervised = self._per_example_mean(
                    sft_tokens,
                    batch_indices,
                    valid_token_weights,
                    batch_size,
                )

                peft_model = model.module if hasattr(model, "module") else model
                if not hasattr(peft_model, "disable_adapter"):
                    raise TrainingError(
                        "All-task KL requires a PEFT model with disable_adapter"
                    )
                # IMPORTANT: this reference forward is deliberately unconditional.
                # Even a row with kl_weight=0 runs it, keeping all DDP ranks aligned.
                with torch.no_grad(), peft_model.disable_adapter():
                    reference_logits = peft_model(**inputs).logits[..., :-1, :]
                valid_reference_logits = reference_logits[valid]
                temperature = KL_TEMPERATURE
                student_log_probs = functional.log_softmax(
                    valid_student_logits.float() / temperature, dim=-1
                )
                base_probs = functional.softmax(
                    valid_reference_logits.float() / temperature, dim=-1
                )
                kl_tokens = functional.kl_div(
                    student_log_probs,
                    base_probs,
                    reduction="none",
                ).sum(dim=-1) * (temperature * temperature)
                unit_token_weights = torch.ones_like(kl_tokens)
                kl = self._per_example_mean(
                    kl_tokens,
                    batch_indices,
                    unit_token_weights,
                    batch_size,
                )
                kl_weights = kl_weights.to(device=kl.device, dtype=kl.dtype)
                combined = (1.0 - kl_weights) * supervised + kl_weights * kl
                self._record_domain_metrics(domain_ids, supervised, kl, combined)
                sample_weights = sample_weights.to(
                    device=combined.device, dtype=combined.dtype
                )
                loss = (combined * sample_weights).mean()
                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=args.max_steps,
            num_train_epochs=1.0,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            max_grad_norm=MAX_GRAD_NORM,
            bf16=True,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.checkpoint_steps,
            save_total_limit=None,
            eval_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            restore_callback_states_from_checkpoint=True,
            seed=SEED,
            data_seed=SEED,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if init_adapter is not None:
            model = PeftModel.from_pretrained(
                model,
                str(init_adapter),
                is_trainable=True,
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=LORA_DROPOUT,
                    target_modules=list(TARGET_MODULES),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        trainer = AllTaskKLTrainer(
            model=model,
            args=training_args,
            train_dataset=TokenizedRows(rows, tokenizer, MAX_SEQUENCE_LENGTH),
            data_collator=Collator(tokenizer),
        )
        result = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )
        is_main = trainer.is_world_process_zero()
        if is_main:
            checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
            if not checkpoint.is_dir():
                checkpoint = latest_checkpoint(output_dir) or Path()
            if not checkpoint.is_dir():
                raise TrainingError("Training produced no complete checkpoint")
            published = publish_adapter(checkpoint, output_dir)
            tokenizer.save_pretrained(output_dir)
            audit.update(
                {
                    "status": "passed",
                    "global_step": int(trainer.state.global_step),
                    "final_checkpoint": display_path(checkpoint),
                    "published_files": published,
                    "train_metrics": result.metrics,
                    "domain_loss_summary": trainer.domain_summary(),
                    "domain_loss_log": trainer.domain_log_history,
                    "retained_checkpoints": [
                        display_path(path)
                        for path in sorted(
                            output_dir.glob("checkpoint-*"),
                            key=lambda item: int(item.name.rsplit("-", 1)[-1]),
                        )
                        if path.is_dir()
                    ],
                }
            )
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str)
            )
            atomic_write_json(audit_path, audit)
            print(f"P0-A6 Student pilot completed: {display_path(output_dir)}")
            print(f"Audit: {display_path(audit_path)}")
        return 0
    except (
        TrainingError,
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        if is_main:
            audit["status"] = "failed"
            audit.setdefault("errors", []).append(str(exc))
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str)
            )
            atomic_write_json(audit_path, audit)
            print(f"P0-A6 Student training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
