from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_utils import (
    apply_lora_to_model,
    apply_residual_lora_to_model,
    disable_outer_lora,
    load_lora_adapter,
    load_lora_state,
    lora_state_dict,
    save_lora_adapter,
    trainable_parameters,
)
from train_cedd_structured import (
    SYSTEM_PROMPT as STRUCTURED_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE as STRUCTURED_USER_PROMPT_TEMPLATE,
    cleanup_distributed,
    collate_batch,
    display_path,
    is_rank_zero,
    load_jsonl,
    parse_csv_values,
    resolve_path,
    setup_distributed,
    sha256_dir,
    sha256_file,
    sha256_text,
    target_text,
    user_prompt,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_chapter2_capability import (  # noqa: E402
    SYSTEM_PROMPT as CAPABILITY_SYSTEM_PROMPT,
    build_messages as build_capability_messages,
    extract_choice,
    extract_gsm8k_prediction,
    extract_gsm8k_reference,
    load_cmmlu_sample,
    load_gsm8k_sample,
    load_humaneval_sample,
)
from generate_teacher_capability_distill import validate_code_row  # noqa: E402


DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "adapters" / "p0a2_deepseek_recovery"
DEFAULT_DISTILL = ROOT / "data" / "distill" / "distill_dataset.jsonl"
DEFAULT_REPAIR = ROOT / "data" / "distill" / "counterfactual_repair_trace.jsonl"
DEFAULT_CAPABILITY_REHEARSAL = ROOT / "data" / "distill" / "capability_rehearsal_v3.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a2_deepseek_recovery_train.json"
SPLITS = ROOT / "data" / "splits"

REPAIR_SYSTEM_PROMPT = (
    "You are DB4AI-EdgeServe. Return exactly one compact JSON object that repairs the edge decision. "
    "No markdown, no prose outside JSON."
)
REPAIR_USER_PROMPT_TEMPLATE = """Repair the edge decision using the teacher boundary signal.

Return this exact JSON schema:
{
  "object_state": "short observable state",
  "event_type": "math_reasoning|knowledge_choice|industrial_normal|surface_defect|traffic_camera",
  "risk_attr": "low|medium|high",
  "action": "pass|inspect|alert|upload",
  "confidence": 0.0,
  "review_intent": "none|verify_reasoning|inspect_quality|sync_tracking",
  "short_rationale": "one short sentence",
  "evidence_items": ["1-3 short evidence strings"]
}

Boundary signal:
__BOUNDARY_PROMPT__

Task context:
task_type: __TASK_TYPE__
sample_id: __SAMPLE_ID__
"""


class RepairTrainError(RuntimeError):
    pass


def read_split_ids(dataset_key: str, split: str = "train") -> list[str]:
    path = SPLITS / f"{dataset_key}_{split}.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repeat_rows(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    if repeat <= 0:
        return []
    return [row for _ in range(repeat) for row in rows]


def limit_balanced(rows: list[dict[str, Any]], limit: int, seed: int, key: str = "dataset_key") -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(str(row.get(key, "")), []).append(row)
    rng = random.Random(seed)
    for bucket in by_key.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    positions = {bucket_key: 0 for bucket_key in by_key}
    keys = sorted(by_key)
    while len(selected) < limit:
        progressed = False
        for bucket_key in keys:
            pos = positions[bucket_key]
            if pos < len(by_key[bucket_key]):
                selected.append(by_key[bucket_key][pos])
                positions[bucket_key] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def parse_int_map(items: list[str], option_name: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in parse_csv_values(items):
        if "=" not in item:
            raise RepairTrainError(f"Invalid {option_name} entry, expected key=value: {item}")
        key, value_text = item.split("=", 1)
        key = key.strip()
        if not key:
            raise RepairTrainError(f"Invalid {option_name} entry with empty key: {item}")
        try:
            value = int(value_text)
        except ValueError as exc:
            raise RepairTrainError(f"Invalid {option_name} integer value: {item}") from exc
        if value < 0:
            raise RepairTrainError(f"{option_name} value must be >= 0: {item}")
        parsed[key] = value
    return parsed


def apply_source_limits(examples: list[dict[str, Any]], source_limits: dict[str, int], seed: int) -> list[dict[str, Any]]:
    if not source_limits:
        return examples
    rng = random.Random(seed)
    limited: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        by_source.setdefault(str(row.get("source", "")), []).append(row)
    for source, rows in sorted(by_source.items()):
        limit = source_limits.get(source)
        if limit is None or limit <= 0 or len(rows) <= limit:
            limited.extend(rows)
            continue
        rows = list(rows)
        rng.shuffle(rows)
        limited.extend(rows[:limit])
    return limited


def repair_target_text(row: dict[str, Any]) -> str:
    target = row.get("target_json", {})
    decision = dict(target.get("decision_tuple", row.get("teacher_decision_tuple", {})))
    if "short_rationale" not in decision and target.get("short_rationale"):
        decision["short_rationale"] = target["short_rationale"]
    decision.setdefault("evidence_items", [])
    return json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def repair_user_prompt(row: dict[str, Any]) -> str:
    return (
        REPAIR_USER_PROMPT_TEMPLATE.replace("__BOUNDARY_PROMPT__", str(row.get("boundary_prompt", "")))
        .replace("__TASK_TYPE__", str(row.get("task_type", "")))
        .replace("__SAMPLE_ID__", str(row.get("sample_id", "")))
    )


def load_structured_examples(
    path: Path,
    dataset_filter: set[str] | None,
    sample_limit: int,
    seed: int,
    repeat: int,
) -> list[dict[str, Any]]:
    if repeat <= 0:
        return []
    rows = [
        {
            "source": "structured_distill",
            "dataset_key": row.get("dataset_key", ""),
            "sample_id": row.get("sample_id", ""),
            "messages": [
                {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(row)},
            ],
            "answer": target_text(row),
        }
        for row in load_jsonl(path)
        if row.get("used_for_training") is True
        and (dataset_filter is None or str(row.get("dataset_key", "")) in dataset_filter)
    ]
    rows = limit_balanced(rows, sample_limit, seed)
    return repeat_rows(rows, repeat)


def load_repair_examples(
    path: Path,
    dataset_filter: set[str] | None,
    sample_limit: int,
    seed: int,
    repeat: int,
) -> list[dict[str, Any]]:
    if repeat <= 0:
        return []
    rows = [
        {
            "source": "counterfactual_repair",
            "dataset_key": row.get("dataset_key", ""),
            "sample_id": row.get("sample_id", ""),
            "messages": [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": repair_user_prompt(row)},
            ],
            "answer": repair_target_text(row),
        }
        for row in load_jsonl(path)
        if row.get("used_for_training") is True
        and (dataset_filter is None or str(row.get("dataset_key", "")) in dataset_filter)
    ]
    rows = limit_balanced(rows, sample_limit, seed)
    return repeat_rows(rows, repeat)


def load_capability_sample(sample_id: str) -> dict[str, Any]:
    dataset_key = sample_id.split("/", 1)[0]
    if dataset_key == "gsm8k":
        sample = load_gsm8k_sample(sample_id)
        answer = str(sample.get("answer", "")).strip()
    elif dataset_key == "cmmlu":
        sample = load_cmmlu_sample(sample_id)
        answer = str(sample.get("reference", "")).strip()
    elif dataset_key == "humaneval":
        sample = load_humaneval_sample(sample_id)
        answer = str(sample.get("canonical_solution", "")).strip()
    else:
        raise RepairTrainError(f"Unsupported capability dataset: {dataset_key}")
    messages, _ = build_capability_messages(sample)
    return {
        "source": "capability_rehearsal",
        "dataset_key": dataset_key,
        "sample_id": sample_id,
        "messages": messages,
        "answer": answer,
    }


def load_capability_examples(
    datasets: set[str],
    sample_limit: int,
    seed: int,
    repeat: int,
) -> list[dict[str, Any]]:
    if repeat <= 0 or not datasets:
        return []
    sample_ids: list[str] = []
    for dataset_key in sorted(datasets):
        sample_ids.extend(read_split_ids(dataset_key, "train"))
    examples = [load_capability_sample(sample_id) for sample_id in sample_ids]
    examples = [row for row in examples if row.get("answer")]
    examples = limit_balanced(examples, sample_limit, seed)
    return repeat_rows(examples, repeat)


def load_capability_rehearsal_jsonl(path: Path, sample_limit: int, seed: int, repeat: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for row in load_jsonl(path):
        if row.get("used_for_training") is not True:
            continue
        messages = row.get("messages", [])
        answer = row.get("answer", "")
        if not isinstance(messages, list) or not answer:
            continue
        copied = {
            "source": str(row.get("source", "capability_rehearsal_jsonl")),
            "dataset_key": str(row.get("dataset_key", "")),
            "sample_id": str(row.get("sample_id", "")),
            "validation_group_id": str(row.get("validation_group_id", "")),
            "messages": messages,
            "answer": str(answer),
        }
        if isinstance(row.get("code_eval"), dict):
            copied["code_eval"] = row["code_eval"]
        rows.append(copied)
    rows = limit_balanced(rows, sample_limit, seed)
    return repeat_rows(rows, repeat)


def load_generation_validation_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RepairTrainError(f"Missing generation validation JSONL: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        if row.get("used_for_training") is not False:
            raise RepairTrainError(
                f"External generation validation row must declare used_for_training=false: "
                f"{row.get('sample_id', '<missing>')}"
            )
        messages = row.get("messages")
        answer = str(row.get("answer", ""))
        if not isinstance(messages, list) or not messages or not answer:
            raise RepairTrainError(
                f"Incomplete external generation validation row: {row.get('sample_id', '<missing>')}"
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
        raise RepairTrainError(f"No external generation validation rows: {display_path(path)}")
    return rows


def render_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    close_reasoning_prefix: bool,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if close_reasoning_prefix and prompt.rstrip().endswith("<think>"):
        prompt += "</think>\n"
    return prompt


class MixedInstructionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        close_reasoning_prefix: bool = False,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.close_reasoning_prefix = close_reasoning_prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt_text = render_generation_prompt(
            self.tokenizer,
            example["messages"],
            self.close_reasoning_prefix,
        )
        answer_text = str(example["answer"]).rstrip() + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def stable_validation_fraction(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def validation_group_id(row: dict[str, Any]) -> str:
    declared_group = str(row.get("validation_group_id", "")).strip()
    if declared_group:
        return f"declared:{row.get('dataset_key', '')}:{declared_group}"
    messages = row.get("messages")
    if isinstance(messages, list) and "answer" in row:
        payload = {
            "dataset_key": row.get("dataset_key", ""),
            "messages": messages,
            "answer": row.get("answer", ""),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"semantic:{sha256_text(canonical)}"
    return f"sample:{row.get('dataset_key', '')}:{row.get('sample_id', '')}"


def split_grouped_validation(
    examples: list[dict[str, Any]],
    validation_fraction: float,
    validation_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_fraction <= 0:
        return list(examples), []
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for row in examples:
        group_id = validation_group_id(row)
        target = validation_rows if stable_validation_fraction(group_id, validation_seed) < validation_fraction else train_rows
        target.append(row)
    if not train_rows or not validation_rows:
        raise RepairTrainError(
            "Grouped validation split produced an empty partition; adjust --validation-fraction or selected data."
        )
    return train_rows, validation_rows


def cap_examples_per_validation_group(
    examples: list[dict[str, Any]],
    max_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return list(examples)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        grouped.setdefault(validation_group_id(row), []).append(row)
    selected: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: sha256_text(
                f"{seed}:{group_id}:{row.get('dataset_key', '')}:{row.get('sample_id', '')}:"
                f"{row.get('answer', '')}"
            ),
        )
        selected.extend(ordered[:max_examples])
    return selected


def sample_ids_hash(examples: list[dict[str, Any]]) -> str:
    values = sorted({f"{row.get('dataset_key', '')}:{row.get('sample_id', '')}" for row in examples})
    return sha256_text("\n".join(values) + "\n")


def validation_group_ids(examples: list[dict[str, Any]]) -> set[str]:
    return {validation_group_id(row) for row in examples}


def validation_group_ids_hash(examples: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(sorted(validation_group_ids(examples))) + "\n")


def evaluate_validation_loss(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    close_reasoning_prefix: bool = False,
) -> float:
    local_examples = examples[rank::world_size]
    dataset = MixedInstructionDataset(
        local_examples, tokenizer, max_length, close_reasoning_prefix
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    model.eval()
    loss_sum = 0.0
    loss_count = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss_sum += float(output.loss.detach().cpu())
            loss_count += 1.0
    stats = torch.tensor([loss_sum, loss_count], dtype=torch.float64, device=device)
    if world_size > 1:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    model.train()
    return float((stats[0] / stats[1]).detach().cpu()) if stats[1].item() else float("inf")


def select_generation_validation_examples(
    examples: list[dict[str, Any]],
    limit: int,
    seed: int,
    examples_per_group: int = 1,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        by_group.setdefault(validation_group_id(row), []).append(row)
    ordered_group_ids = sorted(
        by_group,
        key=lambda group_id: sha256_text(f"{seed}:{group_id}"),
    )
    selected: list[dict[str, Any]] = []
    for example_index in range(examples_per_group):
        for group_id in ordered_group_ids:
            rows = by_group[group_id]
            if example_index >= len(rows):
                continue
            selected.append(rows[example_index])
            if limit > 0 and len(selected) >= limit:
                return selected
    return selected


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


def evaluate_generation_validation(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_new_tokens: int,
    code_timeout_sec: float,
    device: torch.device,
    rank: int,
    world_size: int,
    close_reasoning_prefix: bool = False,
) -> tuple[float, int]:
    local_examples = examples[rank::world_size]
    datasets = ("gsm8k", "humaneval", "cmmlu")
    dataset_to_index = {dataset: index for index, dataset in enumerate(datasets)}
    model.eval()
    score_sums = [0.0] * len(datasets)
    counts = [0] * len(datasets)
    with torch.inference_mode():
        for example in local_examples:
            prompt = render_generation_prompt(
                tokenizer,
                example["messages"],
                close_reasoning_prefix,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_length = int(inputs["input_ids"].shape[1])
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(generated[0, input_length:], skip_special_tokens=True).strip()
            dataset_index = dataset_to_index.get(str(example.get("dataset_key", "")))
            if dataset_index is None:
                continue
            score_sums[dataset_index] += score_generation_validation(example, response, code_timeout_sec)
            counts[dataset_index] += 1
    stats = torch.tensor(
        score_sums + [float(count) for count in counts], dtype=torch.float64, device=device
    )
    if world_size > 1:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    model.train()
    global_score_sums = stats[: len(datasets)]
    global_counts = stats[len(datasets) :]
    valid = global_counts.gt(0)
    global_count = int(global_counts.sum().detach().cpu())
    score = (
        float((global_score_sums[valid] / global_counts[valid]).mean().detach().cpu())
        if valid.any()
        else 0.0
    )
    return score, global_count


def parent_correct_token_mask(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad(), disable_outer_lora(model):
        parent_logits = model(**batch).logits[:, :-1]
        parent_predictions = parent_logits.argmax(dim=-1)
    target_tokens = batch["labels"][:, 1:]
    return target_tokens.ne(-100) & parent_predictions.eq(target_tokens)


def learning_rate_multiplier(step: int, total_steps: int, warmup_ratio: float, scheduler_name: str) -> float:
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    if scheduler_name == "constant":
        return 1.0
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def planned_optimizer_steps(
    sample_count: int,
    world_size: int,
    batch_size: int,
    grad_accum_steps: int,
    epochs: float,
    max_steps: int,
) -> int:
    samples_per_rank = (sample_count + world_size - 1) // world_size
    batches_per_rank = (samples_per_rank + batch_size - 1) // batch_size
    steps = int((batches_per_rank * epochs) / grad_accum_steps)
    if max_steps > 0:
        steps = min(steps or max_steps, max_steps)
    return max(steps, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CEDD-Repair LoRA adapter with repair and capability rehearsal.")
    parser.add_argument("--student-init", "--student_init", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--init-adapter", "--init_adapter", default="none")
    parser.add_argument("--distill-data", "--distill_data", default=str(DEFAULT_DISTILL))
    parser.add_argument("--repair-trace", "--repair_trace", default=str(DEFAULT_REPAIR))
    parser.add_argument(
        "--capability-rehearsal-jsonl",
        "--capability_rehearsal_jsonl",
        action="append",
        default=None,
        help="Extra non-final capability rehearsal JSONL files built by build_capability_rehearsal.py.",
    )
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--stage-name", "--stage_name", default="cedd_repair_v3")
    parser.add_argument(
        "--frozen-base-adapter",
        "--frozen_base_adapter",
        default="none",
        help="Keep this adapter frozen and train a fresh parallel residual LoRA.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Structured distill dataset filter.")
    parser.add_argument("--repair-dataset", "--repair_dataset", action="append", default=[], help="Repair trace dataset filter.")
    parser.add_argument("--capability-dataset", action="append", default=["gsm8k", "humaneval", "cmmlu"])
    parser.add_argument("--distill-sample-limit", "--distill_sample_limit", type=int, default=0)
    parser.add_argument("--repair-sample-limit", "--repair_sample_limit", type=int, default=0)
    parser.add_argument("--capability-sample-limit", "--capability_sample_limit", type=int, default=2048)
    parser.add_argument("--distill-repeat", "--distill_repeat", type=int, default=1)
    parser.add_argument("--repair-repeat", "--repair_repeat", type=int, default=4)
    parser.add_argument("--capability-repeat", "--capability_repeat", type=int, default=1)
    parser.add_argument("--capability-jsonl-repeat", "--capability_jsonl_repeat", type=int, default=1)
    parser.add_argument(
        "--source-limit",
        "--source_limit",
        action="append",
        default=[],
        help="Cap selected examples by source after loading, e.g. gsm8k_train_final_only=512. Repeat or comma-separate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--parent-preservation-weight",
        type=float,
        default=0.0,
        help="Extra CE weight on response tokens the frozen parent predicts correctly.",
    )
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--validation-seed", type=int, default=20260713)
    parser.add_argument(
        "--max-train-examples-per-validation-group",
        type=int,
        default=0,
        help="Cap training rows per semantic/task family after the grouped validation split.",
    )
    parser.add_argument(
        "--min-validation-group-count",
        type=int,
        default=0,
        help="Fail before loading the model when too few held-out semantic/task families were selected.",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-validation-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--min-validation-relative-improvement",
        type=float,
        default=0.0,
        help="Reject the trained residual and restore step 0 unless validation loss improves by this fraction.",
    )
    parser.add_argument("--restore-best-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--validation-selection-metric",
        choices=["loss", "generation"],
        default="loss",
    )
    parser.add_argument("--generation-validation-sample-limit", type=int, default=0)
    parser.add_argument(
        "--generation-validation-jsonl",
        action="append",
        default=[],
        help=(
            "Independent used_for_training=false JSONL used only for generation checkpoint selection. "
            "Repeat or comma-separate; when supplied it replaces the internal validation rows for generation scoring."
        ),
    )
    parser.add_argument(
        "--generation-validation-examples-per-group",
        type=int,
        default=1,
        help="Evaluate up to this many rows per held-out semantic group in balanced rounds.",
    )
    parser.add_argument("--generation-validation-max-new-tokens", type=int, default=192)
    parser.add_argument("--generation-validation-code-timeout-sec", type=float, default=5.0)
    parser.add_argument("--min-generation-validation-improvement", type=float, default=0.0)
    parser.add_argument(
        "--min-optimizer-steps",
        type=int,
        default=0,
        help="Fail before model loading when the selected data/epoch/batch schedule yields fewer updates.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-module", action="append", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--close-reasoning-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Close a DeepSeek-style trailing <think> prefix before targets and generation.",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ["distill_sample_limit", "repair_sample_limit", "capability_sample_limit"]:
        if getattr(args, name) < 0:
            print(f"--{name.replace('_', '-')} must be >= 0", file=sys.stderr)
            return 2
    if args.capability_jsonl_repeat < 0:
        print("--capability-jsonl-repeat must be >= 0", file=sys.stderr)
        return 2
    if args.parent_preservation_weight < 0:
        print("--parent-preservation-weight must be >= 0", file=sys.stderr)
        return 2
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        print("--batch-size and --grad-accum-steps must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.validation_fraction < 1:
        print("--validation-fraction must be in [0, 1)", file=sys.stderr)
        return 2
    if not 0 <= args.warmup_ratio < 1:
        print("--warmup-ratio must be in [0, 1)", file=sys.stderr)
        return 2
    if (
        args.eval_every < 0
        or args.early_stopping_patience < 0
        or args.min_validation_improvement < 0
        or args.min_validation_relative_improvement < 0
        or args.generation_validation_sample_limit < 0
        or args.generation_validation_examples_per_group <= 0
        or args.generation_validation_max_new_tokens <= 0
        or args.generation_validation_code_timeout_sec <= 0
        or args.min_generation_validation_improvement < 0
        or args.max_train_examples_per_validation_group < 0
        or args.min_validation_group_count < 0
        or args.min_optimizer_steps < 0
    ):
        print("Validation scheduling values must be >= 0", file=sys.stderr)
        return 2
    if args.validation_selection_metric == "generation" and (
        args.validation_fraction <= 0
        or not args.restore_best_validation
        or args.generation_validation_sample_limit <= 0
    ):
        print(
            "generation validation selection requires validation, --restore-best-validation, "
            "and --generation-validation-sample-limit > 0",
            file=sys.stderr,
        )
        return 2
    if args.min_validation_relative_improvement > 0 and (
        args.validation_fraction <= 0 or not args.restore_best_validation
    ):
        print(
            "--min-validation-relative-improvement requires validation and --restore-best-validation",
            file=sys.stderr,
        )
        return 2

    try:
        source_limits = parse_int_map(args.source_limit, "--source-limit")
    except RepairTrainError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rank, local_rank, world_size, device = setup_distributed(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = resolve_path(args.student_init)
    init_adapter_dir = None if str(args.init_adapter).strip().lower() in {"", "none", "null"} else resolve_path(args.init_adapter)
    frozen_base_adapter_dir = (
        None
        if str(args.frozen_base_adapter).strip().lower() in {"", "none", "null"}
        else resolve_path(args.frozen_base_adapter)
    )
    if init_adapter_dir is not None and frozen_base_adapter_dir is not None:
        print("--init-adapter and --frozen-base-adapter are mutually exclusive", file=sys.stderr)
        cleanup_distributed(world_size)
        return 2
    parent_reference_kind = (
        "frozen_adapter"
        if frozen_base_adapter_dir is not None
        else "base_model"
        if args.parent_preservation_weight > 0
        else "none"
    )
    distill_path = resolve_path(args.distill_data)
    repair_path = resolve_path(args.repair_trace)
    repair_trace_hash = sha256_file(repair_path) if args.repair_repeat > 0 else ""
    capability_rehearsal_args = (
        args.capability_rehearsal_jsonl
        if args.capability_rehearsal_jsonl is not None
        else [str(DEFAULT_CAPABILITY_REHEARSAL)]
    )
    capability_rehearsal_paths = [resolve_path(path) for path in capability_rehearsal_args]
    generation_validation_paths = [
        resolve_path(path) for path in parse_csv_values(args.generation_validation_jsonl)
    ]
    output_dir = resolve_path(args.output_dir)
    audit_path = resolve_path(args.audit)
    dataset_filter = set(parse_csv_values(args.dataset)) if args.dataset else None
    repair_dataset_filter = set(parse_csv_values(args.repair_dataset)) if args.repair_dataset else None
    capability_datasets = set(parse_csv_values(args.capability_dataset)) if args.capability_dataset else set()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    examples = []
    examples.extend(
        load_structured_examples(
            distill_path,
            dataset_filter,
            args.distill_sample_limit,
            args.seed,
            args.distill_repeat,
        )
    )
    examples.extend(
        load_repair_examples(
            repair_path,
            repair_dataset_filter,
            args.repair_sample_limit,
            args.seed,
            args.repair_repeat,
        )
    )
    examples.extend(load_capability_examples(capability_datasets, args.capability_sample_limit, args.seed, args.capability_repeat))
    for path in capability_rehearsal_paths:
        examples.extend(
            load_capability_rehearsal_jsonl(
                path,
                args.capability_sample_limit,
                args.seed,
                args.capability_jsonl_repeat,
            )
        )
    source_counts_before_limits = Counter(str(row.get("source", "")) for row in examples)
    dataset_counts_before_limits = Counter(str(row.get("dataset_key", "")) for row in examples)
    examples = apply_source_limits(examples, source_limits, args.seed)
    if not examples:
        print("No repair training examples selected.", file=sys.stderr)
        cleanup_distributed(world_size)
        return 1
    selected_examples = list(examples)
    try:
        examples, validation_examples = split_grouped_validation(
            selected_examples,
            args.validation_fraction,
            args.validation_seed,
        )
    except RepairTrainError as exc:
        print(str(exc), file=sys.stderr)
        cleanup_distributed(world_size)
        return 2
    validation_group_count = len(validation_group_ids(validation_examples))
    if validation_group_count < args.min_validation_group_count:
        print(
            "Grouped validation split is too small: "
            f"groups={validation_group_count} required={args.min_validation_group_count}",
            file=sys.stderr,
        )
        cleanup_distributed(world_size)
        return 2
    training_sample_count_before_group_cap = len(examples)
    examples = cap_examples_per_validation_group(
        examples,
        args.max_train_examples_per_validation_group,
        args.seed,
    )
    if not examples:
        print("No training examples remain after the validation-group cap.", file=sys.stderr)
        cleanup_distributed(world_size)
        return 2
    random.Random(args.seed).shuffle(examples)

    selected_ids_hash = sample_ids_hash(selected_examples)
    training_ids_hash = sample_ids_hash(examples)
    validation_ids_hash = sample_ids_hash(validation_examples) if validation_examples else ""
    selected_group_ids_hash = validation_group_ids_hash(selected_examples)
    training_group_ids_hash = validation_group_ids_hash(examples)
    validation_group_ids_hash_value = validation_group_ids_hash(validation_examples) if validation_examples else ""
    validation_group_overlap_count = len(validation_group_ids(examples) & validation_group_ids(validation_examples))
    if validation_group_overlap_count:
        print(
            f"Training/validation semantic-group overlap detected: {validation_group_overlap_count}",
            file=sys.stderr,
        )
        cleanup_distributed(world_size)
        return 2
    source_counts = Counter(str(row.get("source", "")) for row in examples)
    dataset_counts = Counter(str(row.get("dataset_key", "")) for row in examples)
    validation_source_counts = Counter(str(row.get("source", "")) for row in validation_examples)
    validation_dataset_counts = Counter(str(row.get("dataset_key", "")) for row in validation_examples)
    try:
        external_generation_validation_examples = [
            row
            for path in generation_validation_paths
            for row in load_generation_validation_jsonl(path)
        ]
    except RepairTrainError as exc:
        print(str(exc), file=sys.stderr)
        cleanup_distributed(world_size)
        return 2
    external_sample_keys = [
        f"{row.get('dataset_key', '')}:{row.get('sample_id', '')}"
        for row in external_generation_validation_examples
    ]
    external_duplicate_sample_count = len(external_sample_keys) - len(set(external_sample_keys))
    external_selected_group_overlap_count = len(
        validation_group_ids(selected_examples) & validation_group_ids(external_generation_validation_examples)
    )
    if external_duplicate_sample_count:
        print(
            f"Duplicate external generation validation samples detected: {external_duplicate_sample_count}",
            file=sys.stderr,
        )
        cleanup_distributed(world_size)
        return 2
    if external_selected_group_overlap_count:
        print(
            "Training/external-generation-validation semantic-group overlap detected: "
            f"{external_selected_group_overlap_count}",
            file=sys.stderr,
        )
        cleanup_distributed(world_size)
        return 2
    generation_validation_pool = (
        external_generation_validation_examples
        if generation_validation_paths
        else validation_examples
    )
    generation_validation_examples = (
        select_generation_validation_examples(
            generation_validation_pool,
            args.generation_validation_sample_limit,
            args.validation_seed,
            args.generation_validation_examples_per_group,
        )
        if args.validation_selection_metric == "generation"
        else []
    )
    generation_validation_ids_hash = (
        sample_ids_hash(generation_validation_examples) if generation_validation_examples else ""
    )
    generation_validation_group_ids_hash = (
        validation_group_ids_hash(generation_validation_examples) if generation_validation_examples else ""
    )
    generation_validation_dataset_counts = Counter(
        str(row.get("dataset_key", "")) for row in generation_validation_examples
    )
    generation_validation_source_counts = Counter(
        str(row.get("source", "")) for row in generation_validation_examples
    )
    generation_validation_code_scoring = (
        "formal_humaneval_v11_mbpp_assert_execution_v1"
        if generation_validation_examples
        and all(
            str(row.get("dataset_key", "")) == "humaneval"
            and isinstance(row.get("code_eval"), dict)
            and row["code_eval"].get("kind") == "mbpp_assert_tests_v1"
            for row in generation_validation_examples
        )
        else "differential_execution_v1.2"
    )
    scheduled_optimizer_steps = planned_optimizer_steps(
        len(examples),
        world_size,
        args.batch_size,
        args.grad_accum_steps,
        args.epochs,
        args.max_steps,
    )
    if scheduled_optimizer_steps < args.min_optimizer_steps:
        print(
            "Training schedule is too short: "
            f"optimizer_steps={scheduled_optimizer_steps} required={args.min_optimizer_steps}",
            file=sys.stderr,
        )
        cleanup_distributed(world_size)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False
    target_module_args = args.target_module if args.target_module is not None else ["q_proj", "v_proj"]
    target_modules = tuple(dict.fromkeys(parse_csv_values(target_module_args)))
    replaced_modules: list[str] = []
    frozen_base_adapter_config: dict[str, Any] = {}
    frozen_parent_modules: list[str] = []
    if init_adapter_dir is not None:
        init_adapter_config = load_lora_adapter(model, init_adapter_dir)
    else:
        init_adapter_config = {
            "adapter_type": "manual_lora",
            "stage": args.stage_name,
            "base_model": display_path(model_dir),
            "target_modules": list(target_modules),
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        }
        if frozen_base_adapter_dir is not None:
            frozen_base_adapter_config = load_lora_adapter(model, frozen_base_adapter_dir)
            replaced_modules, frozen_parent_modules = apply_residual_lora_to_model(
                model,
                target_modules,
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
            )
        else:
            replaced_modules = apply_lora_to_model(
                model,
                target_modules,
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
            )
    model.to(device)
    model.train()
    trainable_count, total_count = trainable_parameters(model)
    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    train_dataset = MixedInstructionDataset(
        examples, tokenizer, args.max_length, args.close_reasoning_prefix
    )
    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [param for param in train_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_update_steps = scheduled_optimizer_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(step, total_update_steps, args.warmup_ratio, args.lr_scheduler),
    )

    started = time.perf_counter()
    losses: list[float] = []
    total_losses: list[float] = []
    parent_preservation_losses: list[float] = []
    protected_token_count = 0
    supervised_token_count = 0
    global_step = 0
    update_step = 0
    epoch_index = 0
    validation_history: list[dict[str, float | int]] = []
    initial_validation_loss: float | None = None
    initial_generation_validation_score: float | None = None
    initial_adapter_state: dict[str, torch.Tensor] | None = None
    best_validation_loss: float | None = None
    best_generation_validation_score: float | None = None
    peak_generation_validation_score: float | None = None
    peak_generation_validation_step = 0
    best_update_step = 0
    best_adapter_state: dict[str, torch.Tensor] | None = None
    bad_validation_checks = 0
    stopped_early = False
    if validation_examples:
        initial_validation_loss = evaluate_validation_loss(
            model,
            validation_examples,
            tokenizer,
            args.max_length,
            args.batch_size,
            device,
            rank,
            world_size,
            args.close_reasoning_prefix,
        )
        history_row: dict[str, float | int] = {"update_step": 0, "loss": initial_validation_loss}
        if generation_validation_examples:
            initial_generation_validation_score, generation_count = evaluate_generation_validation(
                model,
                generation_validation_examples,
                tokenizer,
                args.max_length,
                args.generation_validation_max_new_tokens,
                args.generation_validation_code_timeout_sec,
                device,
                rank,
                world_size,
                args.close_reasoning_prefix,
            )
            history_row["generation_score"] = initial_generation_validation_score
            history_row["generation_count"] = generation_count
            best_generation_validation_score = initial_generation_validation_score
            peak_generation_validation_score = initial_generation_validation_score
        validation_history.append(history_row)
        best_validation_loss = initial_validation_loss
        initial_adapter_state = lora_state_dict(model)
        best_adapter_state = initial_adapter_state
        if is_rank_zero(rank):
            generation_text = (
                f" generation_score={initial_generation_validation_score:.6f}"
                if initial_generation_validation_score is not None
                else ""
            )
            print(
                f"[VALIDATION] update=0 loss={initial_validation_loss:.6f}{generation_text} "
                "baseline=init_adapter",
                flush=True,
            )
    optimizer.zero_grad(set_to_none=True)
    while update_step < total_update_steps:
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        for batch in loader:
            global_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            protected_mask: torch.Tensor | None = None
            if args.parent_preservation_weight > 0:
                protected_mask = parent_correct_token_mask(model, batch)
            output = train_model(**batch)
            total_loss = output.loss
            if protected_mask is not None:
                shifted_logits = output.logits[:, :-1].contiguous()
                shifted_labels = batch["labels"][:, 1:].contiguous()
                token_losses = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]),
                    shifted_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shifted_labels)
                if protected_mask.any():
                    preservation_loss = token_losses[protected_mask].mean()
                    total_loss = total_loss + args.parent_preservation_weight * preservation_loss
                    parent_preservation_losses.append(float(preservation_loss.detach().cpu()))
                protected_token_count += int(protected_mask.sum().detach().cpu())
                supervised_token_count += int(shifted_labels.ne(-100).sum().detach().cpu())
            loss = total_loss / args.grad_accum_steps
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
            total_losses.append(float(total_loss.detach().cpu()))
            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in train_model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if is_rank_zero(rank) and (update_step % args.log_every == 0 or update_step == 1):
                    mean_loss = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
                    mean_total_loss = sum(total_losses[-args.log_every :]) / min(
                        len(total_losses), args.log_every
                    )
                    print(
                        f"[REPAIR] update={update_step}/{total_update_steps} "
                        f"global_step={global_step} world_size={world_size} loss={mean_loss:.4f} "
                        f"total_loss={mean_total_loss:.4f}",
                        flush=True,
                    )
                should_validate = bool(
                    validation_examples
                    and args.eval_every > 0
                    and (update_step % args.eval_every == 0 or update_step >= total_update_steps)
                )
                if should_validate:
                    validation_loss = evaluate_validation_loss(
                        model,
                        validation_examples,
                        tokenizer,
                        args.max_length,
                        args.batch_size,
                        device,
                        rank,
                        world_size,
                        args.close_reasoning_prefix,
                    )
                    history_row = {"update_step": update_step, "loss": validation_loss}
                    generation_score: float | None = None
                    if generation_validation_examples:
                        generation_score, generation_count = evaluate_generation_validation(
                            model,
                            generation_validation_examples,
                            tokenizer,
                            args.max_length,
                            args.generation_validation_max_new_tokens,
                            args.generation_validation_code_timeout_sec,
                            device,
                            rank,
                            world_size,
                            args.close_reasoning_prefix,
                        )
                        history_row["generation_score"] = generation_score
                        history_row["generation_count"] = generation_count
                        if peak_generation_validation_score is None or generation_score > peak_generation_validation_score:
                            peak_generation_validation_score = generation_score
                            peak_generation_validation_step = update_step
                    validation_history.append(history_row)
                    if args.validation_selection_metric == "generation":
                        improved = bool(
                            generation_score is not None
                            and (
                                best_generation_validation_score is None
                                or generation_score
                                > best_generation_validation_score + args.min_generation_validation_improvement
                            )
                        )
                    else:
                        improved = (
                            best_validation_loss is None
                            or validation_loss < best_validation_loss - args.min_validation_improvement
                        )
                    if improved:
                        best_validation_loss = validation_loss
                        if generation_score is not None:
                            best_generation_validation_score = generation_score
                        best_update_step = update_step
                        best_adapter_state = lora_state_dict(model)
                        bad_validation_checks = 0
                    else:
                        bad_validation_checks += 1
                    if is_rank_zero(rank):
                        generation_text = (
                            f" generation_score={generation_score:.6f} "
                            f"best_generation={best_generation_validation_score:.6f}"
                            if generation_score is not None and best_generation_validation_score is not None
                            else ""
                        )
                        print(
                            f"[VALIDATION] update={update_step} loss={validation_loss:.6f}"
                            f"{generation_text} "
                            f"best={best_validation_loss:.6f} best_update={best_update_step} "
                            f"bad_checks={bad_validation_checks}",
                            flush=True,
                        )
                    if (
                        args.early_stopping_patience > 0
                        and bad_validation_checks >= args.early_stopping_patience
                        and update_step >= args.min_optimizer_steps
                    ):
                        stopped_early = True
                if update_step >= total_update_steps:
                    break
                if stopped_early:
                    break
        epoch_index += 1
        if stopped_early:
            break

    if validation_examples and (not validation_history or validation_history[-1]["update_step"] != update_step):
        validation_loss = evaluate_validation_loss(
            model,
            validation_examples,
            tokenizer,
            args.max_length,
            args.batch_size,
            device,
            rank,
            world_size,
            args.close_reasoning_prefix,
        )
        history_row = {"update_step": update_step, "loss": validation_loss}
        generation_score = None
        if generation_validation_examples:
            generation_score, generation_count = evaluate_generation_validation(
                model,
                generation_validation_examples,
                tokenizer,
                args.max_length,
                args.generation_validation_max_new_tokens,
                args.generation_validation_code_timeout_sec,
                device,
                rank,
                world_size,
                args.close_reasoning_prefix,
            )
            history_row["generation_score"] = generation_score
            history_row["generation_count"] = generation_count
            if peak_generation_validation_score is None or generation_score > peak_generation_validation_score:
                peak_generation_validation_score = generation_score
                peak_generation_validation_step = update_step
        validation_history.append(history_row)
        if args.validation_selection_metric == "generation":
            improved = bool(
                generation_score is not None
                and (
                    best_generation_validation_score is None
                    or generation_score
                    > best_generation_validation_score + args.min_generation_validation_improvement
                )
            )
        else:
            improved = (
                best_validation_loss is None
                or validation_loss < best_validation_loss - args.min_validation_improvement
            )
        if improved:
            best_validation_loss = validation_loss
            if generation_score is not None:
                best_generation_validation_score = generation_score
            best_update_step = update_step
            best_adapter_state = lora_state_dict(model)

    candidate_best_validation_loss = best_validation_loss
    candidate_best_update_step = best_update_step
    validation_relative_improvement = (
        (initial_validation_loss - candidate_best_validation_loss) / initial_validation_loss
        if initial_validation_loss is not None
        and candidate_best_validation_loss is not None
        and initial_validation_loss > 0
        else None
    )
    validation_guard_rejected = bool(
        args.min_validation_relative_improvement > 0
        and validation_relative_improvement is not None
        and validation_relative_improvement < args.min_validation_relative_improvement
    )
    generation_guard_rejected = bool(
        args.validation_selection_metric == "generation" and best_update_step == 0
    )
    if validation_guard_rejected and initial_adapter_state is not None and initial_validation_loss is not None:
        best_adapter_state = initial_adapter_state
        best_validation_loss = initial_validation_loss
        best_update_step = 0
        if is_rank_zero(rank):
            print(
                f"[VALIDATION] rejected candidate relative_improvement={validation_relative_improvement:.6f} "
                f"required={args.min_validation_relative_improvement:.6f}; restoring step=0",
                flush=True,
            )

    if args.restore_best_validation and best_adapter_state is not None:
        load_lora_state(model, best_adapter_state)
        if world_size > 1:
            torch.distributed.barrier()
        if is_rank_zero(rank):
            generation_text = (
                f" generation_score={best_generation_validation_score:.6f}"
                if best_generation_validation_score is not None
                else ""
            )
            print(
                f"[VALIDATION] restored best adapter from update={best_update_step} "
                f"loss={best_validation_loss:.6f}{generation_text}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    loss_stats = torch.tensor([sum(losses), float(len(losses))], device=device)
    training_stats = torch.tensor(
        [
            sum(total_losses),
            float(len(total_losses)),
            sum(parent_preservation_losses),
            float(len(parent_preservation_losses)),
            float(protected_token_count),
            float(supervised_token_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        torch.distributed.all_reduce(loss_stats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(training_stats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()
    global_mean_loss = float((loss_stats[0] / loss_stats[1]).detach().cpu()) if loss_stats[1].item() else None
    global_mean_total_loss = (
        float((training_stats[0] / training_stats[1]).detach().cpu()) if training_stats[1].item() else None
    )
    global_mean_parent_preservation_loss = (
        float((training_stats[2] / training_stats[3]).detach().cpu()) if training_stats[3].item() else None
    )
    global_parent_correct_token_ratio = (
        float((training_stats[4] / training_stats[5]).detach().cpu()) if training_stats[5].item() else None
    )
    base_model = train_model.module if hasattr(train_model, "module") else train_model
    if not is_rank_zero(rank):
        cleanup_distributed(world_size)
        return 0

    adapter_config = dict(init_adapter_config)
    adapter_config.update(
        {
            "stage": args.stage_name,
            "base_model": display_path(model_dir),
            "init_adapter": display_path(init_adapter_dir) if init_adapter_dir is not None else "",
            "frozen_base_adapter": (
                display_path(frozen_base_adapter_dir) if frozen_base_adapter_dir is not None else ""
            ),
            "frozen_base_adapter_hash": (
                sha256_dir(frozen_base_adapter_dir) if frozen_base_adapter_dir is not None else ""
            ),
            "frozen_base_adapter_config": frozen_base_adapter_config,
            "frozen_parent_module_count": len(frozen_parent_modules),
            "frozen_parent_modules_hash": (
                sha256_text("\n".join(frozen_parent_modules) + "\n") if frozen_parent_modules else ""
            ),
            "distributed_world_size": world_size,
            "target_modules": list(target_modules),
            "rank": int(adapter_config.get("rank", args.lora_rank)),
            "alpha": float(adapter_config.get("alpha", args.lora_alpha)),
            "dropout": float(adapter_config.get("dropout", args.lora_dropout)),
            "dtype": args.dtype,
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
            "selected_sample_count": len(selected_examples),
            "selected_sample_ids_hash": selected_ids_hash,
            "selected_validation_group_ids_hash": selected_group_ids_hash,
            "training_sample_count": len(examples),
            "training_sample_count_before_group_cap": training_sample_count_before_group_cap,
            "max_train_examples_per_validation_group": args.max_train_examples_per_validation_group,
            "training_sample_ids_hash": training_ids_hash,
            "training_validation_group_ids_hash": training_group_ids_hash,
            "validation_sample_count": len(validation_examples),
            "validation_sample_ids_hash": validation_ids_hash,
            "validation_group_ids_hash": validation_group_ids_hash_value,
            "validation_group_overlap_count": validation_group_overlap_count,
            "validation_group_count": validation_group_count,
            "min_validation_group_count": args.min_validation_group_count,
            "distill_data_hash": sha256_file(distill_path),
            "repair_trace_hash": repair_trace_hash,
            "capability_rehearsal_paths": [display_path(path) for path in capability_rehearsal_paths],
            "capability_rehearsal_hashes": {
                display_path(path): sha256_file(path) for path in capability_rehearsal_paths if path.is_file()
            },
            "structured_prompt_template_hash": sha256_text(STRUCTURED_SYSTEM_PROMPT + "\n" + STRUCTURED_USER_PROMPT_TEMPLATE),
            "repair_prompt_template_hash": sha256_text(REPAIR_SYSTEM_PROMPT + "\n" + REPAIR_USER_PROMPT_TEMPLATE),
            "capability_prompt_template_hash": sha256_text(CAPABILITY_SYSTEM_PROMPT),
            "source_limits": dict(sorted(source_limits.items())),
            "pre_limit_source_counts": dict(sorted(source_counts_before_limits.items())),
            "pre_limit_dataset_counts": dict(sorted(dataset_counts_before_limits.items())),
            "mixture_source_counts": dict(sorted(source_counts.items())),
            "mixture_dataset_counts": dict(sorted(dataset_counts.items())),
            "validation_source_counts": dict(sorted(validation_source_counts.items())),
            "validation_dataset_counts": dict(sorted(validation_dataset_counts.items())),
            "generation_validation_sample_count": len(generation_validation_examples),
            "generation_validation_examples_per_group": args.generation_validation_examples_per_group,
            "generation_validation_sample_ids_hash": generation_validation_ids_hash,
            "generation_validation_group_ids_hash": generation_validation_group_ids_hash,
            "generation_validation_dataset_counts": dict(
                sorted(generation_validation_dataset_counts.items())
            ),
            "generation_validation_source_counts": dict(
                sorted(generation_validation_source_counts.items())
            ),
            "generation_validation_paths": [display_path(path) for path in generation_validation_paths],
            "generation_validation_hashes": {
                display_path(path): sha256_file(path) for path in generation_validation_paths
            },
            "generation_validation_external": bool(generation_validation_paths),
            "generation_validation_external_duplicate_sample_count": external_duplicate_sample_count,
            "generation_validation_external_group_overlap_count": external_selected_group_overlap_count,
            "generation_validation_code_scoring": generation_validation_code_scoring,
            "generation_validation_aggregation": "macro_accuracy",
            "close_reasoning_prefix": bool(args.close_reasoning_prefix),
            "generation_validation_code_timeout_sec": args.generation_validation_code_timeout_sec,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "validation_selection_metric": args.validation_selection_metric,
            "scheduled_optimizer_steps": scheduled_optimizer_steps,
            "min_optimizer_steps": args.min_optimizer_steps,
            "best_validation_loss": best_validation_loss,
            "best_update_step": best_update_step,
            "initial_generation_validation_score": initial_generation_validation_score,
            "best_generation_validation_score": best_generation_validation_score,
            "peak_generation_validation_score": peak_generation_validation_score,
            "peak_generation_validation_step": peak_generation_validation_step,
            "min_generation_validation_improvement": args.min_generation_validation_improvement,
            "generation_guard_rejected": generation_guard_rejected,
            "initial_validation_loss": initial_validation_loss,
            "candidate_best_validation_loss": candidate_best_validation_loss,
            "candidate_best_update_step": candidate_best_update_step,
            "validation_relative_improvement": validation_relative_improvement,
            "min_validation_relative_improvement": args.min_validation_relative_improvement,
            "validation_guard_rejected": validation_guard_rejected,
            "parent_preservation_weight": args.parent_preservation_weight,
            "parent_reference_kind": parent_reference_kind,
            "restore_best_validation": bool(args.restore_best_validation),
            "target_schema": "mixed_structured_repair_and_capability_rehearsal",
            "replaced_module_count": len(replaced_modules),
        }
    )
    save_lora_adapter(base_model, output_dir, adapter_config)
    adapter_hash = sha256_dir(output_dir)
    status = "passed" if losses and adapter_hash else "failed"
    audit = {
        "gate": f"G-KD-TRACE-{args.stage_name}-train-smoke" if args.smoke else f"G-KD-TRACE-{args.stage_name}-train",
        "check_version": "1.5",
        "created_by": "model_compression/train_cedd_repair.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "smoke": bool(args.smoke),
        "student_init": display_path(model_dir),
        "init_adapter": display_path(init_adapter_dir) if init_adapter_dir is not None else "",
        "init_adapter_hash": sha256_dir(init_adapter_dir) if init_adapter_dir is not None else "",
        "frozen_base_adapter": display_path(frozen_base_adapter_dir) if frozen_base_adapter_dir is not None else "",
        "frozen_base_adapter_hash": (
            sha256_dir(frozen_base_adapter_dir) if frozen_base_adapter_dir is not None else ""
        ),
        "frozen_parent_module_count": len(frozen_parent_modules),
        "frozen_parent_modules_hash": (
            sha256_text("\n".join(frozen_parent_modules) + "\n") if frozen_parent_modules else ""
        ),
        "distill_data_path": display_path(distill_path),
        "distill_data_hash": sha256_file(distill_path),
        "repair_trace_path": display_path(repair_path),
        "repair_trace_hash": repair_trace_hash,
        "capability_rehearsal_paths": [display_path(path) for path in capability_rehearsal_paths],
        "capability_rehearsal_hashes": {
            display_path(path): sha256_file(path) for path in capability_rehearsal_paths if path.is_file()
        },
        "output_dir": display_path(output_dir),
        "adapter_hash": adapter_hash,
        "adapter_config_hash": sha256_file(output_dir / "adapter_config.json"),
        "adapter_model_hash": sha256_file(output_dir / "adapter_model.pt"),
        "source_counts": dict(sorted(source_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "pre_limit_source_counts": dict(sorted(source_counts_before_limits.items())),
        "pre_limit_dataset_counts": dict(sorted(dataset_counts_before_limits.items())),
        "source_limits": dict(sorted(source_limits.items())),
        "selected_sample_count": len(selected_examples),
        "selected_sample_ids_hash": selected_ids_hash,
        "selected_validation_group_ids_hash": selected_group_ids_hash,
        "training_sample_count": len(examples),
        "training_sample_count_before_group_cap": training_sample_count_before_group_cap,
        "max_train_examples_per_validation_group": args.max_train_examples_per_validation_group,
        "training_sample_ids_hash": training_ids_hash,
        "training_validation_group_ids_hash": training_group_ids_hash,
        "validation_sample_count": len(validation_examples),
        "validation_sample_ids_hash": validation_ids_hash,
        "validation_group_ids_hash": validation_group_ids_hash_value,
        "validation_group_overlap_count": validation_group_overlap_count,
        "validation_group_count": validation_group_count,
        "min_validation_group_count": args.min_validation_group_count,
        "validation_source_counts": dict(sorted(validation_source_counts.items())),
        "validation_dataset_counts": dict(sorted(validation_dataset_counts.items())),
        "generation_validation_sample_count": len(generation_validation_examples),
        "generation_validation_sample_ids_hash": generation_validation_ids_hash,
        "generation_validation_group_ids_hash": generation_validation_group_ids_hash,
        "generation_validation_dataset_counts": dict(
            sorted(generation_validation_dataset_counts.items())
        ),
        "generation_validation_source_counts": dict(
            sorted(generation_validation_source_counts.items())
        ),
        "generation_validation_paths": [display_path(path) for path in generation_validation_paths],
        "generation_validation_hashes": {
            display_path(path): sha256_file(path) for path in generation_validation_paths
        },
        "generation_validation_external": bool(generation_validation_paths),
        "generation_validation_external_duplicate_sample_count": external_duplicate_sample_count,
        "generation_validation_external_group_overlap_count": external_selected_group_overlap_count,
        "generation_validation_code_scoring": generation_validation_code_scoring,
        "generation_validation_aggregation": "macro_accuracy",
        "close_reasoning_prefix": bool(args.close_reasoning_prefix),
        "generation_validation_code_timeout_sec": args.generation_validation_code_timeout_sec,
        "capability_datasets": sorted(capability_datasets),
        "structured_dataset_filter": sorted(dataset_filter) if dataset_filter is not None else [],
        "repair_dataset_filter": sorted(repair_dataset_filter) if repair_dataset_filter is not None else [],
        "distill_sample_limit": args.distill_sample_limit,
        "repair_sample_limit": args.repair_sample_limit,
        "capability_sample_limit": args.capability_sample_limit,
        "distill_repeat": args.distill_repeat,
        "repair_repeat": args.repair_repeat,
        "capability_repeat": args.capability_repeat,
        "capability_jsonl_repeat": args.capability_jsonl_repeat,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parent_preservation_weight": args.parent_preservation_weight,
        "parent_reference_kind": parent_reference_kind,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "validation_selection_metric": args.validation_selection_metric,
        "eval_every": args.eval_every,
        "early_stopping_patience": args.early_stopping_patience,
        "min_validation_improvement": args.min_validation_improvement,
        "min_validation_relative_improvement": args.min_validation_relative_improvement,
        "restore_best_validation": bool(args.restore_best_validation),
        "validation_history": validation_history,
        "initial_validation_loss": initial_validation_loss,
        "candidate_best_validation_loss": candidate_best_validation_loss,
        "candidate_best_update_step": candidate_best_update_step,
        "validation_relative_improvement": validation_relative_improvement,
        "validation_guard_rejected": validation_guard_rejected,
        "generation_validation_sample_limit": args.generation_validation_sample_limit,
        "generation_validation_examples_per_group": args.generation_validation_examples_per_group,
        "generation_validation_max_new_tokens": args.generation_validation_max_new_tokens,
        "min_generation_validation_improvement": args.min_generation_validation_improvement,
        "initial_generation_validation_score": initial_generation_validation_score,
        "best_generation_validation_score": best_generation_validation_score,
        "peak_generation_validation_score": peak_generation_validation_score,
        "peak_generation_validation_step": peak_generation_validation_step,
        "generation_guard_rejected": generation_guard_rejected,
        "best_validation_loss": best_validation_loss,
        "best_update_step": best_update_step,
        "stopped_early": stopped_early,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(target_modules),
        "replaced_module_count": len(replaced_modules),
        "distributed_world_size": world_size,
        "distributed_backend": "nccl" if world_size > 1 else "none",
        "effective_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_parameter_ratio": trainable_count / total_count if total_count else 0.0,
        "global_steps": global_step,
        "optimizer_steps": update_step,
        "scheduled_optimizer_steps": scheduled_optimizer_steps,
        "min_optimizer_steps": args.min_optimizer_steps,
        "mean_loss": global_mean_loss,
        "mean_total_loss": global_mean_total_loss,
        "mean_parent_preservation_loss": global_mean_parent_preservation_loss,
        "parent_correct_token_ratio": global_parent_correct_token_ratio,
        "protected_token_count": int(training_stats[4].detach().cpu()),
        "supervised_token_count": int(training_stats[5].detach().cpu()),
        "final_loss": losses[-1] if losses else None,
        "final_total_loss": total_losses[-1] if total_losses else None,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_sec": elapsed,
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_dir)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"adapter_hash={adapter_hash}")
    cleanup_distributed(world_size)
    if status != "passed":
        return 1
    print("CEDD-Repair training passed.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
