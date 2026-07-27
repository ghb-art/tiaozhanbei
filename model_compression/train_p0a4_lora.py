#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a4_distillation.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")
FORMAL_MARKERS = ("gsm8k/test/", "cmmlu/test/", "humaneval/", "official_full", "final_test")
FORBIDDEN_INPUT_ROOTS = (
    ROOT / "reports/sealed",
    ROOT / "data/eval",
    ROOT / "data/splits/p0a4_official_full",
)
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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


def validate_gpu_zero3_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"Missing DeepSpeed config: {display_path(path)}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"Invalid DeepSpeed JSON: {display_path(path)}: {exc}") from exc
    if not isinstance(config, dict):
        raise TrainingError(f"DeepSpeed config must be an object: {display_path(path)}")
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or int(zero.get("stage", 0)) != 3:
        raise TrainingError("P0-A4 Teacher requires DeepSpeed ZeRO stage 3")
    forbidden = [key for key in ("offload_optimizer", "offload_param") if key in zero]
    if forbidden:
        raise TrainingError(
            "P0-A4 GPU ZeRO-3 forbids CPU offload settings: " + ", ".join(forbidden)
        )
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingError(f"Missing training data: {display_path(path)}")
    resolved = path.resolve()
    if any(resolved == root.resolve() or root.resolve() in resolved.parents for root in FORBIDDEN_INPUT_ROOTS):
        raise TrainingError(f"Formal/sealed artifact cannot be a training input: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TrainingError(f"Non-object row {display_path(path)}:{line_number}")
            identity = " ".join(
                str(value.get(key, "")).lower()
                for key in ("sample_id", "source_sample_id", "validation_group_id", "source", "split_role")
            )
            if any(marker in identity for marker in FORMAL_MARKERS):
                raise TrainingError(
                    f"Formal-test reference rejected at {display_path(path)}:{line_number}"
                )
            messages = value.get("messages")
            if not isinstance(messages, list) or not messages or not str(value.get("answer", "")).strip():
                raise TrainingError(f"Incomplete SFT row {display_path(path)}:{line_number}")
            rows.append(value)
    if not rows:
        raise TrainingError(f"Empty training data: {display_path(path)}")
    return rows


def balance_rows(
    rows: list[dict[str, Any]],
    seed: int,
    max_upsample_factor: float | None = None,
    target_rows_by_task: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    by_task = {task: [row for row in rows if row.get("dataset_key") == task] for task in TASKS}
    if any(not values for values in by_task.values()):
        missing = [task for task, values in by_task.items() if not values]
        raise TrainingError(f"Missing task data: {missing}")
    if max_upsample_factor is not None and max_upsample_factor < 1:
        raise TrainingError("max_upsample_factor must be >= 1")
    if target_rows_by_task is not None:
        if set(target_rows_by_task) != set(TASKS):
            raise TrainingError(f"target_rows_by_task must contain exactly {TASKS}")
        targets = {task: int(target_rows_by_task[task]) for task in TASKS}
        if any(value < 1 for value in targets.values()):
            raise TrainingError("target_rows_by_task values must be positive")
        if max_upsample_factor is not None:
            excessive = {
                task: {"target": targets[task], "maximum": math.floor(len(by_task[task]) * max_upsample_factor)}
                for task in TASKS
                if targets[task] > math.floor(len(by_task[task]) * max_upsample_factor)
            }
            if excessive:
                raise TrainingError(f"target_rows_by_task exceeds max_upsample_factor: {excessive}")
    else:
        target = max(len(values) for values in by_task.values())
        if max_upsample_factor is not None:
            target = min(
                target,
                max(1, math.floor(min(map(len, by_task.values())) * max_upsample_factor)),
            )
        targets = {task: target for task in TASKS}
    balanced: list[dict[str, Any]] = []
    for task, values in by_task.items():
        ordered = sorted(values, key=lambda row: sha256_text(f"{seed}:{row.get('sample_id', '')}"))
        target = targets[task]
        repeats = math.ceil(target / len(ordered))
        balanced.extend((ordered * repeats)[:target])
    random.Random(seed).shuffle(balanced)
    return balanced


def unique_prompt_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    prompts: dict[str, set[str]] = {task: set() for task in TASKS}
    for row in rows:
        task = str(row.get("dataset_key", ""))
        if task in prompts:
            prompts[task].add(
                sha256_text(json.dumps(row.get("messages", []), ensure_ascii=False, sort_keys=True))
            )
    return {task: len(values) for task, values in prompts.items()}


def equal_source_loss_weights(
    rows: list[dict[str, Any]],
    source_key: str,
) -> tuple[dict[str, int], dict[str, float]]:
    """Give each declared source equal total loss mass without duplicating rows."""
    if not source_key.strip():
        raise TrainingError("source balance key must not be empty")
    values = [str(row.get(source_key, "")).strip() for row in rows]
    if any(not value for value in values):
        raise TrainingError(f"Missing source balance value for key: {source_key}")
    counts = dict(Counter(values))
    if len(counts) < 2:
        raise TrainingError(
            f"Source balancing requires at least two sources for {source_key}: {counts}"
        )
    row_count = len(rows)
    source_count = len(counts)
    weights = {
        source: row_count / (source_count * count)
        for source, count in sorted(counts.items())
    }
    return counts, weights


def explicit_sample_weight_summary(
    rows: list[dict[str, Any]],
    weight_key: str,
) -> dict[str, Any]:
    """Validate pre-registered row weights and summarize their loss mass."""
    if not weight_key.strip():
        raise TrainingError("sample weight key must not be empty")
    mass_by_task: Counter[str] = Counter()
    mass_by_code_origin: Counter[str] = Counter()
    minimum = math.inf
    maximum = 0.0
    for row in rows:
        if weight_key not in row:
            raise TrainingError(
                f"Missing explicit sample weight key {weight_key}: "
                f"{row.get('sample_id', '<missing>')}"
            )
        weight = float(row[weight_key])
        if not math.isfinite(weight) or weight <= 0:
            raise TrainingError(
                f"Invalid explicit sample weight {weight}: "
                f"{row.get('sample_id', '<missing>')}"
            )
        task = str(row.get("dataset_key", ""))
        mass_by_task[task] += weight
        if task == "humaneval":
            origin = str(row.get("origin", "")).strip()
            if not origin:
                raise TrainingError(
                    "Code row with explicit weights is missing origin: "
                    f"{row.get('sample_id', '<missing>')}"
                )
            mass_by_code_origin[origin] += weight
        minimum = min(minimum, weight)
        maximum = max(maximum, weight)
    return {
        "key": weight_key,
        "minimum": minimum,
        "maximum": maximum,
        "mass_by_task": dict(mass_by_task),
        "mass_by_code_origin": dict(mass_by_code_origin),
        "row_duplication": False,
    }


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in ("torch", "transformers", "accelerate", "deepspeed", "peft"):
        spec = importlib.util.find_spec(module)
        if spec is None:
            versions[module] = "missing"
            continue
        loaded = __import__(module)
        versions[module] = str(getattr(loaded, "__version__", "unknown"))
    return versions


def publish_best_adapter_checkpoint(best_checkpoint: Path, output_dir: Path) -> list[str]:
    """Publish a Trainer PEFT checkpoint without reloading it inside active DDP."""
    config_path = best_checkpoint / "adapter_config.json"
    weight_candidates = (
        best_checkpoint / "adapter_model.safetensors",
        best_checkpoint / "adapter_model.bin",
    )
    weight_path = next((path for path in weight_candidates if path.is_file()), None)
    if not config_path.is_file() or weight_path is None:
        raise TrainingError(
            f"Best checkpoint is missing PEFT adapter files: {display_path(best_checkpoint)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for source in (config_path, weight_path, best_checkpoint / "README.md"):
        if source.is_file():
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            published.append(destination.name)
    return published


def render_tokenized(tokenizer: Any, row: dict[str, Any], max_length: int, disable_thinking: bool) -> dict[str, Any]:
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in row["messages"]
        if isinstance(message, dict) and message.get("role") in {"system", "user"}
    ]
    full_messages = messages + [{"role": "assistant", "content": str(row["answer"])}]
    template_kwargs = {"tokenize": True, "add_generation_prompt": True}
    full_kwargs = {"tokenize": True, "add_generation_prompt": False}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
        full_kwargs["enable_thinking"] = False
    prompt_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    full_ids = tokenizer.apply_chat_template(full_messages, **full_kwargs)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise TrainingError(f"Answer was truncated completely: {row.get('sample_id', '<missing>')}")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class TokenizedDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        disable_thinking: bool,
        source_balance_key: str = "",
        source_loss_weights: dict[str, float] | None = None,
        sample_weight_key: str = "",
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.disable_thinking = disable_thinking
        self.source_balance_key = source_balance_key
        self.source_loss_weights = source_loss_weights or {}
        self.sample_weight_key = sample_weight_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = render_tokenized(
            self.tokenizer, self.rows[index], self.max_length, self.disable_thinking
        )
        if self.source_balance_key:
            source = str(self.rows[index].get(self.source_balance_key, "")).strip()
            try:
                feature["sample_weight"] = float(self.source_loss_weights[source])
            except KeyError as exc:
                raise TrainingError(f"Missing loss weight for source: {source}") from exc
        elif self.sample_weight_key:
            feature["sample_weight"] = float(
                self.rows[index][self.sample_weight_key]
            )
        return feature


class SupervisedCollator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if any("sample_weight" in feature for feature in features):
            if not all("sample_weight" in feature for feature in features):
                raise TrainingError("A batch cannot mix weighted and unweighted training rows")
            batch["sample_weight"] = torch.tensor(
                [float(feature["sample_weight"]) for feature in features],
                dtype=torch.float32,
            )
        return batch


def role_settings(config: dict[str, Any], role: str) -> tuple[dict[str, Any], str, str]:
    if role == "teacher":
        return (
            dict(config["training"]["teacher"]),
            str(config["models"]["distill_teacher"]["local_dir"]),
            str(config["data"]["train"]),
        )
    if role == "student_shared":
        return (
            dict(config["training"]["student_shared"]),
            str(config["models"]["student"]["local_dir"]),
            str(config["data"]["teacher_distill"]),
        )
    if role == "student_expert":
        settings = dict(config["training"]["student_expert"])
        settings.update(
            {
                "lora_rank": int(settings["initial_rank"]),
                "max_seq_length": int(config["training"]["student_shared"]["max_seq_length"]),
                "learning_rate": float(config["training"]["student_shared"]["learning_rate"]),
                "epochs": 2,
            }
        )
        return (
            settings,
            str(config["models"]["student"]["shared_merged_dir"]),
            str(config["data"]["teacher_distill"]),
        )
    raise TrainingError(f"Unsupported training role: {role}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-safe P0-A4 LoRA trainer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--role", choices=("teacher", "student_shared", "student_expert"), required=True)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model-dir")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument(
        "--source-balance-key",
        default="",
        help=(
            "Optional row field used for equal-source loss weighting. This preserves every "
            "unique row exactly once and does not create duplicated training examples."
        ),
    )
    parser.add_argument(
        "--sample-weight-key",
        default="",
        help=(
            "Optional pre-registered positive row field used as the per-example loss "
            "weight. Unlike --source-balance-key, this is allowed for shared Student "
            "training and must be present on every selected training row."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epoch-based training for bounded compatibility smoke runs.",
    )
    parser.add_argument("--deepspeed", default="")
    parser.add_argument(
        "--external-checkpoint-selection",
        action="store_true",
        help="Keep every epoch checkpoint and do not publish an adapter until an external task metric selects it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    is_main_process = True
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        settings, default_model, default_train = role_settings(config, args.role)
        if args.role == "teacher":
            limit = int(config["models"]["distill_teacher"]["max_candidates"])
        else:
            limit = int(config["models"]["student"]["max_selection_candidates"])
        if args.candidate_index < 1 or args.candidate_index > limit:
            raise TrainingError(f"candidate-index must be in [1, {limit}] for {args.role}")
        candidate_overrides = settings.pop("candidate_overrides", {})
        if candidate_overrides:
            if not isinstance(candidate_overrides, dict):
                raise TrainingError("candidate_overrides must be an object")
            override = candidate_overrides.get(str(args.candidate_index), {})
            if not isinstance(override, dict):
                raise TrainingError(
                    f"candidate_overrides[{args.candidate_index}] must be an object"
                )
            settings.update(override)
        if args.role == "student_expert" and not args.task:
            raise TrainingError("student_expert requires --task")
        if args.external_checkpoint_selection and args.role != "student_expert":
            raise TrainingError("External checkpoint selection is supported only for student_expert")
        if args.role == "student_expert" and args.task:
            for setting_key, map_key, cast in (
                ("lora_rank", "rank_by_task", int),
                ("lora_alpha", "alpha_by_task", int),
                ("learning_rate", "learning_rate_by_task", float),
                ("epochs", "epochs_by_task", float),
            ):
                values = settings.get(map_key, {})
                if values and args.task in values:
                    settings[setting_key] = cast(values[args.task])

        model_dir = resolve_path(args.model_dir or default_model)
        train_path = resolve_path(args.train_data or default_train)
        validation_path = resolve_path(
            args.validation_data or config["data"]["teacher_validation"]
        )
        deepspeed_path = resolve_path(args.deepspeed) if args.deepspeed else None
        deepspeed_config = None
        if args.role == "teacher":
            if deepspeed_path is None:
                raise TrainingError("Teacher training requires the four-GPU ZeRO-3 config")
            deepspeed_config = validate_gpu_zero3_config(deepspeed_path)
        suffix = f"-{args.task}" if args.task else ""
        output_dir = resolve_path(
            args.output_dir
            or f"models/checkpoints/p0a4/{args.role}{suffix}-v{args.candidate_index}"
        )
        audit_path = resolve_path(
            args.audit
            or f"reports/audit/gate_p0a4_train_{args.role}{suffix}_v{args.candidate_index}.json"
        )
        train_rows = read_jsonl(train_path)
        source_train_counts = dict(Counter(str(row["dataset_key"]) for row in train_rows))
        source_unique_prompt_counts = unique_prompt_counts(train_rows)
        if args.role == "student_shared":
            required_unique = {
                key: int(value)
                for key, value in settings.get("min_unique_prompts_by_task", {}).items()
            }
            unique_failures = {
                task: {
                    "actual": source_unique_prompt_counts.get(task, 0),
                    "required": required,
                }
                for task, required in required_unique.items()
                if source_unique_prompt_counts.get(task, 0) < required
            }
            if unique_failures:
                raise TrainingError(
                    f"Insufficient unique Student supervision prompts: {unique_failures}"
                )
        validation_rows = read_jsonl(validation_path)
        if args.task:
            train_rows = [row for row in train_rows if row.get("dataset_key") == args.task]
            validation_rows = [row for row in validation_rows if row.get("dataset_key") == args.task]
        elif args.role != "student_expert":
            train_rows = balance_rows(
                train_rows,
                int(config["seed"]) + args.candidate_index,
                (
                    float(settings["max_upsample_factor"])
                    if settings.get("max_upsample_factor") is not None
                    else None
                ),
                (
                    {key: int(value) for key, value in settings["target_rows_by_task"].items()}
                    if settings.get("target_rows_by_task") is not None
                    else None
                ),
            )
        if not train_rows or not validation_rows:
            raise TrainingError("Task filtering produced an empty train or validation set")
        source_balance_counts: dict[str, int] = {}
        source_loss_weights: dict[str, float] = {}
        explicit_weight_summary: dict[str, Any] = {}
        if args.source_balance_key and args.sample_weight_key:
            raise TrainingError(
                "--source-balance-key and --sample-weight-key are mutually exclusive"
            )
        if args.source_balance_key:
            if args.role != "student_expert":
                raise TrainingError("Source-balanced loss is supported only for student_expert")
            if args.per_device_train_batch_size != 1:
                raise TrainingError(
                    "Source-balanced loss requires per-device train batch size 1"
                )
            source_balance_counts, source_loss_weights = equal_source_loss_weights(
                train_rows,
                args.source_balance_key,
            )
        elif args.sample_weight_key:
            explicit_weight_summary = explicit_sample_weight_summary(
                train_rows,
                args.sample_weight_key,
            )

        rank = int(args.rank or settings.get("lora_rank", settings.get("initial_rank", 8)))
        alpha = int(args.lora_alpha if args.lora_alpha is not None else settings["lora_alpha"])
        dropout = float(
            args.lora_dropout if args.lora_dropout is not None else settings["lora_dropout"]
        )
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise TrainingError("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
        learning_rate = float(args.learning_rate or settings["learning_rate"])
        epochs = float(args.epochs or settings["epochs"])
        max_length = int(args.max_seq_length or settings["max_seq_length"])
        dependencies = dependency_versions()
        plan = {
            "gate": "P0-A4-LORA-TRAIN",
            "check_version": "1.0",
            "created_by": "model_compression/train_p0a4_lora.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "dry_run_passed" if args.dry_run else "running",
            "role": args.role,
            "task": args.task or "shared",
            "candidate_index": args.candidate_index,
            "model_dir": display_path(model_dir),
            "train_data": display_path(train_path),
            "train_data_hash": sha256_file(train_path),
            "validation_data": display_path(validation_path),
            "validation_data_hash": sha256_file(validation_path),
            "train_rows": len(train_rows),
            "source_train_counts": source_train_counts,
            "source_unique_prompt_counts": source_unique_prompt_counts,
            "validation_rows": len(validation_rows),
            "train_counts": dict(Counter(str(row["dataset_key"]) for row in train_rows)),
            "output_dir": display_path(output_dir),
            "lora": {"rank": rank, "alpha": alpha, "dropout": dropout, "target_modules": TARGET_MODULES},
            "optimization": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "max_upsample_factor": settings.get("max_upsample_factor"),
                "target_rows_by_task": settings.get("target_rows_by_task"),
                "max_seq_length": max_length,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "optimizer": "adamw_torch",
                "optimizer_placement": (
                    "gpu_zero3_partitioned" if deepspeed_config else "gpu_ddp_replicated"
                ),
                "parameter_placement": (
                    "gpu_zero3_partitioned" if deepspeed_config else "gpu_ddp_replicated"
                ),
                "deepspeed": display_path(deepspeed_path) if deepspeed_path else "",
                "deepspeed_config_hash": sha256_file(deepspeed_path) if deepspeed_path else "",
                "deepspeed_stage": (
                    int(deepspeed_config["zero_optimization"]["stage"])
                    if deepspeed_config is not None
                    else 0
                ),
                "zero3_initialized_before_model_load": args.role == "teacher",
                "cpu_offload": False,
                "gradient_checkpointing": {
                    "enabled": True,
                    "use_reentrant": False,
                },
                "ddp_find_unused_parameters": False,
                "load_best_model_in_trainer": args.role == "teacher",
                "external_checkpoint_selection": bool(args.external_checkpoint_selection),
                "source_balance": {
                    "enabled": bool(args.source_balance_key),
                    "key": args.source_balance_key,
                    "counts": source_balance_counts,
                    "loss_weights": source_loss_weights,
                    "weighted_mass_by_source": {
                        source: source_balance_counts[source] * weight
                        for source, weight in source_loss_weights.items()
                    },
                    "row_duplication": False,
                },
                "explicit_sample_weights": {
                    "enabled": bool(args.sample_weight_key),
                    **explicit_weight_summary,
                },
            },
            "dependencies": dependencies,
            "formal_test_reference_count": 0,
            "errors": [],
        }
        if args.dry_run:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            plan["report_hash"] = sha256_text(json.dumps(plan, sort_keys=True))
            audit_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"P0-A4 training dry-run passed: {display_path(audit_path)}")
            return 0
        missing = [name for name in ("accelerate", "deepspeed", "peft") if dependencies[name] == "missing"]
        if missing:
            raise TrainingError(
                f"Missing training dependencies {missing}; install requirements-p0a4.txt"
            )
        if not model_dir.is_dir():
            raise TrainingError(f"Missing model snapshot: {display_path(model_dir)}")

        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        class SourceBalancedTrainer(Trainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                sample_weight = inputs.pop("sample_weight", None)
                if sample_weight is None:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                token_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shift_labels)
                valid_tokens = shift_labels.ne(-100)
                per_example_loss = (
                    (token_loss * valid_tokens).sum(dim=1)
                    / valid_tokens.sum(dim=1).clamp_min(1)
                )
                loss = (
                    per_example_loss
                    * sample_weight.to(
                        device=per_example_loss.device,
                        dtype=per_example_loss.dtype,
                    )
                ).mean()
                return (loss, outputs) if return_outputs else loss

        # TrainingArguments must exist before from_pretrained. Its DeepSpeed integration
        # installs the global ZeRO-3 context that partitions the 14B parameters while
        # they are constructed and loaded. Creating it after the model would first copy
        # the complete BF16 model to every GPU and OOM before ZeRO-3 can take control.
        # DeepSpeed Teacher training has a coordinated best-model reload path. Plain
        # DDP + PEFT must not call load_adapter while DDP is still active; Student
        # training publishes the best saved adapter directly after Trainer returns.
        load_best_model_in_trainer = args.role == "teacher"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            learning_rate=learning_rate,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=None if args.external_checkpoint_selection else 2,
            load_best_model_at_end=load_best_model_in_trainer,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            # LoRA parameters are reused inside checkpointed transformer blocks. DDP's
            # unused-parameter traversal can register a second reduction hook for the
            # same adapter parameter, so keep the graph explicit and fixed.
            ddp_find_unused_parameters=False,
            seed=int(config["seed"]) + args.candidate_index,
            deepspeed=args.deepspeed or None,
        )
        if args.role == "teacher":
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            if not is_deepspeed_zero3_enabled():
                raise TrainingError("DeepSpeed ZeRO-3 was not active before Teacher model loading")

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        # PyTorch's reentrant checkpoint implementation is incompatible with LoRA
        # under multi-process DDP: the same adapter parameter can be marked ready by
        # more than one reentrant backward pass. The non-reentrant implementation
        # records a single autograd graph and is the supported path for this trainer.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(TARGET_MODULES),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        disable_thinking = args.role != "teacher"
        trainer_class = (
            SourceBalancedTrainer
            if args.source_balance_key or args.sample_weight_key
            else Trainer
        )
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=TokenizedDataset(
                train_rows,
                tokenizer,
                max_length,
                disable_thinking,
                source_balance_key=args.source_balance_key,
                source_loss_weights=source_loss_weights,
                sample_weight_key=args.sample_weight_key,
            ),
            eval_dataset=TokenizedDataset(validation_rows, tokenizer, max_length, disable_thinking),
            data_collator=SupervisedCollator(tokenizer),
        )
        is_main_process = trainer.is_world_process_zero()
        train_result = trainer.train()
        # Without load_best_model_at_end, non-zero ranks can leave Trainer before
        # rank 0 has finished creating the checkpoint directory. Synchronize first
        # and derive the path from the globally agreed best step on every rank.
        trainer.accelerator.wait_for_everyone()
        metrics = dict(train_result.metrics)
        best_checkpoint_path: Path | None = (
            Path(trainer.state.best_model_checkpoint)
            if trainer.state.best_model_checkpoint
            else None
        )
        published_adapter_files: list[str] = []
        if args.role == "teacher":
            metrics.update(trainer.evaluate())
            trainer.save_model(str(output_dir))
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(output_dir)
        elif args.external_checkpoint_selection:
            checkpoint_candidates = sorted(
                (
                    path
                    for path in output_dir.glob("checkpoint-*")
                    if path.is_dir()
                    and (path / "adapter_config.json").is_file()
                    and any(
                        (path / name).is_file()
                        for name in ("adapter_model.safetensors", "adapter_model.bin")
                    )
                ),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            if not checkpoint_candidates:
                raise TrainingError("External selection training produced no complete PEFT checkpoints")
            best_checkpoint_path = None
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(output_dir)
            metrics["external_checkpoint_candidate_count"] = len(checkpoint_candidates)
        else:
            if trainer.state.best_global_step is None:
                raise TrainingError("Student training did not produce a best PEFT checkpoint")
            best_checkpoint_path = output_dir / f"checkpoint-{trainer.state.best_global_step}"
            # Validate on every rank before entering the barrier. Rank 0 then copies
            # the already-selected PEFT files without touching the DDP-wrapped model.
            config_exists = (best_checkpoint_path / "adapter_config.json").is_file()
            weights_exist = any(
                (best_checkpoint_path / name).is_file()
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            )
            if not config_exists or not weights_exist:
                raise TrainingError(
                    f"Best checkpoint is incomplete: {display_path(best_checkpoint_path)}"
                )
            if trainer.is_world_process_zero():
                published_adapter_files = publish_best_adapter_checkpoint(
                    best_checkpoint_path, output_dir
                )
                tokenizer.save_pretrained(output_dir)
            if trainer.state.best_metric is not None:
                metrics["best_eval_loss"] = float(trainer.state.best_metric)
        trainer.accelerator.wait_for_everyone()
        plan.update(
            {
                "status": "passed",
                "metrics": metrics,
                "best_checkpoint": (
                    display_path(best_checkpoint_path) if best_checkpoint_path else ""
                ),
                "checkpoint_candidates": (
                    [display_path(path) for path in checkpoint_candidates]
                    if args.external_checkpoint_selection
                    else []
                ),
                "published_adapter_files": published_adapter_files,
                "deployment_ready": not args.external_checkpoint_selection,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            }
        )
        plan["report_hash"] = sha256_text(json.dumps(plan, sort_keys=True, default=str))
        if trainer.is_world_process_zero():
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.end_training()
    except (OSError, KeyError, ValueError, TrainingError, json.JSONDecodeError) as exc:
        print(f"P0-A4 training failed: {exc}", file=sys.stderr)
        return 1
    if is_main_process:
        print(f"P0-A4 training completed: {display_path(output_dir)}")
        print(f"Audit: {display_path(audit_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
