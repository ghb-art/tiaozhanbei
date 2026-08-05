#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a5_capability.json"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FORBIDDEN_INPUT_ROOTS = (
    ROOT / "reports/sealed",
    ROOT / "data/eval",
    ROOT / "data/splits",
)
FORMAL_MARKERS = ("gsm8k/test/", "cmmlu/test/", "humaneval/", "official_full", "final_test")


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


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in ("torch", "transformers", "accelerate", "deepspeed", "peft"):
        if importlib.util.find_spec(module) is None:
            versions[module] = "missing"
            continue
        loaded = __import__(module)
        versions[module] = str(getattr(loaded, "__version__", "unknown"))
    return versions


def gradient_checkpointing_config(role: str) -> dict[str, bool]:
    if role not in {"teacher", "student"}:
        raise TrainingError(f"Unknown training role: {role}")
    return {
        "enabled": True,
        "use_reentrant": role == "teacher",
    }


def checkpoint_due(global_step: int, interval: int) -> bool:
    return interval > 0 and global_step > 0 and global_step % interval == 0


def synchronize_distributed_stop(
    local_stop: bool, torch_module: Any, device: Any
) -> bool:
    """Return the MAX-reduced early-stop decision across all initialized ranks."""
    if not torch_module.distributed.is_initialized():
        return bool(local_stop)
    stop = torch_module.tensor(
        int(local_stop), device=device, dtype=torch_module.int32
    )
    torch_module.distributed.all_reduce(
        stop, op=torch_module.distributed.ReduceOp.MAX
    )
    return bool(stop.item())


def early_stopping_config(settings: dict[str, Any]) -> dict[str, Any]:
    default_enabled = all(
        key in settings
        for key in ("eval_steps", "early_stopping_patience", "early_stopping_threshold")
    )
    if not bool(settings.get("early_stopping_enabled", default_enabled)):
        return {"enabled": False}
    eval_steps = int(settings.get("eval_steps", 0))
    patience = int(settings.get("early_stopping_patience", 0))
    threshold = float(settings.get("early_stopping_threshold", 0.0))
    if eval_steps <= 0 or patience <= 0 or threshold < 0:
        raise TrainingError(
            "Early stopping requires eval_steps > 0, patience > 0, "
            "and threshold >= 0"
        )
    return {
        "enabled": True,
        "metric": "eval_loss",
        "eval_steps": eval_steps,
        "patience": patience,
        "threshold": threshold,
    }


def validate_zero3(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or int(zero.get("stage", 0)) != 3:
        raise TrainingError("Teacher requires four-GPU DeepSpeed ZeRO-3")
    if "offload_optimizer" in zero or "offload_param" in zero:
        raise TrainingError("CPU parameter/optimizer offload is forbidden")
    return config


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingError(f"Missing data: {display_path(path)}")
    resolved = path.resolve()
    if any(resolved == root.resolve() or root.resolve() in resolved.parents for root in FORBIDDEN_INPUT_ROOTS):
        raise TrainingError(f"Evaluation/sealed data cannot train a model: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = " ".join(
                str(row.get(key, "")).casefold()
                for key in ("sample_id", "source", "split_role")
            )
            if any(marker in identity for marker in FORMAL_MARKERS):
                raise TrainingError(f"Formal-test reference at line {line_number}")
            if (
                not isinstance(row.get("messages"), list)
                or not str(row.get("answer", "")).strip()
                or str(row.get("split_role", "")) not in {"train", "internal_validation"}
            ):
                raise TrainingError(f"Invalid SFT row at line {line_number}")
            rows.append(row)
    if not rows:
        raise TrainingError(f"Empty data: {display_path(path)}")
    return rows


def render(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in row["messages"]
        if isinstance(item, dict) and item.get("role") in {"system", "user"}
    ]
    full_messages = messages + [{"role": "assistant", "content": str(row["answer"])}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if len(full_ids) > max_length:
        raise TrainingError(
            f"Sequence exceeds max length: {row.get('sample_id')} "
            f"{len(full_ids)} > {max_length}"
        )
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(value != -100 for value in labels):
        raise TrainingError(f"Answer was fully truncated: {row.get('sample_id')}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sample_weight": float(row.get("training_weight", 1.0)),
        "preserve_math": bool(row.get("preserve_math", False)),
        "kl_weight": float(row.get("kl_weight", 0.0)),
    }


def scan_token_budget(
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    maximum = 0
    maximum_sample = ""
    for row in [*train_rows, *validation_rows]:
        rendered = render(tokenizer, row, max_length)
        length = len(rendered["input_ids"])
        if length > maximum:
            maximum = length
            maximum_sample = str(row.get("sample_id", ""))
    return {
        "status": "passed",
        "scanned_rows": len(train_rows) + len(validation_rows),
        "max_seq_length": max_length,
        "maximum_observed_tokens": maximum,
        "maximum_sample_id": maximum_sample,
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
        batch: dict[str, Any] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "sample_weight": torch.tensor(
                [feature["sample_weight"] for feature in features], dtype=torch.float32
            ),
            "preserve_math": torch.tensor(
                [feature["preserve_math"] for feature in features], dtype=torch.bool
            ),
            "kl_weight": torch.tensor(
                [feature["kl_weight"] for feature in features], dtype=torch.float32
            ),
        }


def publish_adapter(checkpoint: Path, output: Path) -> list[str]:
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
        raise TrainingError(f"Checkpoint has no adapter: {display_path(checkpoint)}")
    output.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for source in (config_path, weights, checkpoint / "README.md"):
        if source.is_file():
            shutil.copy2(source, output / source.name)
            published.append(source.name)
    return published


def read_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not state_path.is_file() or not adapter_path.is_file():
        raise TrainingError(
            f"Incomplete checkpoint for evaluation: {display_path(checkpoint)}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state.get("global_step", -1))
    if global_step < 1 or checkpoint.name != f"checkpoint-{global_step}":
        raise TrainingError(
            f"Checkpoint step mismatch: {display_path(checkpoint)} "
            f"state={global_step}"
        )
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0-A5 shared Teacher/Student LoRA trainer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--role", choices=("teacher", "student"), required=True)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--model-dir")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--deepspeed", default="")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    is_main = True
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.candidate_index not in (1, 2):
            raise TrainingError("Only preregistered candidates 1 and 2 are allowed")
        if args.role == "teacher":
            settings = dict(config["teacher_training"])
            model_dir = resolve_path(
                args.model_dir or "models/pretrained/Qwen--Qwen2.5-14B-Instruct"
            )
            train_path = resolve_path(
                args.train_data or config["artifacts"]["source_train"]
            )
            zero_path = resolve_path(args.deepspeed) if args.deepspeed else None
            if zero_path is None:
                raise TrainingError("Teacher training requires --deepspeed")
            zero_config = validate_zero3(zero_path)
        else:
            settings = dict(config["student_training"])
            override = dict(settings.get("candidate_overrides", {}).get(str(args.candidate_index), {}))
            settings.update(override)
            model_dir = resolve_path(args.model_dir or config["policy"]["student_base"])
            train_path = resolve_path(
                args.train_data or config["artifacts"]["distill_train"]
            )
            zero_path = None
            zero_config = None
        validation_path = resolve_path(
            args.validation_data or config["artifacts"]["source_validation"]
        )
        output_dir = resolve_path(args.output_dir)
        audit_path = resolve_path(args.audit)
        train_rows = [row for row in read_rows(train_path) if row["split_role"] == "train"]
        validation_rows = [
            row
            for row in read_rows(validation_path)
            if row["split_role"] == "internal_validation"
        ]
        train_counts = Counter(str(row["dataset_key"]) for row in train_rows)
        validation_counts = Counter(str(row["dataset_key"]) for row in validation_rows)
        expected_tasks = {"gsm8k", "opencodeinstruct", "cmmlu"}
        if set(train_counts) != expected_tasks or set(validation_counts) != expected_tasks:
            raise TrainingError(
                f"Training tasks changed: train={train_counts}, validation={validation_counts}"
            )
        rank = int(settings["lora_rank"])
        alpha = int(settings["lora_alpha"])
        dropout = float(settings["lora_dropout"])
        learning_rate = float(settings["learning_rate"])
        epochs = float(settings["epochs"])
        max_length = int(settings["max_seq_length"])
        checkpoint_steps = int(settings.get("checkpoint_steps", 0))
        if checkpoint_steps < 0:
            raise TrainingError("checkpoint_steps must be non-negative")
        early_stopping = early_stopping_config(settings)
        resume_checkpoint = (
            resolve_path(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
        if resume_checkpoint is not None and not resume_checkpoint.is_dir():
            raise TrainingError(
                f"Missing resume checkpoint: {display_path(resume_checkpoint)}"
            )
        if args.evaluate_only:
            if resume_checkpoint is None:
                raise TrainingError("--evaluate-only requires --resume-from-checkpoint")
            checkpoint_state = read_checkpoint_state(resume_checkpoint)
        else:
            checkpoint_state = {}
        math_config = dict(config["student_training"]["math_preservation"])
        base_preservation = dict(
            settings.get(
                "base_preservation",
                {"enabled": False, "temperature": 1.0},
            )
        )
        checkpointing = gradient_checkpointing_config(args.role)
        if args.role == "student" and "math_kl_weight" in settings:
            math_config["kl_weight"] = float(settings["math_kl_weight"])
            math_config["supervised_weight"] = 1.0 - float(settings["math_kl_weight"])
        if args.role == "student":
            weight_mass = Counter()
            for row in train_rows:
                weight = float(row.get("training_weight", 0.0))
                if not math.isfinite(weight) or weight <= 0:
                    raise TrainingError("Student row has invalid training_weight")
                weight_mass[str(row["dataset_key"])] += weight
            total_mass = sum(weight_mass.values())
            observed_mass = {key: weight_mass[key] / total_mass for key in expected_tasks}
            target_mass = {
                key: float(value)
                for key, value in config["student_training"]["task_loss_mass"].items()
            }
            if any(abs(observed_mass[key] - target_mass[key]) > 1e-6 for key in expected_tasks):
                raise TrainingError(
                    f"Student weighted task mass changed: observed={observed_mass}, target={target_mass}"
                )
        else:
            observed_mass = {}
        dependencies = dependency_versions()
        token_budget_scan: dict[str, Any] = {}
        if args.dry_run:
            if dependencies.get("transformers") == "missing":
                raise TrainingError("transformers is required for token-budget preflight")
            if not model_dir.is_dir():
                raise TrainingError(f"Missing model directory: {display_path(model_dir)}")
            from transformers import AutoTokenizer

            dry_run_tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True,
            )
            token_budget_scan = scan_token_budget(
                dry_run_tokenizer,
                train_rows,
                validation_rows,
                max_length,
            )
        audit = {
            "gate": "P0-A5-LORA-TRAIN",
            "check_version": "1.0",
            "created_by": "model_compression/train_p0a5_lora.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "dry_run_passed" if args.dry_run else "running",
            "role": args.role,
            "mode": "checkpoint_evaluation" if args.evaluate_only else "training",
            "candidate_index": args.candidate_index,
            "config": display_path(config_path),
            "config_hash": sha256_file(config_path),
            "model_dir": display_path(model_dir),
            "train_data": display_path(train_path),
            "train_data_hash": sha256_file(train_path),
            "validation_data": display_path(validation_path),
            "validation_data_hash": sha256_file(validation_path),
            "train_counts": dict(sorted(train_counts.items())),
            "validation_counts": dict(sorted(validation_counts.items())),
            "weighted_task_mass": observed_mass,
            "lora": {
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": TARGET_MODULES,
            },
            "optimization": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "max_seq_length": max_length,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "deepspeed": display_path(zero_path) if zero_path else "",
                "deepspeed_stage": (
                    int(zero_config["zero_optimization"]["stage"]) if zero_config else 0
                ),
                "cpu_offload": False,
                "gradient_checkpointing": checkpointing["enabled"],
                "gradient_checkpointing_use_reentrant": checkpointing["use_reentrant"],
                "checkpoint_steps": checkpoint_steps,
                "save_total_limit": 2,
                "resume_from_checkpoint": (
                    display_path(resume_checkpoint) if resume_checkpoint else ""
                ),
                "early_stopping": early_stopping,
                "weight_decay": float(settings.get("weight_decay", 0.0)),
                "warmup_ratio": float(settings.get("warmup_ratio", 0.03)),
                "max_grad_norm": float(settings.get("max_grad_norm", 1.0)),
                "label_smoothing": float(settings.get("label_smoothing", 0.0)),
            },
            "math_preservation": math_config if args.role == "student" else {"enabled": False},
            "base_preservation": (
                base_preservation if args.role == "student" else {"enabled": False}
            ),
            "formal_test_reference_count": 0,
            "dependencies": dependencies,
            "token_budget_scan": token_budget_scan,
            "errors": [],
        }
        if args.dry_run:
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True)
            )
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"P0-A5 training dry-run passed: {display_path(audit_path)}")
            return 0
        missing = [
            module
            for module in ("accelerate", "peft")
            if dependencies.get(module) == "missing"
        ]
        if args.role == "teacher" and dependencies.get("deepspeed") == "missing":
            missing.append("deepspeed")
        if missing:
            raise TrainingError(f"Missing training dependencies: {missing}")
        if not model_dir.is_dir():
            raise TrainingError(f"Missing model directory: {display_path(model_dir)}")

        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        class FixedStepCheckpointCallback(TrainerCallback):
            def __init__(self, interval: int):
                self.interval = interval

            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                del args, kwargs
                if checkpoint_due(state.global_step, self.interval):
                    control.should_save = True
                return control

        class ResumeMetricBaselineCallback(TrainerCallback):
            def __init__(self, checkpoint: Path):
                self.checkpoint = checkpoint

            def on_evaluate(
                self,
                args: Any,
                state: Any,
                control: Any,
                metrics: dict[str, Any],
                **kwargs: Any,
            ) -> Any:
                del args, kwargs
                if state.best_metric is None and "eval_loss" in metrics:
                    state.best_metric = float(metrics["eval_loss"])
                    state.best_global_step = int(state.global_step)
                    state.best_model_checkpoint = str(self.checkpoint)
                return control

        class PreservationTrainer(Trainer):
            def _save_checkpoint(self, model: Any, trial: Any) -> None:
                for callback in self.callback_handler.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        self.state.stateful_callbacks.setdefault(
                            callback.__class__.__name__,
                            callback.state(),
                        )
                super()._save_checkpoint(model, trial)

            def _maybe_log_save_evaluate(
                self,
                tr_loss: Any,
                grad_norm: Any,
                model: Any,
                trial: Any,
                epoch: Any,
                ignore_keys_for_eval: Any,
                start_time: Any,
                learning_rate: Any = None,
            ) -> None:
                super()._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    model,
                    trial,
                    epoch,
                    ignore_keys_for_eval,
                    start_time,
                    learning_rate,
                )
                # EarlyStoppingCallback updates TrainerControl independently on
                # every DDP rank. A rank-local difference here can make one rank
                # enter the terminal barrier while another starts the next DDP
                # forward, producing an ALLREDUCE/ALLGATHER mismatch. Reduce the
                # stop flag so all ranks leave at the same checkpoint boundary.
                if early_stopping["enabled"]:
                    self.control.should_training_stop = synchronize_distributed_stop(
                        self.control.should_training_stop,
                        torch,
                        self.accelerator.device,
                    )

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                weights = inputs.pop("sample_weight")
                preserve_math = inputs.pop("preserve_math")
                kl_weights = inputs.pop("kl_weight")
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                valid = shift_labels.ne(-100)
                if not bool(valid.any()):
                    raise TrainingError("Batch has no assistant answer tokens")
                batch_size = int(shift_labels.size(0))
                batch_indices = (
                    torch.arange(batch_size, device=shift_labels.device)
                    .unsqueeze(1)
                    .expand_as(shift_labels)[valid]
                )
                token_loss = functional.cross_entropy(
                    shift_logits[valid].float(),
                    shift_labels[valid],
                    reduction="none",
                    label_smoothing=float(settings.get("label_smoothing", 0.0)),
                )
                supervised_sum = torch.zeros(
                    batch_size, device=token_loss.device, dtype=token_loss.dtype
                )
                supervised_count = torch.zeros_like(supervised_sum)
                supervised_sum.scatter_add_(0, batch_indices, token_loss)
                supervised_count.scatter_add_(
                    0, batch_indices, torch.ones_like(token_loss)
                )
                supervised = supervised_sum / supervised_count.clamp_min(1)
                per_example = supervised
                if args.role == "student" and base_preservation.get("enabled"):
                    if bool((kl_weights < 0).any() or (kl_weights >= 1).any()):
                        raise TrainingError("All-task KL weights must be in [0, 1)")
                    peft_model = model.module if hasattr(model, "module") else model
                    if not hasattr(peft_model, "disable_adapter"):
                        raise TrainingError(
                            "All-task KL requires a PEFT model with disable_adapter"
                        )
                    # Every rank performs this reference forward for every batch.
                    # That invariant prevents the mixed-domain DDP deadlock caused
                    # by the former conditional Math-only reference pass.
                    with torch.no_grad(), peft_model.disable_adapter():
                        reference = peft_model(**inputs).logits[..., :-1, :].contiguous()
                    temperature = float(base_preservation.get("temperature", 2.0))
                    kl_tokens = functional.kl_div(
                        functional.log_softmax(
                            shift_logits[valid].float() / temperature, dim=-1
                        ),
                        functional.softmax(
                            reference[valid].float() / temperature, dim=-1
                        ),
                        reduction="none",
                    ).sum(dim=-1) * (temperature * temperature)
                    kl_sum = torch.zeros_like(supervised)
                    kl_sum.scatter_add_(0, batch_indices, kl_tokens)
                    kl = kl_sum / supervised_count.clamp_min(1)
                    local_kl = kl_weights.to(
                        device=supervised.device, dtype=supervised.dtype
                    )
                    per_example = (1.0 - local_kl) * supervised + local_kl * kl
                elif args.role == "student" and math_config.get("enabled") and preserve_math.any():
                    peft_model = model.module if hasattr(model, "module") else model
                    if not hasattr(peft_model, "disable_adapter"):
                        raise TrainingError("Math KL requires a PEFT model with disable_adapter")
                    with torch.no_grad(), peft_model.disable_adapter():
                        # Run the conditional reference pass on the local PEFT module.
                        # Calling the DDP wrapper here deadlocks when different ranks
                        # receive different task types and only some ranks enter KL.
                        reference = peft_model(**inputs).logits[..., :-1, :].contiguous()
                    temperature = float(math_config["temperature"])
                    kl_tokens = functional.kl_div(
                        functional.log_softmax(shift_logits[valid] / temperature, dim=-1),
                        functional.softmax(reference[valid] / temperature, dim=-1),
                        reduction="none",
                    ).sum(dim=-1) * (temperature * temperature)
                    kl_sum = torch.zeros_like(supervised)
                    kl_sum.scatter_add_(0, batch_indices, kl_tokens)
                    kl = kl_sum / supervised_count.clamp_min(1)
                    combined_math = (
                        float(math_config["supervised_weight"]) * supervised
                        + float(math_config["kl_weight"]) * kl
                    )
                    per_example = torch.where(preserve_math, combined_math, supervised)
                loss = (
                    per_example
                    * weights.to(device=per_example.device, dtype=per_example.dtype)
                ).mean()
                return (loss, outputs) if return_outputs else loss

        configured_eval_steps = (
            int(early_stopping["eval_steps"]) if early_stopping["enabled"] else None
        )
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=float(settings.get("weight_decay", 0.0)),
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=float(settings.get("warmup_ratio", 0.03)),
            max_grad_norm=float(settings.get("max_grad_norm", 1.0)),
            bf16=True,
            logging_steps=10,
            eval_strategy=(
                "no"
                if args.evaluate_only
                else ("steps" if early_stopping["enabled"] else "epoch")
            ),
            eval_steps=configured_eval_steps,
            eval_on_start=bool(
                not args.evaluate_only and early_stopping["enabled"] and resume_checkpoint
            ),
            save_strategy=(
                "no"
                if args.evaluate_only
                else ("steps" if early_stopping["enabled"] else "epoch")
            ),
            save_steps=configured_eval_steps or 500,
            save_total_limit=2,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=(
                checkpointing["enabled"] and not args.evaluate_only
            ),
            gradient_checkpointing_kwargs={
                "use_reentrant": checkpointing["use_reentrant"]
            },
            restore_callback_states_from_checkpoint=True,
            seed=int(config["seed"]) + args.candidate_index,
            deepspeed=args.deepspeed or None,
        )
        if args.role == "teacher":
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            if not is_deepspeed_zero3_enabled():
                raise TrainingError("ZeRO-3 was not active before Teacher model loading")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
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
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=list(TARGET_MODULES),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        if checkpointing["use_reentrant"]:
            model.enable_input_require_grads()
        callbacks: list[Any] = []
        if checkpoint_steps and not args.evaluate_only:
            callbacks.append(FixedStepCheckpointCallback(checkpoint_steps))
        if early_stopping["enabled"] and not args.evaluate_only:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(early_stopping["patience"]),
                    early_stopping_threshold=float(early_stopping["threshold"]),
                )
            )
        if (
            resume_checkpoint is not None
            and early_stopping["enabled"]
            and not args.evaluate_only
        ):
            callbacks.append(ResumeMetricBaselineCallback(resume_checkpoint))
        trainer = PreservationTrainer(
            model=model,
            args=training_args,
            train_dataset=TokenizedRows(train_rows, tokenizer, max_length),
            eval_dataset=TokenizedRows(validation_rows, tokenizer, max_length),
            data_collator=Collator(tokenizer),
            callbacks=callbacks or None,
        )
        if args.evaluate_only:
            trainer._load_from_checkpoint(str(resume_checkpoint), model=model)
            metrics = trainer.evaluate()
            is_main = trainer.is_world_process_zero()
            if is_main:
                audit.update(
                    {
                        "status": "passed",
                        "checkpoint": display_path(resume_checkpoint),
                        "checkpoint_step": int(checkpoint_state["global_step"]),
                        "checkpoint_state_hash": sha256_file(
                            resume_checkpoint / "trainer_state.json"
                        ),
                        "adapter_hash": sha256_file(
                            resume_checkpoint / "adapter_model.safetensors"
                        ),
                        "evaluation_metrics": metrics,
                        "evaluated_rows": len(validation_rows),
                    }
                )
                audit["report_hash"] = sha256_text(
                    json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str)
                )
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                print(
                    "P0-A5 checkpoint evaluation completed: "
                    f"{display_path(resume_checkpoint)}"
                )
                print(f"eval_loss={metrics.get('eval_loss')}")
                print(f"Audit: {display_path(audit_path)}")
            return 0
        result = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
        is_main = trainer.is_world_process_zero()
        if is_main:
            checkpoint = Path(trainer.state.best_model_checkpoint or "")
            if not checkpoint.is_dir():
                checkpoints = sorted(
                    output_dir.glob("checkpoint-*"),
                    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
                )
                if not checkpoints:
                    raise TrainingError("Student training produced no checkpoint")
                checkpoint = checkpoints[-1]
            published = publish_adapter(checkpoint, output_dir)
            tokenizer.save_pretrained(output_dir)
            audit.update(
                {
                    "status": "passed",
                    "best_checkpoint": display_path(checkpoint) if checkpoint else "",
                    "published_files": published,
                    "train_metrics": result.metrics,
                    "global_step": trainer.state.global_step,
                    "best_metric": trainer.state.best_metric,
                }
            )
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str)
            )
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"P0-A5 training completed: {display_path(output_dir)}")
            print(f"Audit: {display_path(audit_path)}")
        return 0
    except (TrainingError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if is_main:
            print(f"P0-A5 training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
