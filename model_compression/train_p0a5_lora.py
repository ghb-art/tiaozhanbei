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
    full_ids = list(full_ids)[:max_length]
    prompt_length = min(len(list(prompt_ids)), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(value != -100 for value in labels):
        raise TrainingError(f"Answer was fully truncated: {row.get('sample_id')}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sample_weight": float(row.get("training_weight", 1.0)),
        "preserve_math": bool(row.get("preserve_math", False)),
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
        math_config = dict(config["student_training"]["math_preservation"])
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
        audit = {
            "gate": "P0-A5-LORA-TRAIN",
            "check_version": "1.0",
            "created_by": "model_compression/train_p0a5_lora.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "dry_run_passed" if args.dry_run else "running",
            "role": args.role,
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
            },
            "math_preservation": math_config if args.role == "student" else {"enabled": False},
            "formal_test_reference_count": 0,
            "dependencies": dependencies,
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        class PreservationTrainer(Trainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                weights = inputs.pop("sample_weight")
                preserve_math = inputs.pop("preserve_math")
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                token_loss = functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shift_labels)
                valid = shift_labels.ne(-100)
                supervised = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                per_example = supervised
                if args.role == "student" and math_config.get("enabled") and preserve_math.any():
                    peft_model = model.module if hasattr(model, "module") else model
                    if not hasattr(peft_model, "disable_adapter"):
                        raise TrainingError("Math KL requires a PEFT model with disable_adapter")
                    with torch.no_grad(), peft_model.disable_adapter():
                        reference = model(**inputs).logits[..., :-1, :].contiguous()
                    temperature = float(math_config["temperature"])
                    kl_tokens = functional.kl_div(
                        functional.log_softmax(shift_logits / temperature, dim=-1),
                        functional.softmax(reference / temperature, dim=-1),
                        reduction="none",
                    ).sum(dim=-1) * (temperature * temperature)
                    kl = (kl_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
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

        load_best = args.role == "teacher"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=learning_rate,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
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
        trainer = PreservationTrainer(
            model=model,
            args=training_args,
            train_dataset=TokenizedRows(train_rows, tokenizer, max_length),
            eval_dataset=TokenizedRows(validation_rows, tokenizer, max_length),
            data_collator=Collator(tokenizer),
        )
        result = trainer.train()
        is_main = trainer.is_world_process_zero()
        if is_main:
            checkpoint = Path(trainer.state.best_model_checkpoint or "")
            if load_best:
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                published = sorted(path.name for path in output_dir.iterdir())
            else:
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
