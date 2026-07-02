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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_utils import apply_lora_to_model, save_lora_adapter, trainable_parameters


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "Qwen--Qwen2.5-1.5B-Instruct"
DEFAULT_DISTILL = ROOT / "data" / "distill" / "distill_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "adapters" / "cedd_structured"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_cedd_structured_train.json"
SYSTEM_PROMPT = "You are DB4AI-EdgeServe. Return exactly one compact JSON object for edge decision distillation."


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_csv_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def select_rows(
    rows: list[dict[str, Any]],
    dataset_filter: set[str] | None,
    sample_limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if row.get("used_for_training") is True
        and (dataset_filter is None or str(row.get("dataset_key", "")) in dataset_filter)
    ]
    if not filtered:
        raise ValueError("No training rows selected")
    if sample_limit is None:
        return filtered

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        by_dataset.setdefault(str(row.get("dataset_key", "")), []).append(row)
    rng = random.Random(seed)
    for dataset_rows in by_dataset.values():
        rng.shuffle(dataset_rows)

    selected: list[dict[str, Any]] = []
    positions = {key: 0 for key in by_dataset}
    keys = sorted(by_dataset)
    while len(selected) < sample_limit:
        progressed = False
        for key in keys:
            pos = positions[key]
            if pos < len(by_dataset[key]):
                selected.append(by_dataset[key][pos])
                positions[key] += 1
                progressed = True
                if len(selected) >= sample_limit:
                    break
        if not progressed:
            break
    return selected


def target_text(row: dict[str, Any]) -> str:
    target = row.get("target_json", {})
    return json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def user_prompt(row: dict[str, Any]) -> str:
    return (
        "Create the structured edge decision JSON for this task.\n"
        f"task_type: {row.get('task_type')}\n"
        f"input: {row.get('input_text')}"
    )


class DistillDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        answer_text = target_text(row) + self.tokenizer.eos_token
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


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row_index, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[row_index, :length] = item["input_ids"]
        labels[row_index, :length] = item["labels"]
        attention_mask[row_index, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CEDD-Structured LoRA adapter on distill JSONL.")
    parser.add_argument("--student-init", "--student_init", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--distill-data", "--distill_data", default=str(DEFAULT_DISTILL))
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dataset", action="append", default=[], help="Dataset filter; repeat or comma-separate.")
    parser.add_argument("--sample-limit", "--sample_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-module", action="append", default=["q_proj", "v_proj"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Mark audit as smoke/pilot training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        print("--sample-limit must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        print("--batch-size and --grad-accum-steps must be positive", file=sys.stderr)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = resolve_path(args.student_init)
    distill_path = resolve_path(args.distill_data)
    output_dir = resolve_path(args.output_dir)
    audit_path = resolve_path(args.audit)
    dataset_filter = set(parse_csv_values(args.dataset)) if args.dataset else None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    all_rows = load_jsonl(distill_path)
    rows = select_rows(all_rows, dataset_filter, args.sample_limit, args.seed)
    dataset_counts = Counter(str(row.get("dataset_key", "")) for row in rows)
    selected_ids_hash = sha256_text("\n".join(str(row.get("sample_id", "")) for row in rows) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            input_embeddings = model.get_input_embeddings()

            def make_inputs_require_grad(_module: torch.nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)

    for param in model.parameters():
        param.requires_grad = False
    target_modules = tuple(parse_csv_values(args.target_module))
    replaced = apply_lora_to_model(
        model,
        target_modules,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )
    model.to(args.device)
    model.train()
    trainable_count, total_count = trainable_parameters(model)

    train_dataset = DistillDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
    )

    total_update_steps = int((len(loader) * args.epochs) / args.grad_accum_steps)
    if args.max_steps > 0:
        total_update_steps = min(total_update_steps or args.max_steps, args.max_steps)
    total_update_steps = max(total_update_steps, 1)

    started = time.perf_counter()
    losses: list[float] = []
    global_step = 0
    update_step = 0
    optimizer.zero_grad(set_to_none=True)
    while update_step < total_update_steps:
        for batch in loader:
            global_step += 1
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.grad_accum_steps
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step % args.log_every == 0 or update_step == 1:
                    mean_loss = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
                    print(
                        f"[TRAIN] update={update_step}/{total_update_steps} "
                        f"global_step={global_step} loss={mean_loss:.4f}",
                        flush=True,
                    )
                if update_step >= total_update_steps:
                    break

    elapsed = time.perf_counter() - started
    adapter_config = {
        "adapter_type": "manual_lora",
        "stage": "cedd_structured",
        "base_model": display_path(model_dir),
        "target_modules": list(target_modules),
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "dtype": args.dtype,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "selected_sample_count": len(rows),
        "selected_sample_ids_hash": selected_ids_hash,
        "distill_data_hash": sha256_file(distill_path),
    }
    save_lora_adapter(model, output_dir, adapter_config)
    adapter_hash = sha256_dir(output_dir)
    status = "passed" if losses and adapter_hash else "failed"
    audit = {
        "gate": "G-KD-TRACE-cedd-structured-train-smoke" if args.smoke else "G-KD-TRACE-cedd-structured-train",
        "check_version": "1.0",
        "created_by": "model_compression/train_cedd_structured.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "smoke": bool(args.smoke),
        "student_init": display_path(model_dir),
        "distill_data_path": display_path(distill_path),
        "distill_data_hash": sha256_file(distill_path),
        "output_dir": display_path(output_dir),
        "adapter_hash": adapter_hash,
        "adapter_config_hash": sha256_file(output_dir / "adapter_config.json"),
        "adapter_model_hash": sha256_file(output_dir / "adapter_model.pt"),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "selected_sample_count": len(rows),
        "selected_sample_ids_hash": selected_ids_hash,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(target_modules),
        "replaced_module_count": len(replaced),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_parameter_ratio": trainable_count / total_count if total_count else 0.0,
        "global_steps": global_step,
        "optimizer_steps": update_step,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "final_loss": losses[-1] if losses else None,
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
    if status != "passed":
        return 1
    print("CEDD-Structured training passed.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
