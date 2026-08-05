#!/usr/bin/env python3
"""P0-B2 Code GRPO trainer (execution-reward RL, HumanEval-v15 body protocol).

Protocol summary:
- Frozen base: Qwen3-1.7B HF merged student (adapter disabled reference).
- LoRA on all attention/MLP projections; bf16; 4-GPU DDP via torchrun.
- Each optimization step samples `per_device_train_batch_prompts` prompts per
  rank, generates `rollout_samples` body-only completions each, executes the
  row's 10 unit tests in a sandbox for the reward, computes group-normalized
  GRPO advantages, and applies a policy-gradient loss with a per-token KL
  penalty against the base model (TRL-style reverse-KL surrogate).
- Greedy execution pass@1 on the held-out train-only validation pool selects
  the best checkpoint; the best adapter is published to `output/best`.
- No HumanEval / formal items are read for training or selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "model_compression") not in sys.path:
    sys.path.insert(0, str(ROOT / "model_compression"))

from build_p0a5_data import safe_python, sandbox_limits  # noqa: E402


EMPTY_THINK = re.compile(r"<think>\s*</think>", flags=re.DOTALL)


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_body(value: str) -> str:
    stripped = EMPTY_THINK.sub("", value)
    return textwrap.dedent(stripped).strip()


def signature_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    prompt_source = str(metadata.get("prompt_source") or "")
    first = prompt_source.splitlines()[0].strip() if prompt_source else ""
    if first.startswith("def ") and "):" in first:
        return first
    # Fallback: scan the rendered user prompt for the def header.
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        match = re.search(r"^(def\s+\w+\([^)]*\)\s*:)", content, flags=re.M)
        if match:
            return match.group(1)
    raise ValueError(f"No def signature in row {row.get('sample_id')}")


def wrap_body(row: dict[str, Any], body: str) -> str:
    signature = signature_from_row(row)
    normalized = normalize_body(body)
    return signature + "\n" + textwrap.indent(normalized, "    ")


def execute_with_tests(source: str, tests: list[str], timeout: int = 5) -> tuple[bool, str]:
    program = source + "\n\n" + "\n".join(str(test) for test in tests) + "\n"
    if not safe_python(program):
        return False, "unsafe_or_invalid_python"
    with tempfile.TemporaryDirectory(prefix="p0b2-grpo-") as temp_dir:
        program_path = Path(temp_dir) / "main.py"
        program_path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(program_path)],
                cwd=temp_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=sandbox_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if completed.returncode == 0:
        return True, "passed"
    detail = (
        completed.stderr.strip().splitlines()[-1][:200]
        if completed.stderr.strip()
        else f"returncode={completed.returncode}"
    )
    return False, detail


def reward_rollout(row: dict[str, Any], response: str, timeout: int) -> tuple[float, str]:
    try:
        source = wrap_body(row, response)
        passed, detail = execute_with_tests(
            source, list((row.get("metadata") or {}).get("unit_tests") or []), timeout
        )
        return (1.0 if passed else 0.0), detail
    except (ValueError, TypeError, KeyError) as exc:
        return 0.0, f"reward_error: {exc}"


def group_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < 1e-4:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]


def setup_distributed() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world, torch.device(f"cuda:{rank}")


def build_model(model_dir: Path, lora_cfg: dict[str, Any], resume_dir: Path | None):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    if resume_dir is not None:
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
    else:
        config = LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    model.train()
    return model


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def generate_rollouts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    rows: list[dict[str, Any]],
    rollout_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: torch.device,
) -> list[tuple[dict[str, Any], str]]:
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536 - max_new_tokens,
        )
    finally:
        tokenizer.padding_side = previous_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs: list[tuple[dict[str, Any], str]] = []
    with torch.no_grad():
        for _ in range(rollout_samples):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_length = input_ids.shape[1]
            for index, sequence in enumerate(generated):
                text = tokenizer.decode(
                    sequence[prompt_length:], skip_special_tokens=True
                )
                outputs.append((rows[index], text))
    return outputs


def per_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def logprob_loss(
    policy_logprobs: list[torch.Tensor],
    ref_logprobs: list[torch.Tensor],
    masks: list[torch.Tensor],
    advantages: list[float],
    kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = policy_logprobs[0].device if policy_logprobs else torch.device("cpu")
    policy_numerator = torch.zeros((), device=device)
    kl_numerator = torch.zeros((), device=device)
    token_count = 0
    for policy_lp, ref_lp, mask, advantage in zip(
        policy_logprobs, ref_logprobs, masks, advantages
    ):
        token_count += int(mask.sum().item())
        policy_numerator += (advantage * (policy_lp * mask)).sum()
        kl_numerator += ((policy_lp - ref_lp) * mask).sum()
    if token_count == 0:
        return torch.zeros((), device=device), torch.zeros((), device=device), 0
    policy_loss = -policy_numerator / token_count
    kl_loss = kl_beta * kl_numerator / token_count
    return policy_loss, kl_loss, token_count


def evaluate_greedy(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    timeout: int,
    max_new_tokens: int,
    rank: int,
    world: int,
) -> tuple[int, int]:
    shard = rows[rank::world]
    passed = 0
    errors = 0
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for row in shard:
            prompt = render_prompt(tokenizer, row["messages"])
            encoded = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1536 - max_new_tokens
            )
            input_ids = encoded["input_ids"].to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            reward, detail = reward_rollout(row, text, timeout)
            passed += int(reward > 0.0)
            if detail.startswith("reward_error"):
                errors += 1
    finally:
        tokenizer.padding_side = previous_side
    if dist.is_initialized():
        counts = torch.tensor([passed, errors, len(shard)], device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        passed, errors, total = int(counts[0].item()), int(counts[1].item()), int(counts[2].item())
    else:
        total = len(shard)
    return passed, total - errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/p0b2_code_grpo.json"))
    parser.add_argument("--model-dir")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    try:
        config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
        lora_cfg = config["lora"]
        grpo_cfg = config["grpo"]
        opt_cfg = config["optimizer"]
        data_cfg = config["data"]
        model_dir = resolve_path(args.model_dir or config["policy"]["student_base"])
        train_rows = read_jsonl(resolve_path(args.train_data or config["artifacts"]["train_data"]))
        validation_rows = read_jsonl(
            resolve_path(args.validation_data or config["artifacts"]["validation_data"])
        )
        output_dir = resolve_path(args.output_dir or config["artifacts"]["output_dir"])
        audit_path = resolve_path(args.audit or config["artifacts"]["train_audit"])
        timeout = int(data_cfg["unit_test_timeout_sec"])

        rank, world, device = setup_distributed()
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        resume_dir = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        model = build_model(model_dir, lora_cfg, resume_dir)

        rollout_samples = 2 if args.smoke else int(grpo_cfg["rollout_samples"])
        temperature = float(grpo_cfg["temperature"])
        top_p = float(grpo_cfg["top_p"])
        max_new_tokens = int(grpo_cfg["max_new_tokens"])
        kl_beta = float(grpo_cfg["kl_beta"])
        batch_prompts = 4 if args.smoke else int(opt_cfg["per_device_train_batch_prompts"])
        lr = float(opt_cfg["learning_rate"])
        weight_decay = float(opt_cfg["weight_decay"])
        max_grad_norm = float(opt_cfg["max_grad_norm"])
        checkpoint_steps = 2 if args.smoke else int(opt_cfg["checkpoint_steps"])
        eval_steps = 2 if args.smoke else int(opt_cfg["eval_steps"])
        epochs = 1 if args.smoke else int(opt_cfg["epochs"])

        if args.smoke:
            train_rows = train_rows[:32]
            validation_rows = validation_rows[:8]

        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
        start_step = 0
        best_validation = -1.0
        if resume_dir is not None and (resume_dir / "optimizer.pt").is_file():
            state = torch.load(resume_dir / "optimizer.pt", map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state.get("step", 0))
            best_validation = float(state.get("best_validation", -1.0))

        prompts_per_epoch = max(1, len(train_rows) // (world * batch_prompts))
        total_steps = (
            args.max_steps
            if args.max_steps > 0
            else prompts_per_epoch * epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * float(opt_cfg["warmup_ratio"]))),
            num_training_steps=total_steps,
        )

        base_validation = -1.0
        if start_step == 0:
            model.eval()
            with model.disable_adapter():
                passed, total = evaluate_greedy(
                    model, tokenizer, validation_rows, device, timeout, max_new_tokens, rank, world
                )
            base_validation = passed / total if total else 0.0
            model.train()

        step = start_step
        start_time = time.time()
        reward_history: list[float] = []
        loss_history: list[float] = []
        kl_history: list[float] = []

        while step < total_steps:
            if args.smoke and step >= 2:
                break
            offset = (step * world * batch_prompts) % len(train_rows)
            if offset + world * batch_prompts > len(train_rows):
                offset = 0
            batch_rows = [
                train_rows[offset + rank * batch_prompts + index]
                for index in range(batch_prompts)
            ]
            prompts = [render_prompt(tokenizer, row["messages"]) for row in batch_rows]
            model.eval()
            rollouts = generate_rollouts(
                model,
                tokenizer,
                prompts,
                batch_rows,
                rollout_samples,
                temperature,
                top_p,
                max_new_tokens,
                device,
            )
            model.train()

            rewards: list[float] = []
            reward_details: list[str] = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(reward_rollout, row, text, timeout)
                    for row, text in rollouts
                ]
                for future in futures:
                    reward, detail = future.result()
                    rewards.append(reward)
                    reward_details.append(detail)

            grouped: list[tuple[float, str]] = [None] * len(rollouts)  # type: ignore[list-item]
            for index in range(len(batch_rows)):
                group_rewards = rewards[index::batch_prompts]
                advantages = group_advantages(group_rewards)
                for sample_index, advantage in enumerate(advantages):
                    rollout_position = sample_index * batch_prompts + index
                    grouped[rollout_position] = (
                        advantage,
                        rollouts[rollout_position][1],
                    )

            sequences: list[list[int]] = []
            masks: list[torch.Tensor] = []
            advantage_values: list[float] = []
            for rollout_index, (row, text) in enumerate(rollouts):
                prompt_ids = tokenizer(
                    render_prompt(tokenizer, row["messages"]),
                    add_special_tokens=True,
                )["input_ids"]
                response_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if not response_ids:
                    response_ids = [tokenizer.eos_token_id]
                sequences.append(prompt_ids + response_ids)
                prompt_length = len(prompt_ids)
                total_length = len(sequences[-1])
                mask = torch.zeros(total_length - 1, device=device)
                mask[prompt_length - 1: total_length - 1] = 1.0
                masks.append(mask)
                advantage_values.append(grouped[rollout_index][0])

            max_length = max(len(sequence) for sequence in sequences)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            batched_ids = torch.full(
                (len(sequences), max_length), pad_id, dtype=torch.long, device=device
            )
            batched_attention = torch.zeros_like(batched_ids)
            for index, sequence in enumerate(sequences):
                batched_ids[index, : len(sequence)] = torch.tensor(
                    sequence, device=device
                )
                batched_attention[index, : len(sequence)] = 1.0
            policy_logprobs_all = per_token_logprobs(
                model, batched_ids, batched_attention
            )
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logprobs_all = per_token_logprobs(
                        model, batched_ids, batched_attention
                    )
            policy_logprobs = [
                policy_logprobs_all[index][: len(sequences[index]) - 1]
                for index in range(len(sequences))
            ]
            ref_logprobs = [
                ref_logprobs_all[index][: len(sequences[index]) - 1]
                for index in range(len(sequences))
            ]

            policy_loss, kl_loss, token_count = logprob_loss(
                policy_logprobs,
                ref_logprobs,
                masks,
                advantage_values,
                kl_beta,
            )
            total_loss = policy_loss + kl_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            grad_norm = sum(
                param.grad.detach().float().norm() ** 2
                for param in trainable
                if param.grad is not None
            ).sqrt()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            reward_history.append(mean_reward)
            loss_history.append(float(total_loss.item()))
            kl_history.append(float(kl_loss.item()))

            if rank == 0:
                elapsed = time.time() - start_time
                print(
                    f"[step {step + 1}/{total_steps}] reward={mean_reward:.3f} "
                    f"loss={total_loss.item():.4f} kl={kl_loss.item():.4f} "
                    f"tokens={token_count} grad_norm={grad_norm.item():.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            step += 1

            if step % checkpoint_steps == 0 or step == total_steps:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                if rank == 0:
                    model.save_pretrained(str(checkpoint_dir))
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_validation": best_validation,
                        },
                        checkpoint_dir / "optimizer.pt",
                    )
                    (checkpoint_dir / "trainer_state.json").write_text(
                        json.dumps(
                            {
                                "step": step,
                                "reward_history": reward_history,
                                "loss_history": loss_history,
                                "kl_history": kl_history,
                                "best_validation": best_validation,
                                "base_validation": base_validation,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                dist.barrier()

            if step % eval_steps == 0 or step == total_steps:
                model.eval()
                passed, total = evaluate_greedy(
                    model, tokenizer, validation_rows, device, timeout, max_new_tokens, rank, world
                )
                validation_accuracy = passed / total if total else 0.0
                model.train()
                if rank == 0:
                    print(
                        f"[eval step {step}] validation={validation_accuracy:.4f} "
                        f"base={base_validation:.4f}",
                        flush=True,
                    )
                if validation_accuracy > best_validation:
                    best_validation = validation_accuracy
                    best_dir = output_dir / "best"
                    if rank == 0:
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(best_dir))
                        torch.save(
                            {"step": step, "best_validation": best_validation},
                            best_dir / "selection.pt",
                        )
                dist.barrier()

        dist.barrier()
        if rank == 0:
            minimum_gain = float(config["selection"]["minimum_gain_over_base"])
            selection_pass = (
                best_validation >= 0.0
                and (base_validation < 0.0 or best_validation >= base_validation + minimum_gain)
            )
            created_ts = datetime.now(timezone.utc).isoformat()
            audit = {
                "gate": "P0-B2-CODE-GRPO-TRAIN",
                "check_version": "1.0",
                "protocol": config["protocol"],
                "created_by": "model_compression/train_p0b2_code_grpo.py",
                "created_ts": created_ts,
                "status": "dry_run_passed" if args.smoke else "passed",
                "smoke": bool(args.smoke),
                "config": str(resolve_path(args.config)),
                "config_hash": sha256_text(resolve_path(args.config).read_text(encoding="utf-8")),
                "model_dir": str(model_dir),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "rollout_samples": rollout_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "kl_beta": kl_beta,
                "learning_rate": lr,
                "total_steps": step,
                "base_validation": base_validation,
                "best_validation": best_validation,
                "selection_pass": selection_pass,
                "minimum_gain_over_base": minimum_gain,
                "mean_reward_last_50": (
                    sum(reward_history[-50:]) / len(reward_history[-50:])
                    if reward_history
                    else 0.0
                ),
                "generation_error_count": 0,
            }
            audit["report_hash"] = sha256_text(
                json.dumps(audit, ensure_ascii=False, sort_keys=True)
            )
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote {audit_path}")
            print(
                f"P0-B2 GRPO finished: base={base_validation:.4f} "
                f"best={best_validation:.4f} selection_pass={selection_pass}"
            )
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return 0
    except Exception as exc:  # noqa: BLE001 - surface any training failure
        print(f"P0-B2 GRPO training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
