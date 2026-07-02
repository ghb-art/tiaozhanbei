from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPRESSION = ROOT / "model_compression"
if str(MODEL_COMPRESSION) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPRESSION))

from run_student_probe import load_local_student, parse_comma_values, sha256_file, sha256_text, write_json, write_jsonl


DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "Qwen--Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_TRACE = ROOT / "reports" / "audit" / "chapter2_capability_eval_smoke.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_chapter2_capability_eval_smoke.json"
SYSTEM_PROMPT = "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested."


class CapabilityEvalError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def extract_gsm8k_reference(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.rsplit("####", 1)[1])
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_gsm8k_prediction(text: str) -> str:
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        matches = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
        if matches:
            return normalize_number(matches[0])
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return normalize_number(matches[-1]) if matches else ""


def extract_choice(text: str) -> str:
    stripped = text.strip().upper()
    match = re.search(r"(?:答案|ANSWER|选项|OPTION)?\s*[:：]?\s*([ABCD])\b", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", stripped)
    return match.group(1) if match else ""


def clean_code_completion(text: str) -> str:
    stripped = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    stripped = re.sub(r"^python\s+", "", stripped, flags=re.IGNORECASE).strip()
    return stripped


def run_humaneval_check(problem: dict[str, Any], completion: str, timeout_sec: float) -> tuple[bool, str]:
    entry_point = str(problem["entry_point"])
    completion = clean_code_completion(completion)
    if f"def {entry_point}" in completion:
        candidate_source = completion
    else:
        candidate_source = str(problem["prompt"]) + completion
    check_source = f"{candidate_source}\n{problem['test']}\ncheck({entry_point})\n"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", check_source],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if completed.returncode == 0:
        return True, "passed"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1][:240] if detail else f"returncode={completed.returncode}"


def generate_text(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True).to(device)
    input_length = int(inputs["input_ids"].shape[1])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    response_ids = output_ids[0, input_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip(), latency_ms


def load_gsm8k(limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(ROOT / "data" / "datasets" / "gsm8k" / "grade_school_math" / "data" / "test.jsonl")
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:limit]):
        samples.append(
            {
                "dataset_key": "gsm8k",
                "sample_id": f"gsm8k/test/{index:05d}",
                "question": row["question"],
                "answer": row["answer"],
                "reference": extract_gsm8k_reference(row["answer"]),
            }
        )
    return samples


def load_cmmlu(limit: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    test_dir = ROOT / "data" / "datasets" / "cmmlu" / "data" / "test"
    for path in sorted(test_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                samples.append(
                    {
                        "dataset_key": "cmmlu",
                        "sample_id": f"cmmlu/test/{path.stem}/{row.get('', len(samples))}",
                        "subject": path.stem,
                        "question": row["Question"],
                        "choices": {key: row[key] for key in ("A", "B", "C", "D")},
                        "reference": row["Answer"].strip().upper(),
                    }
                )
                if len(samples) >= limit:
                    return samples
    return samples


def load_humaneval(limit: int) -> list[dict[str, Any]]:
    path = ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
    samples: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["dataset_key"] = "humaneval"
            row["sample_id"] = row["task_id"]
            samples.append(row)
            if len(samples) >= limit:
                break
    return samples


def build_messages(sample: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    dataset_key = sample["dataset_key"]
    if dataset_key == "gsm8k":
        prompt = (
            "Solve the math problem. Show concise reasoning and put the final answer in the form "
            "#### <number>.\n\n"
            f"Problem: {sample['question']}"
        )
    elif dataset_key == "cmmlu":
        choices = sample["choices"]
        prompt = (
            "以下是单项选择题。只输出一个大写字母 A、B、C 或 D。\n\n"
            f"题目：{sample['question']}\n"
            f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}"
        )
    elif dataset_key == "humaneval":
        prompt = (
            "Complete the Python function. Return only the code needed to complete or define the function.\n\n"
            f"{sample['prompt']}"
        )
    else:
        raise CapabilityEvalError(f"Unsupported dataset: {dataset_key}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return messages, sha256_text(SYSTEM_PROMPT + "\n" + prompt)


def score_sample(sample: dict[str, Any], response_text: str, humaneval_timeout_sec: float) -> tuple[bool, str, str]:
    dataset_key = sample["dataset_key"]
    if dataset_key == "gsm8k":
        prediction = extract_gsm8k_prediction(response_text)
        return prediction == sample["reference"], prediction, "exact_numeric_match"
    if dataset_key == "cmmlu":
        prediction = extract_choice(response_text)
        return prediction == sample["reference"], prediction, "choice_match"
    if dataset_key == "humaneval":
        passed, detail = run_humaneval_check(sample, response_text, humaneval_timeout_sec)
        return passed, "pass" if passed else "fail", detail
    raise CapabilityEvalError(f"Unsupported dataset: {dataset_key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 2 capability evaluation for GSM8K, HumanEval and CMMLU.")
    parser.add_argument("--local-model-dir", "--local_model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--adapter-path", "--adapter_path", default="")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset key filter; repeat or comma-separate.")
    parser.add_argument("--sample-limit-per-dataset", "--sample_limit_per_dataset", type=int, default=1)
    parser.add_argument("--output-trace", "--output_trace", default=str(DEFAULT_OUTPUT_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=160)
    parser.add_argument("--humaneval-timeout-sec", "--humaneval_timeout_sec", type=float, default=3.0)
    parser.add_argument("--min-accuracy", "--min_accuracy", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit_per_dataset <= 0:
        print("--sample-limit-per-dataset must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.min_accuracy <= 1:
        print("--min-accuracy must be in [0, 1]", file=sys.stderr)
        return 2

    requested = set(parse_comma_values(args.dataset)) if args.dataset else {"gsm8k", "humaneval", "cmmlu"}
    loaders = {
        "gsm8k": load_gsm8k,
        "humaneval": load_humaneval,
        "cmmlu": load_cmmlu,
    }
    unsupported = requested - set(loaders)
    if unsupported:
        print(f"Unsupported datasets: {', '.join(sorted(unsupported))}", file=sys.stderr)
        return 2

    model_dir = resolve_path(args.local_model_dir)
    adapter_path = resolve_path(args.adapter_path) if args.adapter_path else None
    output_path = resolve_path(args.output_trace)
    audit_path = resolve_path(args.audit)
    created_ts = datetime.now(timezone.utc).isoformat()

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    tokenizer, model, adapter_config = load_local_student(
        model_dir,
        adapter_path,
        args.device,
        args.dtype,
        quantize_adapter=False,
    )

    trace_rows: list[dict[str, Any]] = []
    for dataset_key in sorted(requested):
        samples = loaders[dataset_key](args.sample_limit_per_dataset)
        for index, sample in enumerate(samples, start=1):
            messages, prompt_hash = build_messages(sample)
            response_text, latency_ms = generate_text(tokenizer, model, messages, args.device, args.max_new_tokens)
            correct, prediction, score_detail = score_sample(sample, response_text, args.humaneval_timeout_sec)
            row = {
                "capability_eval_version": "1.0",
                "created_by": "scripts/evaluate_chapter2_capability.py",
                "created_ts": created_ts,
                "dataset_key": dataset_key,
                "sample_id": sample["sample_id"],
                "prompt_hash": prompt_hash,
                "reference": sample.get("reference", ""),
                "prediction": prediction,
                "correct": correct,
                "score_detail": score_detail,
                "latency_ms": latency_ms,
                "response_text": response_text,
            }
            row["capability_eval_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
            trace_rows.append(row)
            print(
                f"[{dataset_key}] {index}/{len(samples)} {sample['sample_id']} "
                f"correct={correct} latency_ms={latency_ms:.1f}",
                flush=True,
            )

    write_jsonl(output_path, trace_rows)
    counts = Counter(row["dataset_key"] for row in trace_rows)
    correct_counts = Counter(row["dataset_key"] for row in trace_rows if row["correct"])
    accuracy_by_dataset = {
        key: correct_counts[key] / counts[key] if counts[key] else 0.0
        for key in sorted(counts)
    }
    overall_accuracy = sum(1 for row in trace_rows if row["correct"]) / len(trace_rows) if trace_rows else 0.0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    status = "passed" if trace_rows and overall_accuracy >= args.min_accuracy else "failed"
    audit = {
        "gate": "CH2-CAPABILITY-EVAL-smoke",
        "check_version": "1.0",
        "created_by": "scripts/evaluate_chapter2_capability.py",
        "created_ts": created_ts,
        "status": status,
        "local_model_dir": display_path(model_dir),
        "adapter_path": display_path(adapter_path) if adapter_path else "",
        "adapter_config": adapter_config,
        "dtype": args.dtype,
        "device": args.device,
        "output_trace_path": display_path(output_path),
        "capability_eval_trace_hash": sha256_file(output_path),
        "selected_dataset_keys": sorted(requested),
        "sample_limit_per_dataset": args.sample_limit_per_dataset,
        "sample_count": len(trace_rows),
        "dataset_counts": dict(sorted(counts.items())),
        "correct_counts": dict(sorted(correct_counts.items())),
        "accuracy_by_dataset": accuracy_by_dataset,
        "overall_accuracy": overall_accuracy,
        "min_accuracy": args.min_accuracy,
        "peak_memory_mb": peak_memory_mb,
        "prompt_template_hash": sha256_text(SYSTEM_PROMPT),
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"capability_eval_trace_hash={audit['capability_eval_trace_hash']}")
    if status != "passed":
        print("Chapter 2 capability eval smoke failed.", file=sys.stderr)
        return 1
    print("Chapter 2 capability eval smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
