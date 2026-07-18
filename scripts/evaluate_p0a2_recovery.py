#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from inference_utils import load_local_student  # noqa: E402
from train_cedd_repair import (  # noqa: E402
    load_generation_validation_jsonl,
    render_generation_prompt,
    score_generation_validation,
)


DEFAULT_MODEL = ROOT / "models" / "pretrained" / "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATA = ROOT / "data" / "distill" / "p0a2_recovery_validation.jsonl"
DEFAULT_TRACE = ROOT / "data" / "eval" / "p0a2_deepseek_upper_bound.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a2_deepseek_upper_bound.json"
DEFAULT_TOKEN_LIMITS = {"gsm8k": 256, "humaneval": 256, "cmmlu": 32}


class RecoveryEvalError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": display_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not path.is_dir():
        return {"path": display_path(path), "missing": True}
    key_names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "adapter_config.json",
        "adapter_model.pt",
        "model.safetensors.index.json",
        "model.safetensors",
    }
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    key_files = [item for item in files if item.name in key_names]
    digest = hashlib.sha256()
    for item in key_files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        if item.stat().st_size <= 16 * 1024 * 1024:
            digest.update(bytes.fromhex(sha256_file(item)))
    return {
        "path": display_path(path),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "key_metadata_hash": digest.hexdigest(),
    }


def parse_map(values: list[str], value_type: type[int] | type[float], name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise RecoveryEvalError(f"Invalid {name} entry: {item}")
            dataset, value = item.split("=", 1)
            dataset = dataset.strip()
            if dataset not in DEFAULT_TOKEN_LIMITS:
                raise RecoveryEvalError(f"Unsupported dataset in {name}: {dataset}")
            try:
                parsed[dataset] = value_type(value.strip())
            except ValueError as exc:
                raise RecoveryEvalError(f"Invalid {name} value: {item}") from exc
    return parsed


def select_balanced(rows: list[dict[str, Any]], limit_per_dataset: int) -> list[dict[str, Any]]:
    if limit_per_dataset <= 0:
        return rows
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row.get("dataset_key", ""))
        if counts[dataset] >= limit_per_dataset:
            continue
        selected.append(row)
        counts[dataset] += 1
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the unquantized DeepSeek base or P0-A2 adapter on selection-only recovery data."
    )
    parser.add_argument("--local-model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--validation-data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens-map", action="append", default=[])
    parser.add_argument("--min-accuracy-map", action="append", default=[])
    parser.add_argument("--min-macro-accuracy", type=float, default=0.0)
    parser.add_argument("--sample-limit-per-dataset", type=int, default=0)
    parser.add_argument("--code-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--close-reasoning-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Close a DeepSeek-style trailing <think> prefix to evaluate the non-thinking fast path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_input_length <= 0 or args.code_timeout_sec <= 0 or args.sample_limit_per_dataset < 0:
        print("Length, timeout and sample limits are invalid.", file=sys.stderr)
        return 2
    if not 0 <= args.min_macro_accuracy <= 1:
        print("--min-macro-accuracy must be in [0, 1].", file=sys.stderr)
        return 2
    try:
        token_limits = dict(DEFAULT_TOKEN_LIMITS)
        token_limits.update(parse_map(args.max_new_tokens_map, int, "--max-new-tokens-map"))
        min_accuracy = parse_map(args.min_accuracy_map, float, "--min-accuracy-map")
        if any(value <= 0 for value in token_limits.values()):
            raise RecoveryEvalError("All token limits must be positive")
        if any(not 0 <= value <= 1 for value in min_accuracy.values()):
            raise RecoveryEvalError("All minimum accuracies must be in [0, 1]")
        model_dir = resolve_path(args.local_model_dir)
        adapter_path = resolve_path(args.adapter_path) if args.adapter_path else None
        validation_path = resolve_path(args.validation_data)
        output_path = resolve_path(args.output_trace)
        audit_path = resolve_path(args.audit)
        if not model_dir.is_dir():
            raise RecoveryEvalError(f"Missing model: {display_path(model_dir)}")
        if adapter_path is not None and not adapter_path.is_dir():
            raise RecoveryEvalError(f"Missing adapter: {display_path(adapter_path)}")
        examples = select_balanced(
            load_generation_validation_jsonl(validation_path), args.sample_limit_per_dataset
        )
        if not examples:
            raise RecoveryEvalError("No validation rows selected")

        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tokenizer, model, adapter_config = load_local_student(
            model_dir, adapter_path, args.device, args.dtype
        )
        trace_rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples, start=1):
            dataset = str(example.get("dataset_key", ""))
            max_new_tokens = int(token_limits[dataset])
            prompt = render_generation_prompt(
                tokenizer, example["messages"], args.close_reasoning_prefix
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(args.device)
            input_length = int(inputs["input_ids"].shape[1])
            started = time.perf_counter()
            error = ""
            response = ""
            score = 0.0
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    generated[0, input_length:], skip_special_tokens=True
                ).strip()
                score = score_generation_validation(example, response, args.code_timeout_sec)
            except Exception as exc:  # per-sample failures are evidence, not a lost run
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000
            trace_rows.append(
                {
                    "index": index,
                    "dataset_key": dataset,
                    "sample_id": str(example.get("sample_id", "")),
                    "validation_group_id": str(example.get("validation_group_id", "")),
                    "correct": score >= 1.0,
                    "score": score,
                    "latency_ms": latency_ms,
                    "max_new_tokens": max_new_tokens,
                    "response": response,
                    "generation_error": error,
                    "used_for_training": False,
                }
            )
            print(
                f"[{index}/{len(examples)}] {dataset} correct={score >= 1.0} "
                f"latency_ms={latency_ms:.1f}",
                flush=True,
            )

        dataset_counts = Counter(str(row["dataset_key"]) for row in trace_rows)
        dataset_correct = Counter(
            str(row["dataset_key"]) for row in trace_rows if row["correct"] is True
        )
        accuracy = {
            dataset: dataset_correct[dataset] / count
            for dataset, count in sorted(dataset_counts.items())
        }
        macro_accuracy = sum(accuracy.values()) / len(accuracy) if accuracy else 0.0
        threshold_failures = {
            dataset: {"actual": accuracy.get(dataset, 0.0), "required": required}
            for dataset, required in sorted(min_accuracy.items())
            if accuracy.get(dataset, 0.0) < required
        }
        if macro_accuracy < args.min_macro_accuracy:
            threshold_failures["macro_accuracy"] = {
                "actual": macro_accuracy,
                "required": args.min_macro_accuracy,
            }
        status = "passed" if not threshold_failures else "failed"
        write_jsonl(output_path, trace_rows)
        report: dict[str, Any] = {
            "gate": "P0-A2-recovery-dev",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a2_recovery.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "backend": "local_transformers_adapter" if adapter_path else "local_transformers_base",
            "model": artifact_fingerprint(model_dir),
            "adapter": artifact_fingerprint(adapter_path) if adapter_path else None,
            "adapter_config": adapter_config,
            "validation_data": display_path(validation_path),
            "validation_data_sha256": sha256_file(validation_path),
            "output_trace": display_path(output_path),
            "output_trace_sha256": sha256_file(output_path),
            "sample_count": len(trace_rows),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "correct_counts": dict(sorted(dataset_correct.items())),
            "accuracy_by_dataset": accuracy,
            "macro_accuracy": macro_accuracy,
            "min_accuracy_by_dataset": min_accuracy,
            "min_macro_accuracy": args.min_macro_accuracy,
            "threshold_gate_enabled": bool(min_accuracy or args.min_macro_accuracy > 0),
            "threshold_failures": threshold_failures,
            "generation_error_count": sum(bool(row["generation_error"]) for row in trace_rows),
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace_rows) / len(trace_rows),
            "peak_gpu_memory_mb_decimal": (
                torch.cuda.max_memory_allocated() / 1_000_000
                if args.device.startswith("cuda")
                else 0.0
            ),
            "dtype": args.dtype,
            "max_input_length": args.max_input_length,
            "max_new_tokens_by_dataset": token_limits,
            "close_reasoning_prefix": bool(args.close_reasoning_prefix),
            "formal_test_labels_used": False,
            "errors": [],
        }
        report["report_hash"] = sha256_text(
            json.dumps(
                {key: value for key, value in report.items() if key != "report_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        write_json(audit_path, report)
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(f"status={status} macro_accuracy={macro_accuracy:.6f}")
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return 0 if status == "passed" else 1
    except (KeyError, RecoveryEvalError) as exc:
        print(f"P0-A2 recovery evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
