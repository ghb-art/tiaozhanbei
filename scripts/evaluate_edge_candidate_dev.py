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
from urllib.request import Request, urlopen

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPRESSION = ROOT / "model_compression"
if str(MODEL_COMPRESSION) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPRESSION))

from inference_utils import load_local_student  # noqa: E402
from edge_candidate_eval_utils import (  # noqa: E402
    EdgeCandidateEvalError,
    load_generation_validation_jsonl,
    render_generation_prompt,
    score_generation_validation,
)


DEFAULT_MODEL = ROOT / "models" / "pretrained" / "Qwen--Qwen3-1.7B"
DEFAULT_DATA = ROOT / "data" / "distill" / "p0a2_recovery_validation.jsonl"
DEFAULT_TRACE = ROOT / "data" / "eval" / "p0a3_qwen3_1p7b_dev.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a3_qwen3_1p7b_dev.json"
DEFAULT_TOKEN_LIMITS = {"gsm8k": 512, "humaneval": 512, "cmmlu": 256}


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
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model.safetensors",
    }
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    key_files = [item for item in files if item.name in key_names]
    digest = hashlib.sha256()
    for item in key_files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        if item.name.startswith("adapter_model.") or item.stat().st_size <= 16 * 1024 * 1024:
            digest.update(bytes.fromhex(sha256_file(item)))
    return {
        "path": display_path(path),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "key_metadata_hash": digest.hexdigest(),
    }


def apply_peft_adapter_scale(model: Any, scale: float) -> int:
    """Scale every active PEFT LoRA branch without modifying checkpoint files."""
    updated = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name, value in list(scaling.items()):
            scaling[adapter_name] = value * scale
            updated += 1
    if updated == 0:
        raise RecoveryEvalError("PEFT adapter scale requested but no LoRA scaling entries were found")
    return updated


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


def parse_model_id_map(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise RecoveryEvalError(f"Invalid endpoint-model-id map entry: {item}")
            dataset, model_id = item.split("=", 1)
            dataset = dataset.strip()
            model_id = model_id.strip()
            if dataset not in DEFAULT_TOKEN_LIMITS:
                raise RecoveryEvalError(f"Unsupported endpoint-model-id dataset: {dataset}")
            if not model_id:
                raise RecoveryEvalError(f"Empty endpoint model ID for dataset: {dataset}")
            parsed[dataset] = model_id
    return parsed


def parse_request_extra_map(values: list[str]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for item in values:
        if "=" not in item:
            raise RecoveryEvalError(f"Invalid request-extra map entry: {item}")
        dataset, payload_text = item.split("=", 1)
        dataset = dataset.strip()
        if dataset not in DEFAULT_TOKEN_LIMITS:
            raise RecoveryEvalError(f"Unsupported request-extra dataset: {dataset}")
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise RecoveryEvalError("Each request-extra map value must be a JSON object")
        parsed[dataset] = payload
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


def chat_completions_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def endpoint_health_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value + "/health"


def endpoint_models_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value + "/v1/models"


def select_served_model_id(model_ids: list[str], requested: str) -> str:
    available = [value for value in model_ids if value]
    if not available:
        raise RecoveryEvalError("Endpoint returned no served model IDs")
    requested = requested.strip()
    if requested.lower() in {"", "auto"}:
        if len(available) != 1:
            raise RecoveryEvalError(
                f"Endpoint serves multiple models; set --endpoint-model-id explicitly: {available}"
            )
        return available[0]
    if requested not in available:
        raise RecoveryEvalError(
            f"Requested endpoint model ID is unavailable: {requested}; available={available}"
        )
    return requested


def list_endpoint_model_ids(endpoint: str, timeout_sec: float = 5.0) -> list[str]:
    try:
        with urlopen(Request(endpoint_models_url(endpoint), method="GET"), timeout=timeout_sec) as response:
            value = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RecoveryEvalError(f"Cannot list endpoint models: {exc}") from exc
    return [
        str(item.get("id", ""))
        for item in value.get("data", [])
        if isinstance(item, dict)
    ]


def resolve_endpoint_model_id(endpoint: str, requested: str, timeout_sec: float = 5.0) -> str:
    return select_served_model_id(list_endpoint_model_ids(endpoint, timeout_sec), requested)


def require_endpoint_health(endpoint: str, timeout_sec: float = 5.0) -> None:
    try:
        with urlopen(Request(endpoint_health_url(endpoint), method="GET"), timeout=timeout_sec) as response:
            if not 200 <= int(response.status) < 300:
                raise RecoveryEvalError(f"Endpoint health returned HTTP {response.status}")
    except RecoveryEvalError:
        raise
    except Exception as exc:
        raise RecoveryEvalError(f"Endpoint health check failed: {exc}") from exc


def request_chat_completion(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    timeout_sec: float,
    disable_thinking: bool,
    request_extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if request_extra:
        forbidden = set(request_extra) & {"model", "messages", "max_tokens", "temperature", "stream"}
        if disable_thinking and "chat_template_kwargs" in request_extra:
            forbidden.add("chat_template_kwargs")
        if forbidden:
            raise RecoveryEvalError(f"Request-extra cannot override evaluator fields: {sorted(forbidden)}")
        payload.update(request_extra)
    request = Request(
        chat_completions_url(endpoint),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        value = json.loads(response.read().decode("utf-8"))
    choices = value.get("choices", [])
    if not choices:
        raise RecoveryEvalError("Chat endpoint returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise RecoveryEvalError("Chat endpoint returned non-text content")
    return content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an HF or OpenAI-compatible edge candidate on the frozen 170-row Dev protocol."
    )
    parser.add_argument("--local-model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional local PEFT adapter loaded on top of --local-model-dir.",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Inference-only multiplier for the loaded PEFT LoRA; audited and defaulting to 1.",
    )
    parser.add_argument("--endpoint", default="", help="OpenAI-compatible base URL; overrides local HF inference.")
    parser.add_argument("--endpoint-model-id", default="auto")
    parser.add_argument(
        "--endpoint-model-id-map",
        action="append",
        default=[],
        help="Dataset-specific served model IDs as dataset=model_id; repeat or comma-separate entries.",
    )
    parser.add_argument(
        "--model-artifact",
        default="",
        help="Optional HF directory or GGUF file fingerprinted when --endpoint is used.",
    )
    parser.add_argument("--candidate-name", default="edge-candidate")
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
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--request-extra-json",
        default="{}",
        help="Extra endpoint fields, such as one explicit llama.cpp per-request Top-1 adapter vector.",
    )
    parser.add_argument(
        "--request-extra-json-map",
        action="append",
        default=[],
        help="Dataset-specific endpoint extras as dataset=JSON; repeat once per task.",
    )
    parser.add_argument(
        "--kv-cache-type",
        default="",
        help="Audit-only runtime metadata for endpoint inference (for example q8_0 or f16).",
    )
    parser.add_argument(
        "--close-reasoning-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Close a DeepSeek-style trailing <think> prefix to evaluate the non-thinking fast path.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking=false to Qwen3 chat templates/endpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.max_input_length <= 0
        or args.code_timeout_sec <= 0
        or args.request_timeout_sec <= 0
        or args.sample_limit_per_dataset < 0
        or args.adapter_scale <= 0
    ):
        print("Length, timeout and sample limits are invalid.", file=sys.stderr)
        return 2
    if not 0 <= args.min_macro_accuracy <= 1:
        print("--min-macro-accuracy must be in [0, 1].", file=sys.stderr)
        return 2
    try:
        token_limits = dict(DEFAULT_TOKEN_LIMITS)
        token_limits.update(parse_map(args.max_new_tokens_map, int, "--max-new-tokens-map"))
        min_accuracy = parse_map(args.min_accuracy_map, float, "--min-accuracy-map")
        requested_model_id_map = parse_model_id_map(args.endpoint_model_id_map)
        request_extra = json.loads(args.request_extra_json)
        if not isinstance(request_extra, dict):
            raise RecoveryEvalError("--request-extra-json must decode to an object")
        request_extra_map = parse_request_extra_map(args.request_extra_json_map)
        if any(value <= 0 for value in token_limits.values()):
            raise RecoveryEvalError("All token limits must be positive")
        if any(not 0 <= value <= 1 for value in min_accuracy.values()):
            raise RecoveryEvalError("All minimum accuracies must be in [0, 1]")
        endpoint = str(args.endpoint).strip()
        model_dir = resolve_path(args.local_model_dir)
        adapter_dir = resolve_path(args.adapter_dir) if args.adapter_dir else None
        model_artifact = (
            resolve_path(args.model_artifact)
            if args.model_artifact
            else adapter_dir
            if adapter_dir is not None
            else model_dir
        )
        validation_path = resolve_path(args.validation_data)
        output_path = resolve_path(args.output_trace)
        audit_path = resolve_path(args.audit)
        if not endpoint and not model_dir.is_dir():
            raise RecoveryEvalError(f"Missing model: {display_path(model_dir)}")
        if adapter_dir is not None and not adapter_dir.is_dir():
            raise RecoveryEvalError(f"Missing PEFT adapter: {display_path(adapter_dir)}")
        if adapter_dir is None and args.adapter_scale != 1.0:
            raise RecoveryEvalError("--adapter-scale requires --adapter-dir")
        if endpoint and adapter_dir is not None:
            raise RecoveryEvalError("--adapter-dir is only valid for local Transformers evaluation")
        if endpoint and args.model_artifact and not model_artifact.exists():
            raise RecoveryEvalError(f"Missing model artifact: {display_path(model_artifact)}")
        if endpoint:
            require_endpoint_health(endpoint)
            available_model_ids = list_endpoint_model_ids(
                endpoint, min(args.request_timeout_sec, 10.0)
            )
            endpoint_model_id = select_served_model_id(available_model_ids, args.endpoint_model_id)
            endpoint_model_id_map = {
                dataset: select_served_model_id(available_model_ids, requested)
                for dataset, requested in requested_model_id_map.items()
            }
            print(f"Endpoint model resolved: {endpoint_model_id}", flush=True)
            if endpoint_model_id_map:
                print(f"Endpoint task model map: {endpoint_model_id_map}", flush=True)
        else:
            endpoint_model_id = ""
            endpoint_model_id_map = {}
        examples = select_balanced(
            load_generation_validation_jsonl(validation_path), args.sample_limit_per_dataset
        )
        if not examples:
            raise RecoveryEvalError("No validation rows selected")

        if not endpoint and args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tokenizer: Any | None = None
        model: Any | None = None
        if not endpoint:
            tokenizer, model = load_local_student(model_dir, args.device, args.dtype)
            if adapter_dir is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
                apply_peft_adapter_scale(model, float(args.adapter_scale))
                model.eval()
        trace_rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples, start=1):
            dataset = str(example.get("dataset_key", ""))
            max_new_tokens = int(token_limits[dataset])
            started = time.perf_counter()
            error = ""
            response = ""
            score = 0.0
            sample_endpoint_model_id = endpoint_model_id_map.get(dataset, endpoint_model_id)
            try:
                if endpoint:
                    response = request_chat_completion(
                        endpoint,
                        sample_endpoint_model_id,
                        example["messages"],
                        max_new_tokens,
                        args.request_timeout_sec,
                        args.disable_thinking,
                        request_extra_map.get(dataset, request_extra),
                    )
                else:
                    assert tokenizer is not None and model is not None
                    prompt = render_generation_prompt(
                        tokenizer,
                        example["messages"],
                        args.close_reasoning_prefix,
                        args.disable_thinking,
                    )
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_input_length,
                    ).to(args.device)
                    input_length = int(inputs["input_ids"].shape[1])
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
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
                    "endpoint_model_id": sample_endpoint_model_id,
                    "prompt_hash": sha256_text(
                        json.dumps(example["messages"], ensure_ascii=False, sort_keys=True)
                    ),
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
            if error:
                print(
                    f"[FAIL FAST] generation_error on {example.get('sample_id', '')}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                break

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
        generation_error_count = sum(bool(row["generation_error"]) for row in trace_rows)
        failure_reasons: list[str] = []
        if threshold_failures:
            failure_reasons.append("accuracy_threshold_failure")
        if generation_error_count:
            failure_reasons.append("generation_errors_present")
        status = "passed" if not failure_reasons else "failed"
        write_jsonl(output_path, trace_rows)
        report: dict[str, Any] = {
            "gate": "P0-edge-candidate-dev",
            "check_version": "2.0",
            "created_by": "scripts/evaluate_edge_candidate_dev.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "candidate_name": args.candidate_name,
            "backend": (
                "openai_compatible_http"
                if endpoint
                else "local_transformers_peft"
                if adapter_dir is not None
                else "local_transformers_base"
            ),
            "endpoint": endpoint,
            "endpoint_model_id": endpoint_model_id,
            "endpoint_model_id_map": endpoint_model_id_map,
            "endpoint_model_id_map_hash": sha256_text(
                json.dumps(endpoint_model_id_map, ensure_ascii=False, sort_keys=True)
            ),
            "model": artifact_fingerprint(model_artifact),
            "base_model": artifact_fingerprint(model_dir),
            "adapter": artifact_fingerprint(adapter_dir) if adapter_dir is not None else {},
            "adapter_scale": float(args.adapter_scale) if adapter_dir is not None else 0.0,
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
            "generation_error_count": generation_error_count,
            "failure_reasons": failure_reasons,
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in trace_rows) / len(trace_rows),
            "peak_gpu_memory_mb_decimal": (
                torch.cuda.max_memory_allocated() / 1_000_000
                if not endpoint and args.device.startswith("cuda")
                else 0.0
            ),
            "dtype": args.dtype,
            "max_input_length": args.max_input_length,
            "max_new_tokens_by_dataset": token_limits,
            "close_reasoning_prefix": bool(args.close_reasoning_prefix),
            "disable_thinking": bool(args.disable_thinking),
            "kv_cache_type": str(args.kv_cache_type),
            "request_extra_hash": sha256_text(
                json.dumps(request_extra, ensure_ascii=False, sort_keys=True)
            ),
            "request_extra_map_hash": sha256_text(
                json.dumps(request_extra_map, ensure_ascii=False, sort_keys=True)
            ),
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
        if model is not None:
            del model
        gc.collect()
        if not endpoint and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return 0 if status == "passed" else 1
    except (KeyError, ValueError, json.JSONDecodeError, EdgeCandidateEvalError, RecoveryEvalError) as exc:
        print(f"Edge candidate evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
