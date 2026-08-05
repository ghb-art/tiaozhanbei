#!/usr/bin/env python3
"""Build 20k HumanEval-shaped Code rows from train-only OpenCodeInstruct data."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

# Execution verification forks short-lived sandbox workers.  Disable Rust
# tokenizer worker threads before importing Transformers so those forks stay
# quiet and deterministic.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

import build_p0a44_aligned_data as aligned
import build_p0a5_data as source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0b1_converged_shared.json"
SOURCE_CONFIG = ROOT / "configs/p0a5_capability.json"
DOWNLOAD_AUDIT = ROOT / "reports/audit/gate_p0b1_code_download.json"
PRIOR_AUDIT = ROOT / "reports/audit/gate_p0a44_data.json"
PRIOR_TRAIN = ROOT / "data/p0a44/code_train.jsonl"
PRIOR_VALIDATION = ROOT / "data/p0a44/code_validation.jsonl"
SHARDS = (8, 25, 42)
TRACE = ROOT / "data/p0b1/code_build_trace.jsonl"
TRAIN = ROOT / "data/p0b1/code_train.jsonl"
VALIDATION = ROOT / "data/p0b1/code_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0b1_code_data.json"
BASE = ROOT / "models/checkpoints/p0a4/student-shared-merged"
TOTAL_TRAIN = 20000
FRESH_VALIDATION = 1500
# Keep a small verified reserve so a semantic duplicate discovered only after
# HumanEval-style conversion can be removed without shrinking either split.
VERIFIED_RESERVE = 64
SEED = 20260804


class BuildError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prompt_of(row: dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    users = [
        str(item.get("content", ""))
        for item in row.get("messages", [])
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    return users[-1] if users else ""


def token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    messages = [
        {"role": str(item["role"]), "content": str(item.get("content", ""))}
        for item in row["messages"]
        if item.get("role") in {"system", "user"}
    ]
    ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": str(row["answer"])}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(ids)


def source_to_aligned(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    _, status = source.execute_code_case((raw, 5.0))
    if status != "passed":
        return None, f"source_{status}"
    try:
        tests = source.parse_tests(str(raw["unit_tests"]))
    except (ValueError, json.JSONDecodeError):
        return None, "invalid_tests"
    source_row = {
        "sample_id": f"opencodeinstruct/train/{raw['id']}",
        "messages": [
            {"role": "system", "content": aligned.SYSTEM},
            {"role": "user", "content": str(raw["input"])},
        ],
        "answer": str(raw["output"]),
        "metadata": {
            "unit_tests": tests,
            "independent_execution": "passed",
        },
    }
    try:
        converted = aligned.convert_code_row(source_row)
    except (SyntaxError, TypeError, ValueError):
        # Some generated answers redefine the same function name.  The legacy
        # AST ranker cannot order tied FunctionDef objects; such ambiguous
        # samples are not suitable HumanEval-style supervision.
        return None, "untransformable"
    if converted is None:
        return None, "untransformable"
    metadata = converted["metadata"]
    passed, _ = aligned.run_assert_tests_check(
        str(metadata["prompt_source"]),
        str(metadata["entry_point"]),
        [str(value) for value in metadata["unit_tests"]],
        str(converted["answer"]),
        10,
    )
    if not passed:
        return None, "transformed_execution_failed"
    converted["metadata"]["transformed_execution_validation"] = "passed_all_10_tests"
    return converted, "passed"


def load_trace() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if TRACE.is_file():
        with TRACE.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row["source_id"])] = row
    return latest


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-B1-CONVERGED-SHARED-STUDENT":
        raise BuildError("P0-B1 config identity changed")
    download = json.loads(DOWNLOAD_AUDIT.read_text(encoding="utf-8"))
    prior_audit = json.loads(PRIOR_AUDIT.read_text(encoding="utf-8"))
    if download.get("status") != "passed" or prior_audit.get("status") != "passed":
        raise BuildError("Required Code source audit is not passed")
    prior_train = read_jsonl(PRIOR_TRAIN)
    if len(prior_train) != 11351:
        raise BuildError(f"Frozen prior Code count changed: {len(prior_train)}")
    prior_validation = read_jsonl(PRIOR_VALIDATION)
    used_prompts = {
        sha256_text(normalized(prompt_of(row)))
        for row in prior_train + prior_validation
        if prompt_of(row)
    }
    required_fresh_train = TOTAL_TRAIN - len(prior_train)
    required_fresh = required_fresh_train + FRESH_VALIDATION
    required_verified = required_fresh + VERIFIED_RESERVE
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    humaneval_grams = [source.word_ngrams(value) for value in source.load_humaneval_prompts()]
    candidates: dict[str, dict[str, Any]] = {}
    prefilter_rejections: Counter[str] = Counter()
    shard_hashes: dict[str, str] = {}
    for shard in SHARDS:
        path = ROOT / f"data/datasets/opencodeinstruct/data/train-{shard:05d}-of-00050.parquet"
        if not path.is_file():
            raise BuildError(f"Missing downloaded shard: {path.relative_to(ROOT)}")
        shard_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        for raw in parquet.read_table(path).to_pylist():
            reason = source.code_prefilter(raw, source_config, humaneval_grams)
            if reason:
                prefilter_rejections[reason] += 1
                continue
            identity = sha256_text(normalized(str(raw.get("input", ""))))
            if identity in used_prompts or identity in candidates:
                prefilter_rejections["historical_or_new_duplicate"] += 1
                continue
            candidates[identity] = raw
    ordered = sorted(
        candidates.values(),
        key=lambda row: sha256_text(f"{SEED}:{row.get('id', '')}"),
    )
    trace = load_trace()
    accepted: dict[str, dict[str, Any]] = {
        source_id: value["verified_row"]
        for source_id, value in trace.items()
        if isinstance(value.get("verified_row"), dict)
    }
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    rejection_counts: Counter[str] = Counter(
        str(value.get("reason", ""))
        for value in trace.values()
        if not isinstance(value.get("verified_row"), dict)
    )
    with TRACE.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=16) as executor:
        for start in range(0, len(ordered), 256):
            if len(accepted) >= required_verified:
                break
            batch = [row for row in ordered[start : start + 256] if str(row["id"]) not in trace]
            for raw, result in zip(batch, executor.map(source_to_aligned, batch)):
                converted, reason = result
                if converted is not None and token_length(tokenizer, converted) > 1536:
                    converted, reason = None, "token_budget"
                source_id = str(raw["id"])
                record = {
                    "source_id": source_id,
                    "status": "accepted" if converted is not None else "rejected",
                    "reason": reason,
                    "verified_row": converted,
                }
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                trace[source_id] = record
                processed += 1
                if converted is not None:
                    accepted[source_id] = converted
                else:
                    rejection_counts[reason] += 1
            print(
                f"P0-B1 Code accepted={len(accepted)}/{required_verified} "
                f"processed_now={processed} scanned={min(start + 256, len(ordered))}",
                flush=True,
            )
    if len(accepted) < required_verified:
        raise BuildError(f"Only {len(accepted)} fresh aligned Code rows; need {required_verified}")
    fresh_pool = sorted(
        accepted.values(),
        key=lambda row: sha256_text(f"{SEED}:split:{row['sample_id']}"),
    )
    prior_train_prompts = {
        sha256_text(normalized(prompt_of(row))) for row in prior_train if prompt_of(row)
    }
    validation_source: list[dict[str, Any]] = []
    validation_prompts: set[str] = set()
    remaining: list[dict[str, Any]] = []
    for row in fresh_pool:
        identity = sha256_text(normalized(prompt_of(row)))
        if identity in prior_train_prompts or identity in validation_prompts:
            continue
        if len(validation_source) < FRESH_VALIDATION:
            validation_source.append(row)
            validation_prompts.add(identity)
        else:
            remaining.append(row)
    fresh_train: list[dict[str, Any]] = []
    train_prompts = set(prior_train_prompts)
    for row in remaining:
        identity = sha256_text(normalized(prompt_of(row)))
        if identity in validation_prompts or identity in train_prompts:
            continue
        fresh_train.append(row)
        train_prompts.add(identity)
        if len(fresh_train) >= required_fresh_train:
            break
    if len(validation_source) != FRESH_VALIDATION or len(fresh_train) != required_fresh_train:
        raise BuildError(
            "Verified reserve exhausted while enforcing converted-prompt split isolation: "
            f"validation={len(validation_source)}/{FRESH_VALIDATION} "
            f"train={len(fresh_train)}/{required_fresh_train}"
        )
    train_rows: list[dict[str, Any]] = []
    for row in prior_train + fresh_train:
        copied = dict(row)
        copied["dataset_key"] = "opencodeinstruct"
        copied["domain"] = "code"
        copied["split_role"] = "train"
        train_rows.append(copied)
    validation_rows: list[dict[str, Any]] = []
    for row in validation_source:
        copied = dict(row)
        copied["dataset_key"] = "opencodeinstruct"
        copied["domain"] = "code"
        copied["split_role"] = "internal_validation"
        validation_rows.append(copied)
    if len(train_rows) != TOTAL_TRAIN or len(validation_rows) != FRESH_VALIDATION:
        raise BuildError("Final Code split count mismatch")
    all_ids = [str(row["sample_id"]) for row in train_rows + validation_rows]
    if len(all_ids) != len(set(all_ids)):
        raise BuildError("Final Code split has duplicate ids")
    atomic_jsonl(TRAIN, train_rows)
    atomic_jsonl(VALIDATION, validation_rows)
    report = {
        "gate": "P0-B1-HUMANEVAL-SHAPED-CODE-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0b1_code_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "train_rows": len(train_rows),
        "fresh_train_rows": len(fresh_train),
        "fresh_validation_rows": len(validation_rows),
        "contract": "humaneval_v15_body_only",
        "source_and_transformed_execution_required": True,
        "tests_per_row": 10,
        "prefilter_candidates": len(candidates),
        "prefilter_rejections": dict(sorted(prefilter_rejections.items())),
        "execution_rejections": dict(sorted(rejection_counts.items())),
        "formal_policy": {
            "humaneval_prompts_used_only_for_contamination_exclusion": True,
            "humaneval_answers_tests_or_results_loaded": False,
            "formal_item_feedback_used": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            PRIOR_TRAIN.relative_to(ROOT).as_posix(): sha256_file(PRIOR_TRAIN),
            PRIOR_VALIDATION.relative_to(ROOT).as_posix(): sha256_file(PRIOR_VALIDATION),
            **shard_hashes,
        },
        "outputs": {
            TRAIN.relative_to(ROOT).as_posix(): {"rows": len(train_rows), "sha256": sha256_file(TRAIN)},
            VALIDATION.relative_to(ROOT).as_posix(): {"rows": len(validation_rows), "sha256": sha256_file(VALIDATION)},
        },
        "errors": [],
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    atomic_json(AUDIT, report)
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed train={len(train_rows)} validation={len(validation_rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B1 Code data build failed: {exc}")
        raise SystemExit(1)
