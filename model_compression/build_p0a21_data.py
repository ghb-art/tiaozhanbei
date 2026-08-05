#!/usr/bin/env python3
"""Build fresh execution-verified OpenCodeInstruct train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The validator launches isolated Python subprocesses from worker threads.
# Disable tokenizer worker parallelism before transformers/tokenizers loads so
# those safe forks do not emit one warning per executed sample.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from build_p0a5_data import (
    CODE_VALIDATOR_VERSION,
    code_fingerprint,
    code_prefilter,
    execute_code_case,
    load_code_execution_cache,
    load_humaneval_prompts,
    parse_tests,
    word_ngrams,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a21_code_execution_distill.json"
P0A5_CONFIG = ROOT / "configs/p0a5_capability.json"
BASE_MODEL = ROOT / "models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR = ROOT / "data/p0a21"
TRAIN = OUTPUT_DIR / "code_train.jsonl"
VALIDATION = OUTPUT_DIR / "code_validation.jsonl"
EXECUTION_CACHE = OUTPUT_DIR / "code_execution_cache.jsonl"
BASE_EXECUTION_CACHE = ROOT / "data/capability_v2/code_execution_cache.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a21_data.json"
USED_MANIFESTS = (
    ROOT / "data/capability_v2/distill_train.jsonl",
    ROOT / "data/capability_v2/internal_validation.jsonl",
    ROOT / "data/capability_v2/gate300.jsonl",
)
SHARDS = {
    0: "342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891",
    17: "7bf03f8606c18a93ca5e07b6af252e30de46b5609b2aa60133d8fe78c4c30154",
    33: "98a8b473a6dc7db7cb9c4d741c1789504b9935885876f30ada09729d562b6409",
}
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
TRAIN_ROWS = 6000
VALIDATION_ROWS = 1000
REQUIRED_ROWS = TRAIN_ROWS + VALIDATION_ROWS
MAX_SEQUENCE_LENGTH = 1536
MAX_PROMPT_WITH_GENERATION = 2048
MAX_GENERATION = 768
SYSTEM_PROMPT = (
    "Return only a complete Python function implementation in one python code block. "
    "Do not use files, network access, third-party packages, or explanatory prose."
)


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def used_ids() -> tuple[set[str], dict[str, str]]:
    result: set[str] = set()
    hashes: dict[str, str] = {}
    for path in USED_MANIFESTS:
        if not path.is_file():
            raise BuildError(f"Missing history manifest: {path.relative_to(ROOT)}")
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset_key") == "opencodeinstruct":
                result.add(str(row["sample_id"]).rsplit("/", 1)[-1])
    return result, hashes


def visible_prompt(raw_prompt: str, tests: list[str]) -> str:
    return (
        raw_prompt.strip()
        + "\n\nYour implementation must satisfy these public examples:\n"
        + "\n".join(test.strip() for test in tests[:3])
    )


def training_row(raw: dict[str, Any]) -> dict[str, Any]:
    tests = parse_tests(str(raw["unit_tests"]))
    return {
        "sample_id": f"opencodeinstruct/train/{raw['id']}",
        "dataset_key": "opencodeinstruct",
        "domain": "code",
        "source": f"nvidia/OpenCodeInstruct@{REVISION}",
        "split_role": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": visible_prompt(str(raw["input"]), tests)},
        ],
        "answer": str(raw["output"]).strip(),
        "answer_token_weight": 1.0,
        "training_weight": 1.0,
        "quality_weight": 1.0,
        "kl_weight": 0.08,
        "distill_validation": "source_tests_plus_independent_execution",
        "p0a21_role": "fresh_execution_verified_function",
        "metadata": {
            "unit_tests": tests,
            "public_test_count": 3,
            "hidden_test_count": len(tests) - 3,
            "average_test_score": float(raw["average_test_score"]),
            "independent_execution": "passed",
        },
    }


def validation_row(raw: dict[str, Any]) -> dict[str, Any]:
    tests = parse_tests(str(raw["unit_tests"]))
    return {
        "sample_id": f"opencodeinstruct/fresh-validation/{raw['id']}",
        "dataset_key": "opencodeinstruct",
        "domain": "code",
        "source": f"nvidia/OpenCodeInstruct@{REVISION}",
        "split_role": "p0a21_external_validation",
        "prompt": visible_prompt(str(raw["input"]), tests),
        "reference": "unit_tests",
        "validator": "python_unit_tests",
        "unit_tests": tests,
        "metadata": {
            "public_test_count": 3,
            "hidden_test_count": len(tests) - 3,
            "reference_code_written_to_manifest": False,
            "independent_source_execution": "passed",
        },
    }


def token_lengths(tokenizer: Any, row: dict[str, Any]) -> tuple[int, int]:
    messages = row["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": str(row["answer"])}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    return len(prompt_ids), len(full_ids)


def build(workers: int) -> int:
    try:
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("pyarrow and transformers are required") from exc
    if not 1 <= workers <= 32:
        raise BuildError("--workers must be in [1, 32]")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A21-EXECUTION-VERIFIED-CODE-DISTILL":
        raise BuildError("P0-A21 config identity mismatch")
    if config["source"]["revision"] != REVISION:
        raise BuildError("P0-A21 source revision mismatch")
    source_config = json.loads(P0A5_CONFIG.read_text(encoding="utf-8"))
    old_ids, history_hashes = used_ids()

    # Formal prompts are used only by an automatic n-gram contamination
    # exclusion.  No HumanEval answer, test, result, or model output is loaded.
    humaneval_grams = [word_ngrams(value) for value in load_humaneval_prompts()]
    raw_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for shard, expected_hash in SHARDS.items():
        path = ROOT / f"data/datasets/opencodeinstruct/data/train-{shard:05d}-of-00050.parquet"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise BuildError(f"OpenCodeInstruct shard hash mismatch: {path.relative_to(ROOT)}")
        source_hashes[path.relative_to(ROOT).as_posix()] = expected_hash
        raw_rows.extend(parquet.read_table(path).to_pylist())

    rejected: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        row_id = str(raw.get("id", ""))
        if not row_id or row_id in old_ids:
            rejected["previously_used_id"] += 1
            continue
        reason = code_prefilter(raw, source_config, humaneval_grams)
        if reason:
            rejected[reason] += 1
            continue
        try:
            tests = parse_tests(str(raw["unit_tests"]))
        except (ValueError, json.JSONDecodeError):
            rejected["invalid_tests"] += 1
            continue
        if len(tests) != 10:
            rejected["not_ten_tests"] += 1
            continue
        if len(str(raw["input"])) > 3000 or len(str(raw["output"])) > 2400:
            rejected["short_function_budget"] += 1
            continue
        unique.setdefault(normalize(str(raw["input"])), raw)
    candidates = sorted(
        unique.values(),
        key=lambda row: sha256_text(f"20260801:p0a21:{row['id']}"),
    )
    if len(candidates) < REQUIRED_ROWS + 1000:
        raise BuildError(f"Insufficient fresh prefiltered candidates: {len(candidates)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
    base_cache = load_code_execution_cache(BASE_EXECUTION_CACHE)
    phase_cache = load_code_execution_cache(EXECUTION_CACHE)
    cache = {**base_cache, **phase_cache}
    accepted: list[dict[str, Any]] = []
    cache_hits = 0
    executed = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with EXECUTION_CACHE.open("a", encoding="utf-8") as cache_handle, ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), 512):
            batch = candidates[start : start + 512]
            pending = [row for row in batch if code_fingerprint(row) not in cache]
            pending_fingerprints = {code_fingerprint(row) for row in pending}
            results = list(executor.map(execute_code_case, ((row, 5.0) for row in pending)))
            executed += len(results)
            for raw, status in results:
                fingerprint = code_fingerprint(raw)
                cache[fingerprint] = status
                cache_handle.write(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "status": status,
                            "validator_version": CODE_VALIDATOR_VERSION,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            cache_handle.flush()
            for raw in batch:
                fingerprint = code_fingerprint(raw)
                if fingerprint not in pending_fingerprints:
                    cache_hits += 1
                status = cache[fingerprint]
                if status != "passed":
                    rejected[f"execution_{status}"] += 1
                    continue
                row = training_row(raw)
                prompt_length, full_length = token_lengths(tokenizer, row)
                if full_length > MAX_SEQUENCE_LENGTH:
                    rejected["training_token_budget"] += 1
                    continue
                if prompt_length + MAX_GENERATION > MAX_PROMPT_WITH_GENERATION:
                    rejected["validation_context_budget"] += 1
                    continue
                accepted.append(raw)
                if len(accepted) >= REQUIRED_ROWS:
                    break
            print(
                f"P0-A21 execution accepted={len(accepted)}/{REQUIRED_ROWS} "
                f"checked={min(start+len(batch),len(candidates))} executed={executed}",
                flush=True,
            )
            if len(accepted) >= REQUIRED_ROWS:
                break
    if len(accepted) != REQUIRED_ROWS:
        raise BuildError(f"Only {len(accepted)} fresh executable rows; need {REQUIRED_ROWS}")

    validation_source = accepted[:VALIDATION_ROWS]
    train_source = accepted[VALIDATION_ROWS:]
    train = [training_row(row) for row in train_source]
    validation = [validation_row(row) for row in validation_source]
    train_ids = {str(row["sample_id"]) for row in train}
    validation_ids = {str(row["sample_id"]).rsplit("/", 1)[-1] for row in validation}
    if {value.rsplit("/", 1)[-1] for value in train_ids} & validation_ids:
        raise BuildError("P0-A21 train-validation id overlap")
    if train_ids & {f"opencodeinstruct/train/{value}" for value in old_ids}:
        raise BuildError("P0-A21 training reuses an old id")
    write_jsonl(TRAIN, train)
    write_jsonl(VALIDATION, validation)
    audit = {
        "gate": "P0-A21-FRESH-EXECUTION-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a21_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "nvidia/OpenCodeInstruct",
            "revision": REVISION,
            "shards": sorted(SHARDS),
            "shard_hashes": source_hashes,
            "declared_license": "cc-by-4.0",
        },
        "selection": {
            "raw_rows": len(raw_rows),
            "previously_used_ids": len(old_ids),
            "fresh_prefiltered_unique": len(candidates),
            "execution_cache_hits": cache_hits,
            "fresh_execution_count": executed,
            "accepted_rows": len(accepted),
            "rejections": dict(sorted(rejected.items())),
            "independent_execution_required": True,
            "tests_per_row": 10,
            "public_tests_per_prompt": 3,
        },
        "separation": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_validation_id_overlap": 0,
            "previous_id_overlap": 0,
            "p0a18_p0a20_per_item_feedback_used": False,
            "human_eval_prompts_used_only_for_contamination_exclusion": True,
            "human_eval_answers_tests_or_results_loaded": False,
            "formal_full_opened": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A5_CONFIG.relative_to(ROOT).as_posix(): sha256_file(P0A5_CONFIG),
            **history_hashes,
        },
        "outputs": {
            "train": {"path": TRAIN.relative_to(ROOT).as_posix(), "rows": len(train), "sha256": sha256_file(TRAIN)},
            "validation": {"path": VALIDATION.relative_to(ROOT).as_posix(), "rows": len(validation), "sha256": sha256_file(VALIDATION)},
            "execution_cache": {"path": EXECUTION_CACHE.relative_to(ROOT).as_posix(), "sha256": sha256_file(EXECUTION_CACHE)},
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(f"P0-A21 data passed train={len(train)} validation={len(validation)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "build":
        return build(args.workers)
    if not AUDIT.is_file():
        raise BuildError("P0-A21 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "selection": value.get("selection")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A21 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
