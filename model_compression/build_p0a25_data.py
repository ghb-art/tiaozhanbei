#!/usr/bin/env python3
"""Build disjoint train-mining, validation, and small-gate Code manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import build_p0a23_data as prior


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a25_code_failure_mining.json"
P0A5_CONFIG = ROOT / "configs/p0a5_capability.json"
BASE_MODEL = ROOT / "models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR = ROOT / "data/p0a25"
TRAIN_POOL = OUTPUT_DIR / "code_train_pool.jsonl"
MINING = OUTPUT_DIR / "code_mining_manifest.jsonl"
VALIDATION = OUTPUT_DIR / "code_validation.jsonl"
GATE = OUTPUT_DIR / "code_gate100.jsonl"
CACHE = OUTPUT_DIR / "code_execution_cache.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a25_data.json"
P0A23_TRAIN = ROOT / "data/p0a23/code_train.jsonl"
P0A23_VALIDATION = ROOT / "data/p0a23/code_validation.jsonl"
P0A23_CACHE = ROOT / "data/p0a23/code_execution_cache.jsonl"
P0A23_SELECTION = ROOT / "reports/audit/p0a23/code_selection.json"
MINING_ROWS = 6000
VALIDATION_ROWS = 1000
GATE_ROWS = 100
REQUIRED_ROWS = MINING_ROWS + VALIDATION_ROWS + GATE_ROWS


class BuildError(RuntimeError):
    pass


def historical_ids() -> tuple[set[str], dict[str, str]]:
    values, hashes = prior.prior_phase_ids()
    for path in (P0A23_TRAIN, P0A23_VALIDATION):
        if not path.is_file():
            raise BuildError(f"Missing prior manifest: {path.relative_to(ROOT)}")
        hashes[path.relative_to(ROOT).as_posix()] = prior.previous.sha256_file(path)
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                values.add(str(row["sample_id"]).rsplit("/", 1)[-1])
    return values, hashes


def make_train(raw: dict[str, Any]) -> dict[str, Any]:
    row = prior.training_row(raw)
    row.pop("p0a23_role", None)
    row["kl_weight"] = 0.10
    row["p0a25_role"] = "failure_mining_pool"
    return row


def make_eval(raw: dict[str, Any], role: str, id_role: str) -> dict[str, Any]:
    tests = prior.previous.parse_tests(str(raw["unit_tests"]))
    return {
        "sample_id": f"opencodeinstruct/{id_role}/{raw['id']}",
        "dataset_key": "opencodeinstruct",
        "domain": "code",
        "source": f"nvidia/OpenCodeInstruct@{prior.previous.REVISION}",
        "split_role": role,
        "prompt": prior.visible_prompt(str(raw["input"]), tests),
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


def make_mining(train: dict[str, Any]) -> dict[str, Any]:
    tests = list(train["metadata"]["unit_tests"])
    return {
        "sample_id": train["sample_id"],
        "dataset_key": "opencodeinstruct",
        "domain": "code",
        "source": train["source"],
        "split_role": "p0a25_train_only_mining",
        "prompt": str(train["messages"][1]["content"]),
        "reference": "unit_tests",
        "validator": "python_unit_tests",
        "unit_tests": tests,
        "metadata": {
            "public_test_count": 3,
            "hidden_test_count": len(tests) - 3,
            "reference_code_written_to_manifest": False,
            "train_only_feedback": True,
        },
    }


def build(workers: int) -> int:
    try:
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("pyarrow and transformers are required") from exc
    if not 1 <= workers <= 32:
        raise BuildError("--workers must be in [1, 32]")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("protocol") != "P0-A25-CODE-FAILURE-MINING":
        raise BuildError("P0-A25 config identity mismatch")
    selection = json.loads(P0A23_SELECTION.read_text(encoding="utf-8"))
    if selection.get("status") != "passed" or selection.get("selected_step") != 96:
        raise BuildError("Frozen P0-A23 selection changed")
    source_config = json.loads(P0A5_CONFIG.read_text(encoding="utf-8"))
    old_ids, history_hashes = historical_ids()
    humaneval_grams = [
        prior.previous.word_ngrams(value)
        for value in prior.previous.load_humaneval_prompts()
    ]
    raw_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for shard, expected_hash in prior.previous.SHARDS.items():
        path = ROOT / f"data/datasets/opencodeinstruct/data/train-{shard:05d}-of-00050.parquet"
        if not path.is_file() or prior.previous.sha256_file(path) != expected_hash:
            raise BuildError(f"Source hash mismatch: {path.relative_to(ROOT)}")
        source_hashes[path.relative_to(ROOT).as_posix()] = expected_hash
        raw_rows.extend(parquet.read_table(path).to_pylist())

    rejected: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        row_id = str(raw.get("id", ""))
        if not row_id or row_id in old_ids:
            rejected["previously_used_id"] += 1
            continue
        reason = prior.previous.code_prefilter(raw, source_config, humaneval_grams)
        if reason:
            rejected[reason] += 1
            continue
        try:
            tests = prior.previous.parse_tests(str(raw["unit_tests"]))
        except (ValueError, json.JSONDecodeError):
            rejected["invalid_tests"] += 1
            continue
        if len(tests) != 10:
            rejected["not_ten_tests"] += 1
            continue
        if len(str(raw["input"])) > 3000 or len(str(raw["output"])) > 2400:
            rejected["short_function_budget"] += 1
            continue
        unique.setdefault(prior.previous.normalize(str(raw["input"])), raw)
    candidates = sorted(
        unique.values(),
        key=lambda row: prior.previous.sha256_text(f"20260802:p0a25:{row['id']}"),
    )
    if len(candidates) < REQUIRED_ROWS + 1000:
        raise BuildError(f"Insufficient fresh candidates: {len(candidates)}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, trust_remote_code=True
    )
    cache = {
        **prior.previous.load_code_execution_cache(prior.previous.BASE_EXECUTION_CACHE),
        **prior.previous.load_code_execution_cache(prior.P0A21_CACHE),
        **prior.previous.load_code_execution_cache(P0A23_CACHE),
        **prior.previous.load_code_execution_cache(CACHE),
    }
    accepted: list[dict[str, Any]] = []
    cache_hits = 0
    executed = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as cache_handle, ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        for start in range(0, len(candidates), 512):
            batch = candidates[start : start + 512]
            pending = [
                row for row in batch
                if prior.previous.code_fingerprint(row) not in cache
            ]
            pending_fingerprints = {
                prior.previous.code_fingerprint(row) for row in pending
            }
            results = list(
                executor.map(
                    prior.previous.execute_code_case,
                    ((row, 5.0) for row in pending),
                )
            )
            executed += len(results)
            for raw, status in results:
                fingerprint = prior.previous.code_fingerprint(raw)
                cache[fingerprint] = status
                cache_handle.write(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "status": status,
                            "validator_version": prior.previous.CODE_VALIDATOR_VERSION,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            cache_handle.flush()
            for raw in batch:
                fingerprint = prior.previous.code_fingerprint(raw)
                if fingerprint not in pending_fingerprints:
                    cache_hits += 1
                status = cache[fingerprint]
                if status != "passed":
                    rejected[f"execution_{status}"] += 1
                    continue
                train = make_train(raw)
                prompt_length, full_length = prior.token_lengths(tokenizer, train)
                if full_length > prior.previous.MAX_SEQUENCE_LENGTH:
                    rejected["training_token_budget"] += 1
                    continue
                if prompt_length + prior.previous.MAX_GENERATION > prior.previous.MAX_PROMPT_WITH_GENERATION:
                    rejected["validation_context_budget"] += 1
                    continue
                accepted.append(raw)
                if len(accepted) >= REQUIRED_ROWS:
                    break
            print(
                f"P0-A25 execution accepted={len(accepted)}/{REQUIRED_ROWS} "
                f"checked={min(start + len(batch), len(candidates))} executed={executed}",
                flush=True,
            )
            if len(accepted) >= REQUIRED_ROWS:
                break
    if len(accepted) != REQUIRED_ROWS:
        raise BuildError(f"Only {len(accepted)} executable rows; need {REQUIRED_ROWS}")

    gate_source = accepted[:GATE_ROWS]
    validation_source = accepted[GATE_ROWS : GATE_ROWS + VALIDATION_ROWS]
    train_source = accepted[GATE_ROWS + VALIDATION_ROWS :]
    train_pool = [make_train(row) for row in train_source]
    mining = [make_mining(row) for row in train_pool]
    validation = [
        make_eval(row, "p0a25_external_validation", "p0a25-validation")
        for row in validation_source
    ]
    gate = [
        make_eval(row, "p0a25_frozen_gate", "p0a25-gate")
        for row in gate_source
    ]
    id_groups = [
        {str(row["sample_id"]).rsplit("/", 1)[-1] for row in rows}
        for rows in (train_pool, validation, gate)
    ]
    if any(id_groups[i] & id_groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise BuildError("P0-A25 split overlap")
    if set().union(*id_groups) & old_ids:
        raise BuildError("P0-A25 historical overlap")
    for path, rows in (
        (TRAIN_POOL, train_pool),
        (MINING, mining),
        (VALIDATION, validation),
        (GATE, gate),
    ):
        prior.previous.write_jsonl(path, rows)
    audit = {
        "gate": "P0-A25-FAILURE-MINING-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a25_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "nvidia/OpenCodeInstruct",
            "revision": prior.previous.REVISION,
            "shard_hashes": source_hashes,
        },
        "selection": {
            "raw_rows": len(raw_rows),
            "previously_used_ids": len(old_ids),
            "fresh_prefiltered_unique": len(candidates),
            "execution_cache_hits": cache_hits,
            "fresh_execution_count": executed,
            "accepted_rows": len(accepted),
            "rejections": dict(sorted(rejected.items())),
            "tests_per_row": 10,
        },
        "separation": {
            "mining_pool_rows": len(train_pool),
            "validation_rows": len(validation),
            "new_gate_rows": len(gate),
            "all_pairwise_id_overlap": 0,
            "all_historical_id_overlap": 0,
            "p0a24_per_item_feedback_used": False,
            "human_eval_prompts_used_only_for_contamination_exclusion": True,
            "human_eval_answers_tests_or_results_loaded": False,
            "formal_full_opened": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): prior.previous.sha256_file(CONFIG),
            P0A23_SELECTION.relative_to(ROOT).as_posix(): prior.previous.sha256_file(P0A23_SELECTION),
            **history_hashes,
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): {
                "rows": len(rows),
                "sha256": prior.previous.sha256_file(path),
            }
            for path, rows in (
                (TRAIN_POOL, train_pool),
                (MINING, mining),
                (VALIDATION, validation),
                (GATE, gate),
                (CACHE, []),
            )
        },
        "errors": [],
    }
    audit["report_hash"] = prior.previous.sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    prior.previous.write_json(AUDIT, audit)
    print(
        f"P0-A25 data passed mining={len(train_pool)} validation={len(validation)} "
        f"gate={len(gate)}"
    )
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "build":
        return build(args.workers)
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "selection": value.get("selection")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A25 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
