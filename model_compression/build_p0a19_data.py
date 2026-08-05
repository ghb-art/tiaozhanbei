#!/usr/bin/env python3
"""Build the leak-safe P0-A19 OpenCodeInstruct + MBPP Code corpus."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a19_code_mixed_distill.json"
BASE_MODEL = ROOT / "models/checkpoints/p0a4/student-shared-merged"
OPENCODE = ROOT / "data/p0a11/code_train.jsonl"
P0A18_VALIDATION = ROOT / "data/p0a18/code_validation.jsonl"
OUTPUT_DIR = ROOT / "data/p0a19"
TRAIN = OUTPUT_DIR / "code_train.jsonl"
VALIDATION = OUTPUT_DIR / "code_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a19_data.json"
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
REPO_URL = "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve"
TRAIN_FILE = "full/train-00000-of-00001.parquet"
VALIDATION_FILE = "sanitized/test-00000-of-00001.parquet"
README_FILE = "README.md"
TRAIN_SHA256 = "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae"
VALIDATION_SHA256 = "e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850"
README_SHA256 = "6377d5c76ba46b9e650daa6d5eb592e671c9b15586e39f23f50ed9bf2ac54cf6"
OPENCODE_ROWS = 8000
MBPP_SOURCE_TRAIN_ROWS = 374
MBPP_TRAIN_ROWS = 373
MBPP_SOURCE_VALIDATION_ROWS = 257
MBPP_VALIDATION_ROWS = 255
MAX_SEQUENCE_LENGTH = 1536
KL_WEIGHT = 0.05
SYSTEM_PROMPT = (
    "Return only a complete Python function implementation in one python code block. "
    "Do not use files, network access, third-party packages, or explanatory prose."
)


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def download(path: str, maximum_bytes: int) -> bytes:
    url = f"{REPO_URL}/{REVISION}/{path}"
    request = Request(url, headers={"User-Agent": "p0a19-data-builder/1.0"})
    with urlopen(request, timeout=120) as response:
        value = response.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise BuildError(f"Download exceeds safety cap: {path}")
    return value


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def user_prompt(row: dict[str, Any]) -> str:
    users = [
        str(item.get("content", ""))
        for item in row.get("messages") or []
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(users) != 1 or not users[0].strip():
        raise BuildError(f"Invalid OpenCode prompt: {row.get('sample_id')}")
    return users[0]


def full_token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in row["messages"]
        if isinstance(item, dict) and item.get("role") in {"system", "user"}
    ]
    messages.append({"role": "assistant", "content": str(row["answer"])})
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(ids)


def mbpp_prompt(description: str, tests: list[str], setup: str = "") -> str:
    parts = [description]
    if setup:
        parts.append("Test fixtures (created after your implementation):\n" + setup)
    parts.append("Your implementation must satisfy these examples:\n" + "\n".join(tests))
    return "\n\n".join(parts)


def training_row(item: dict[str, Any], source_row: int) -> dict[str, Any]:
    task_id = int(item["task_id"])
    description = str(item.get("text", "")).strip()
    code = str(item.get("code", "")).strip()
    setup = str(item.get("test_setup_code", "")).strip()
    tests = [
        str(value).strip()
        for value in [*(item.get("test_list") or []), *(item.get("challenge_test_list") or [])]
        if str(value).strip()
    ]
    if not description or not code or len(tests) != 3:
        raise BuildError(f"Invalid MBPP train task {task_id}")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise BuildError(f"Invalid MBPP reference code for task {task_id}") from exc
    return {
        "sample_id": f"mbpp/full/train/{task_id}",
        "dataset_key": "mbpp_train",
        "domain": "code",
        "source": f"google-research-datasets/mbpp@{REVISION}",
        "source_row": source_row,
        "split_role": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mbpp_prompt(description, tests, setup)},
        ],
        "answer": f"```python\n{code}\n```",
        "answer_token_weight": 1.0,
        "training_weight": 8.0,
        "quality_weight": 1.0,
        "kl_weight": KL_WEIGHT,
        "distill_validation": "human_reference_plus_public_tests",
        "p0a19_role": "mbpp_executable_short_function",
        "metadata": {
            "task_id": task_id,
            "unit_tests": tests,
            "test_setup_code": setup,
            "reference_parse": "passed",
        },
    }


def validation_row(item: dict[str, Any], source_row: int) -> dict[str, Any]:
    task_id = int(item["task_id"])
    description = str(item.get("prompt", "")).strip()
    imports = [str(value).strip() for value in item.get("test_imports") or [] if str(value).strip()]
    tests = [str(value).strip() for value in item.get("test_list") or [] if str(value).strip()]
    if not description or not 3 <= len(tests) <= 5:
        raise BuildError(f"Invalid sanitized MBPP task {task_id}")
    if any(value != "import math" for value in imports):
        raise BuildError(f"Unapproved sanitized MBPP import for task {task_id}: {imports}")
    executable_tests = ["\n".join([*imports, test]) for test in tests]
    visible = [*imports, *tests]
    return {
        "sample_id": f"mbpp/sanitized/evaluation/{task_id}",
        "dataset_key": "mbpp_sanitized",
        "domain": "code",
        "source": f"google-research-datasets/mbpp@{REVISION}",
        "source_row": source_row,
        "split_role": "p0a19_external_validation",
        "prompt": mbpp_prompt(description, visible),
        "reference": "unit_tests",
        "validator": "python_unit_tests",
        "unit_tests": executable_tests,
        "metadata": {
            "task_id": task_id,
            "public_tests_in_prompt": len(tests),
            "test_imports": imports,
            "reference_code_written_to_manifest": False,
        },
    }


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("pyarrow and transformers are required") from exc

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A19-CODE-MIXED-DISTILL":
        raise BuildError("P0-A19 config identity mismatch")
    registered = config["training"]
    if (
        registered["opencode_short_function_rows"] != OPENCODE_ROWS
        or registered["mbpp_rows_after_separation"] != MBPP_TRAIN_ROWS
        or config["validation"]["rows_after_separation_and_dedup"] != MBPP_VALIDATION_ROWS
    ):
        raise BuildError("P0-A19 registered row counts changed")

    raw_train = download(TRAIN_FILE, 256 * 1024)
    raw_validation = download(VALIDATION_FILE, 256 * 1024)
    readme = download(README_FILE, 1024 * 1024)
    for label, value, expected in (
        ("train", raw_train, TRAIN_SHA256),
        ("validation", raw_validation, VALIDATION_SHA256),
        ("README", readme, README_SHA256),
    ):
        if sha256_bytes(value) != expected:
            raise BuildError(f"MBPP {label} hash mismatch")
    train_source = parquet.read_table(pa.BufferReader(raw_train)).to_pylist()
    validation_source = parquet.read_table(pa.BufferReader(raw_validation)).to_pylist()
    if len(train_source) != MBPP_SOURCE_TRAIN_ROWS or len(validation_source) != MBPP_SOURCE_VALIDATION_ROWS:
        raise BuildError("Unexpected MBPP source row count")

    p0a18_rows = read_jsonl(P0A18_VALIDATION)
    p0a18_descriptions = {
        normalize_text(str(row["prompt"]).split("\n\nYour function must satisfy", 1)[0])
        for row in p0a18_rows
    }
    sanitized_descriptions = {
        normalize_text(str(item.get("prompt", ""))) for item in validation_source
    }

    mbpp_train: list[dict[str, Any]] = []
    train_rejections: Counter[str] = Counter()
    for index, item in enumerate(train_source):
        description = normalize_text(str(item.get("text", "")))
        if description in sanitized_descriptions:
            train_rejections["sanitized_validation_prompt_overlap"] += 1
            continue
        if description in p0a18_descriptions:
            train_rejections["p0a18_prompt_overlap"] += 1
            continue
        mbpp_train.append(training_row(item, index))
    if len(mbpp_train) != MBPP_TRAIN_ROWS:
        raise BuildError(
            f"Expected {MBPP_TRAIN_ROWS} separated MBPP train rows, found {len(mbpp_train)}"
        )

    validation: list[dict[str, Any]] = []
    validation_rejections: Counter[str] = Counter()
    seen_validation_prompts: set[str] = set()
    for index, item in enumerate(validation_source):
        description = normalize_text(str(item.get("prompt", "")))
        if description in p0a18_descriptions:
            validation_rejections["p0a18_prompt_overlap"] += 1
            continue
        if description in seen_validation_prompts:
            validation_rejections["duplicate_prompt"] += 1
            continue
        seen_validation_prompts.add(description)
        validation.append(validation_row(item, index))
    if len(validation) != MBPP_VALIDATION_ROWS:
        raise BuildError(
            f"Expected {MBPP_VALIDATION_ROWS} MBPP validation rows, found {len(validation)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, trust_remote_code=True
    )
    forbidden_descriptions = {
        normalize_text(str(row["messages"][-1]["content"]).split("\n\nYour implementation", 1)[0])
        for row in mbpp_train
    } | seen_validation_prompts | p0a18_descriptions
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    opencode_rejections: Counter[str] = Counter()
    for row in read_jsonl(OPENCODE):
        prompt = user_prompt(row)
        if normalize_text(prompt) in forbidden_descriptions:
            opencode_rejections["mbpp_prompt_overlap"] += 1
            continue
        copied = dict(row)
        copied["training_weight"] = 1.0
        copied["kl_weight"] = KL_WEIGHT
        copied["answer_token_weight"] = 1.0
        copied["p0a19_role"] = "opencode_short_function_replay"
        length = full_token_length(tokenizer, copied)
        if length > MAX_SEQUENCE_LENGTH:
            opencode_rejections["sequence_budget"] += 1
            continue
        candidates.append((length, sha256_text(str(copied["sample_id"])), copied))
    candidates.sort(key=lambda value: (value[0], value[1]))
    if len(candidates) < OPENCODE_ROWS:
        raise BuildError(f"Insufficient OpenCode candidates: {len(candidates)}")
    opencode = [value[2] for value in candidates[:OPENCODE_ROWS]]
    selected_lengths = [value[0] for value in candidates[:OPENCODE_ROWS]]

    train = sorted(
        [*opencode, *mbpp_train],
        key=lambda row: sha256_text(f"20260801:p0a19:{row['sample_id']}"),
    )
    train_ids = [str(row["sample_id"]) for row in train]
    if len(train_ids) != len(set(train_ids)):
        raise BuildError("Duplicate P0-A19 training id")
    train_descriptions = {
        normalize_text(user_prompt(row).split("\n\nYour implementation", 1)[0])
        for row in train
    }
    if train_descriptions.intersection(seen_validation_prompts):
        raise BuildError("P0-A19 train-validation prompt overlap")

    write_jsonl(TRAIN, train)
    write_jsonl(VALIDATION, validation)
    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "mbpp_full_train.parquet", raw_train)
    write_bytes(source_dir / "mbpp_sanitized_evaluation.parquet", raw_validation)
    write_bytes(source_dir / "MBPP_README.md", readme)
    audit = {
        "gate": "P0-A19-CODE-MIXED-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a19_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "google-research-datasets/mbpp",
            "revision": REVISION,
            "declared_license": "cc-by-4.0",
            "train_file": TRAIN_FILE,
            "train_sha256": TRAIN_SHA256,
            "validation_file": VALIDATION_FILE,
            "validation_sha256": VALIDATION_SHA256,
            "readme_sha256": README_SHA256,
        },
        "training": {
            "rows": len(train),
            "dataset_counts": dict(sorted(Counter(str(row["dataset_key"]) for row in train).items())),
            "opencode_selected_rows": len(opencode),
            "opencode_candidate_rows": len(candidates),
            "opencode_rejections": dict(sorted(opencode_rejections.items())),
            "opencode_selected_token_min": min(selected_lengths),
            "opencode_selected_token_max": max(selected_lengths),
            "mbpp_source_rows": len(train_source),
            "mbpp_selected_rows": len(mbpp_train),
            "mbpp_rejections": dict(sorted(train_rejections.items())),
            "mbpp_training_weight": 8.0,
            "opencode_training_weight": 1.0,
            "kl_weight": KL_WEIGHT,
        },
        "validation": {
            "source_rows": len(validation_source),
            "selected_rows": len(validation),
            "rejections": dict(sorted(validation_rejections.items())),
            "test_count_distribution": dict(
                sorted(Counter(str(len(row["unit_tests"])) for row in validation).items())
            ),
            "reference_code_written_to_manifest": False,
            "used_for_training": False,
            "maximum_candidate_evaluations": 2,
        },
        "separation": {
            "p0a18_prompt_overlap_in_training": 0,
            "p0a18_prompt_overlap_removed_from_validation": validation_rejections.get("p0a18_prompt_overlap", 0),
            "train_validation_prompt_overlap": 0,
            "human_eval_loaded": False,
            "gate300_loaded": False,
            "formal_full_loaded": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            OPENCODE.relative_to(ROOT).as_posix(): sha256_file(OPENCODE),
            P0A18_VALIDATION.relative_to(ROOT).as_posix(): sha256_file(P0A18_VALIDATION),
        },
        "outputs": {
            "train": {"path": TRAIN.relative_to(ROOT).as_posix(), "rows": len(train), "sha256": sha256_file(TRAIN)},
            "validation": {"path": VALIDATION.relative_to(ROOT).as_posix(), "rows": len(validation), "sha256": sha256_file(VALIDATION)},
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(
        f"P0-A19 data passed train={len(train)} "
        f"(OpenCode={len(opencode)}, MBPP={len(mbpp_train)}) validation={len(validation)}"
    )
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A19 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "training": value.get("training"), "validation": value.get("validation")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A19 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
