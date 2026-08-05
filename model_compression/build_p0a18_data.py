#!/usr/bin/env python3
"""Freeze an independent MBPP validation set for the P0-A18 Code audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a18_code_transfer.json"
OUTPUT_DIR = ROOT / "data/p0a18"
VALIDATION = OUTPUT_DIR / "code_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a18_data.json"
P0A11_TRAIN = ROOT / "data/p0a11/code_train.jsonl"
P0A11_VALIDATION = ROOT / "data/p0a11/code_validation.jsonl"
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
SOURCE_FILE = "full/validation-00000-of-00001.parquet"
SOURCE_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{REVISION}/{SOURCE_FILE}"
)
README_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{REVISION}/README.md"
)
SOURCE_SHA256 = "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80"
README_SHA256 = "6377d5c76ba46b9e650daa6d5eb592e671c9b15586e39f23f50ed9bf2ac54cf6"
EXPECTED_ROWS = 90
EXPECTED_TESTS = 3


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


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a18-data-builder/1.0"})
    with urlopen(request, timeout=120) as response:
        value = response.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise BuildError(f"Download exceeds safety cap: {url}")
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


def training_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    users = [
        str(item.get("content", ""))
        for item in messages
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(users) != 1 or not users[0].strip():
        raise BuildError(f"Invalid training prompt: {row.get('sample_id')}")
    return users[0]


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BuildError("pyarrow is required to read the frozen MBPP source") from exc

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A18-CODE-TRANSFER-AUDIT":
        raise BuildError("P0-A18 config identity mismatch")
    validation_cfg = config.get("validation") or {}
    if validation_cfg.get("revision") != REVISION or validation_cfg.get("rows") != EXPECTED_ROWS:
        raise BuildError("P0-A18 MBPP protocol changed")

    raw = download(SOURCE_URL, 256 * 1024)
    readme = download(README_URL, 1024 * 1024)
    if sha256_bytes(raw) != SOURCE_SHA256:
        raise BuildError("MBPP parquet hash mismatch")
    if sha256_bytes(readme) != README_SHA256:
        raise BuildError("MBPP README hash mismatch")

    table = parquet.read_table(pa.BufferReader(raw))
    if table.num_rows != EXPECTED_ROWS:
        raise BuildError(f"Unexpected MBPP validation rows: {table.num_rows}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    descriptions: set[str] = set()
    prompts: set[str] = set()
    for index, item in enumerate(table.to_pylist()):
        task_id = int(item["task_id"])
        description = str(item.get("text", "")).strip()
        setup = str(item.get("test_setup_code", "")).strip()
        tests = [
            str(value).strip()
            for value in [*(item.get("test_list") or []), *(item.get("challenge_test_list") or [])]
            if str(value).strip()
        ]
        if task_id in seen_ids or not description or setup:
            raise BuildError(f"Invalid/duplicate MBPP row: index={index} task_id={task_id}")
        if len(tests) != EXPECTED_TESTS:
            raise BuildError(f"MBPP task {task_id} does not have exactly three tests")
        if len(set(tests)) != EXPECTED_TESTS or not all(test.startswith("assert ") for test in tests):
            raise BuildError(f"Invalid MBPP tests for task {task_id}")
        # MBPP descriptions do not consistently expose the required function
        # name.  Its three public asserts are therefore part of the problem
        # specification, while the evaluator still executes them independently.
        prompt = description + "\n\nYour function must satisfy these examples:\n" + "\n".join(tests)
        description_norm = normalize_text(description)
        prompt_norm = normalize_text(prompt)
        if description_norm in descriptions or prompt_norm in prompts:
            raise BuildError(f"Duplicate MBPP prompt: task_id={task_id}")
        descriptions.add(description_norm)
        prompts.add(prompt_norm)
        seen_ids.add(task_id)
        rows.append(
            {
                "sample_id": f"mbpp/validation/{task_id}",
                "dataset_key": "mbpp",
                "domain": "code",
                "source": f"google-research-datasets/mbpp@{REVISION}",
                "source_row": index,
                "split_role": "p0a18_external_validation",
                "prompt": prompt,
                "reference": "unit_tests",
                "validator": "python_unit_tests",
                "unit_tests": tests,
                "metadata": {
                    "task_id": task_id,
                    "public_tests_in_prompt": EXPECTED_TESTS,
                    "reference_code_written_to_manifest": False,
                },
            }
        )

    train_prompt_set = {
        normalize_text(training_prompt(row)) for row in read_jsonl(P0A11_TRAIN)
    }
    prior_validation_set = {
        normalize_text(str(row.get("prompt", "")))
        for row in read_jsonl(P0A11_VALIDATION)
    }
    train_description_overlap = descriptions.intersection(train_prompt_set)
    train_full_prompt_overlap = prompts.intersection(train_prompt_set)
    prior_description_overlap = descriptions.intersection(prior_validation_set)
    prior_full_prompt_overlap = prompts.intersection(prior_validation_set)
    if any(
        (
            train_description_overlap,
            train_full_prompt_overlap,
            prior_description_overlap,
            prior_full_prompt_overlap,
        )
    ):
        raise BuildError("MBPP validation has exact prompt overlap with P0-A11 data")

    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "mbpp_validation.parquet", raw)
    write_bytes(source_dir / "MBPP_README.md", readme)
    write_jsonl(VALIDATION, rows)
    audit = {
        "gate": "P0-A18-MBPP-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a18_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "google-research-datasets/mbpp",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "source_rows": EXPECTED_ROWS,
            "source_sha256": SOURCE_SHA256,
            "readme_sha256": README_SHA256,
            "declared_license": "cc-by-4.0",
        },
        "protocol": {
            "public_tests_in_prompt": EXPECTED_TESTS,
            "tests_executed_per_problem": EXPECTED_TESTS,
            "reference_code_written_to_manifest": False,
            "mbpp_validation_used_for_training": False,
            "maximum_candidate_evaluations": 2,
        },
        "separation": {
            "p0a11_train_description_overlap": len(train_description_overlap),
            "p0a11_train_full_prompt_overlap": len(train_full_prompt_overlap),
            "p0a11_validation_description_overlap": len(prior_description_overlap),
            "p0a11_validation_full_prompt_overlap": len(prior_full_prompt_overlap),
            "gate300_loaded": False,
            "formal_full_loaded": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A11_TRAIN.relative_to(ROOT).as_posix(): sha256_file(P0A11_TRAIN),
            P0A11_VALIDATION.relative_to(ROOT).as_posix(): sha256_file(P0A11_VALIDATION),
        },
        "output": {
            "path": VALIDATION.relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(VALIDATION),
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    write_json(AUDIT, audit)
    print(f"P0-A18 MBPP data passed rows={len(rows)} tests_per_row={EXPECTED_TESTS}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A18 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "output": value.get("output")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A18 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
