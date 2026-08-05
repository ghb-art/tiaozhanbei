#!/usr/bin/env python3
"""Freeze the untouched MBPP remainder for P0-A20 Code runtime selection."""

from __future__ import annotations

import argparse
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
CONFIG = ROOT / "configs/p0a20_code_runtime.json"
BASE_MODEL = ROOT / "models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR = ROOT / "data/p0a20"
VALIDATION = OUTPUT_DIR / "code_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a20_data.json"
P0A19_TRAIN = ROOT / "data/p0a19/code_train.jsonl"
HISTORY_FILES = {
    "full_train": ROOT / "data/p0a19/sources/mbpp_full_train.parquet",
    "full_validation": ROOT / "data/p0a18/sources/mbpp_validation.parquet",
    "sanitized_test": ROOT / "data/p0a19/sources/mbpp_sanitized_evaluation.parquet",
}
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
SOURCE_FILE = "full/test-00000-of-00001.parquet"
SOURCE_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{REVISION}/{SOURCE_FILE}"
)
README_URL = (
    "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
    f"{REVISION}/README.md"
)
SOURCE_SHA256 = "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77"
README_SHA256 = "6377d5c76ba46b9e650daa6d5eb592e671c9b15586e39f23f50ed9bf2ac54cf6"
SOURCE_ROWS = 500
OUTPUT_ROWS_BEFORE_CONTEXT = 240
OUTPUT_ROWS = 239
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


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a20-data-builder/1.0"})
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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prompt(description: str, tests: list[str], setup: str) -> str:
    parts = [description]
    if setup:
        parts.append("Test fixtures (created after your implementation):\n" + setup)
    parts.append("Your implementation must satisfy these examples:\n" + "\n".join(tests))
    return "\n\n".join(parts)


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("pyarrow and transformers are required") from exc
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A20-CODE-THINKING-RUNTIME":
        raise BuildError("P0-A20 config identity mismatch")

    raw = download(SOURCE_URL, 256 * 1024)
    readme = download(README_URL, 1024 * 1024)
    if sha256_bytes(raw) != SOURCE_SHA256 or sha256_bytes(readme) != README_SHA256:
        raise BuildError("P0-A20 MBPP source hash mismatch")
    source = parquet.read_table(pa.BufferReader(raw)).to_pylist()
    if len(source) != SOURCE_ROWS:
        raise BuildError(f"Unexpected MBPP source rows: {len(source)}")

    history_ids: set[int] = set()
    history_prompts: set[str] = set()
    history_hashes: dict[str, str] = {}
    for label, path in HISTORY_FILES.items():
        if not path.is_file():
            raise BuildError(f"Missing history source: {path.relative_to(ROOT)}")
        table = parquet.read_table(path)
        for item in table.to_pylist():
            history_ids.add(int(item["task_id"]))
            history_prompts.add(normalize(str(item.get("text", item.get("prompt", "")))))
        history_hashes[label] = sha256_file(path)

    opencode_prompts: set[str] = set()
    for line in P0A19_TRAIN.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset_key") != "opencodeinstruct":
            continue
        users = [
            str(item.get("content", ""))
            for item in row.get("messages") or []
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        if len(users) != 1:
            raise BuildError("Invalid P0-A19 OpenCode prompt")
        opencode_prompts.add(normalize(users[0]))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    for index, item in enumerate(source):
        task_id = int(item["task_id"])
        description = str(item.get("text", "")).strip()
        description_norm = normalize(description)
        if task_id in history_ids:
            rejected["history_task_id_overlap"] += 1
            continue
        if description_norm in history_prompts:
            rejected["history_prompt_overlap"] += 1
            continue
        if description_norm in opencode_prompts:
            rejected["opencode_prompt_overlap"] += 1
            continue
        if description_norm in seen:
            rejected["duplicate_prompt"] += 1
            continue
        seen.add(description_norm)
        setup = str(item.get("test_setup_code", "")).strip()
        tests = [
            str(value).strip()
            for value in [*(item.get("test_list") or []), *(item.get("challenge_test_list") or [])]
            if str(value).strip()
        ]
        if not 3 <= len(tests) <= 6:
            raise BuildError(f"Unexpected test count for MBPP task {task_id}")
        executable = ["\n".join([setup, test]).strip() for test in tests]
        rows.append(
            {
                "sample_id": f"mbpp/full/untouched/{task_id}",
                "dataset_key": "mbpp_remainder",
                "domain": "code",
                "source": f"google-research-datasets/mbpp@{REVISION}",
                "source_row": index,
                "split_role": "p0a20_external_validation",
                "prompt": prompt(description, tests, setup),
                "reference": "unit_tests",
                "validator": "python_unit_tests",
                "unit_tests": executable,
                "metadata": {
                    "task_id": task_id,
                    "public_tests_in_prompt": len(tests),
                    "reference_code_written_to_manifest": False,
                },
            }
        )
    if len(rows) != OUTPUT_ROWS_BEFORE_CONTEXT:
        raise BuildError(
            f"Expected {OUTPUT_ROWS_BEFORE_CONTEXT} untouched rows before context filtering, "
            f"found {len(rows)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
    context_rows: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["prompt"])},
        ]
        prompt_length = len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        )
        if prompt_length + 768 > 2048:
            rejected["deployment_context_budget"] += 1
            continue
        context_rows.append(row)
        prompt_lengths.append(prompt_length)
    rows = context_rows
    if len(rows) != OUTPUT_ROWS or rejected["deployment_context_budget"] != 1:
        raise BuildError(
            f"Expected {OUTPUT_ROWS} context-safe rows, found {len(rows)}; "
            f"context_rejected={rejected['deployment_context_budget']}"
        )

    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "mbpp_full_untouched.parquet", raw)
    write_bytes(source_dir / "MBPP_README.md", readme)
    write_jsonl(VALIDATION, rows)
    audit = {
        "gate": "P0-A20-MBPP-UNTOUCHED-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a20_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "google-research-datasets/mbpp",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "source_rows": SOURCE_ROWS,
            "source_sha256": SOURCE_SHA256,
            "readme_sha256": README_SHA256,
            "declared_license": "cc-by-4.0",
        },
        "filter": {
            "accepted_rows": len(rows),
            "rejected": dict(sorted(rejected.items())),
            "test_count_distribution": dict(
                sorted(Counter(str(len(row["unit_tests"])) for row in rows).items())
            ),
        },
        "context": {
            "maximum_prompt_tokens": max(prompt_lengths),
            "max_generation_tokens": 768,
            "server_context_tokens": 2048,
            "status": "passed",
        },
        "separation": {
            "history_task_id_overlap_after_filter": 0,
            "history_prompt_overlap_after_filter": 0,
            "opencode_prompt_overlap_after_filter": 0,
            "used_for_training": False,
            "human_eval_loaded": False,
            "gate300_loaded": False,
            "formal_full_loaded": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A19_TRAIN.relative_to(ROOT).as_posix(): sha256_file(P0A19_TRAIN),
            **{f"history:{key}": value for key, value in history_hashes.items()},
        },
        "output": {
            "path": VALIDATION.relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(VALIDATION),
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(f"P0-A20 MBPP remainder passed rows={len(rows)} max_prompt={max(prompt_lengths)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A20 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "filter": value.get("filter")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A20 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
