#!/usr/bin/env python3
"""Freeze the numeric ASDiv holdout for P0-A13 runtime-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a13_math_runtime.json"
OUTPUT_DIR = ROOT / "data/p0a13"
VALIDATION = OUTPUT_DIR / "math_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a13_data.json"
P0A12_TRAIN = ROOT / "data/p0a12/math_train.jsonl"
REVISION = "8f95807222d87b4c688c3c22a6ba2801e1fa03e2"
SOURCE_FILE = "asdiv/validation-00000-of-00001.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/EleutherAI/asdiv/resolve/{REVISION}/{SOURCE_FILE}"
README_URL = f"https://huggingface.co/datasets/EleutherAI/asdiv/resolve/{REVISION}/README.md"
SOURCE_SHA256 = "79dbad6536fe3e8cd449e67da859d7a6f5ac0d8f852efe563742ca5faee7fb75"
README_SHA256 = "78e51745c443bc98917621e76d414327e9f66045fef967a90deb3b25b9def70b"
SOURCE_ROWS = 2305
NUMERIC_ROWS_BEFORE_DEDUP = 2238
NUMERIC_ROWS = 2237
NUMBER_PREFIX = re.compile(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?![/%])")


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


def normalize_number(value: str) -> str:
    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation as exc:
        raise BuildError(f"Invalid numeric ASDiv answer: {value!r}") from exc
    if not number.is_finite():
        raise BuildError(f"Non-finite ASDiv answer: {value!r}")
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a13-data-builder/1.0"})
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


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BuildError("pyarrow is required to read the frozen ASDiv source") from exc

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A13-MATH-RUNTIME":
        raise BuildError("P0-A13 config identity mismatch")
    if config["validation"].get("revision") != REVISION:
        raise BuildError("P0-A13 ASDiv revision mismatch")

    raw = download(SOURCE_URL, 2 * 1024 * 1024)
    readme = download(README_URL, 1024 * 1024)
    if sha256_bytes(raw) != SOURCE_SHA256:
        raise BuildError("ASDiv parquet hash mismatch")
    if sha256_bytes(readme) != README_SHA256:
        raise BuildError("ASDiv README hash mismatch")

    table = parquet.read_table(pa.BufferReader(raw))
    if table.num_rows != SOURCE_ROWS:
        raise BuildError(f"Unexpected ASDiv source rows: {table.num_rows}")
    rows: list[dict[str, Any]] = []
    rejected_non_numeric = 0
    rejected_duplicate = 0
    seen_prompts: set[str] = set()
    for index, item in enumerate(table.to_pylist()):
        body = str(item.get("body", "")).strip()
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        match = NUMBER_PREFIX.match(answer)
        if match is None:
            rejected_non_numeric += 1
            continue
        if not body or not question:
            raise BuildError(f"ASDiv row {index} lacks body/question")
        prompt = f"{body} {question}"
        prompt_norm = normalize_text(prompt)
        if prompt_norm in seen_prompts:
            rejected_duplicate += 1
            continue
        seen_prompts.add(prompt_norm)
        sample_hash = sha256_text(f"{REVISION}:{index}:{prompt_norm}")
        rows.append(
            {
                "sample_id": f"asdiv/{sample_hash[:24]}",
                "dataset_key": "asdiv",
                "domain": "math",
                "source": f"EleutherAI/asdiv@{REVISION}",
                "source_row": index,
                "split_role": "p0a13_external_validation",
                "prompt": prompt,
                "reference": normalize_number(match.group(1)),
                "validator": "exact_numeric_answer",
                "unit_tests": [],
                "metadata": {
                    "solution_type": str(item.get("solution_type", "")),
                    "formula": str(item.get("formula", "")),
                },
            }
        )
    if (
        len(rows) != NUMERIC_ROWS
        or rejected_non_numeric != SOURCE_ROWS - NUMERIC_ROWS_BEFORE_DEDUP
        or rejected_duplicate != NUMERIC_ROWS_BEFORE_DEDUP - NUMERIC_ROWS
    ):
        raise BuildError(
            "Unexpected ASDiv numeric filter: "
            f"accepted={len(rows)} non_numeric={rejected_non_numeric} "
            f"duplicate={rejected_duplicate}"
        )

    train_prompts = {
        normalize_text(str(row["messages"][-1]["content"]))
        for row in read_jsonl(P0A12_TRAIN)
    }
    overlap = train_prompts.intersection(seen_prompts)
    if overlap:
        raise BuildError(f"P0-A13 ASDiv overlaps P0-A12 Math train: {len(overlap)}")

    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "asdiv_validation.parquet", raw)
    write_bytes(source_dir / "ASDIV_README.md", readme)
    write_jsonl(VALIDATION, rows)
    audit = {
        "gate": "P0-A13-ASDIV-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a13_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "EleutherAI/asdiv",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "source_rows": SOURCE_ROWS,
            "source_sha256": SOURCE_SHA256,
            "readme_sha256": README_SHA256,
            "declared_license": "cc-by-nc-4.0",
        },
        "filter": {
            "policy": "leading finite decimal; fractions, percentages and categorical answers excluded",
            "accepted_numeric_rows": len(rows),
            "rejected_non_numeric_rows": rejected_non_numeric,
            "rejected_duplicate_prompt_rows": rejected_duplicate,
        },
        "separation": {
            "p0a12_train_prompt_overlap": 0,
            "asdiv_used_for_training": False,
            "svamp_reused": False,
            "gate300_opened": False,
            "formal_full_opened": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A12_TRAIN.relative_to(ROOT).as_posix(): sha256_file(P0A12_TRAIN),
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
    print(f"P0-A13 ASDiv data passed numeric={len(rows)}/{SOURCE_ROWS}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A13 audit is missing")
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": data.get("status"), "filter": data.get("filter")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A13 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
