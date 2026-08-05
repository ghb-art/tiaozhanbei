#!/usr/bin/env python3
"""Freeze the untouched decimal Calc-MAWPS train split for P0-A16 validation."""

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
CONFIG = ROOT / "configs/p0a16_math_joint_runtime.json"
OUTPUT_DIR = ROOT / "data/p0a16"
VALIDATION = OUTPUT_DIR / "math_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a16_data.json"
HISTORY = tuple(ROOT / f"data/p0a{version}/math_validation.jsonl" for version in (12, 13, 14, 15))
REVISION = "38c10053efeafd20ab6ff4e08c3ec17de26c19b7"
SOURCE_FILE = "data/train-00000-of-00001-4bb1451333aad61c.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/MU-NLPC/Calc-mawps/resolve/{REVISION}/{SOURCE_FILE}"
SOURCE_SHA256 = "7a8dd8f7680b5e5ddb908cdfa33867c298d9225d1d7595a17d4771c09e482140"
SOURCE_ROWS = 1089
DECIMAL_ROWS = 1044
FINAL_ROWS = 1041
DECIMAL = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


class BuildError(RuntimeError):
    pass


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_number(value: str) -> str:
    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation as exc:
        raise BuildError(f"Invalid result: {value!r}") from exc
    if not number.is_finite():
        raise BuildError(f"Non-finite result: {value!r}")
    result = format(number.normalize(), "f")
    return "0" if number == 0 else result


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing history: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BuildError("pyarrow is required") from exc
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A16-MATH-JOINT-RUNTIME":
        raise BuildError("P0-A16 config identity mismatch")
    request = Request(SOURCE_URL, headers={"User-Agent": "p0a16-data-builder/1.0"})
    with urlopen(request, timeout=120) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or digest(raw) != SOURCE_SHA256:
        raise BuildError("Calc-MAWPS train source size/hash mismatch")
    source = parquet.read_table(pa.BufferReader(raw)).to_pylist()
    if len(source) != SOURCE_ROWS:
        raise BuildError(f"Unexpected source rows: {len(source)}")
    history = {
        normalize_text(str(row["prompt"])) for path in HISTORY for row in rows(path)
    }
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    fraction = overlap = 0
    for index, item in enumerate(source):
        result = str(item.get("result", "")).strip().replace(",", "")
        if DECIMAL.fullmatch(result) is None:
            fraction += 1
            continue
        prompt = str(item.get("question", "")).strip()
        normalized = normalize_text(prompt)
        if not normalized or normalized in seen:
            raise BuildError(f"Duplicate/missing prompt: {index}")
        seen.add(normalized)
        if normalized in history:
            overlap += 1
            continue
        key = text_digest(f"{REVISION}:train:{item.get('id')}:{normalized}")
        output.append(
            {
                "sample_id": f"calc-mawps-train/{key[:24]}",
                "dataset_key": "calc_mawps",
                "domain": "math",
                "source": f"MU-NLPC/Calc-mawps@{REVISION}",
                "source_row": index,
                "split_role": "p0a16_external_validation",
                "prompt": prompt,
                "reference": normalize_number(result),
                "validator": "exact_numeric_answer",
                "unit_tests": [],
            }
        )
    if len(seen) != DECIMAL_ROWS or fraction != 45 or overlap != 3 or len(output) != FINAL_ROWS:
        raise BuildError(
            f"Unexpected filter decimal={len(seen)} fraction={fraction} overlap={overlap} final={len(output)}"
        )
    atomic(OUTPUT_DIR / "sources/calc_mawps_train.parquet", raw)
    atomic(
        VALIDATION,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output).encode(),
    )
    audit = {
        "gate": "P0-A16-CALC-MAWPS-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a16_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "MU-NLPC/Calc-mawps",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
            "rows": SOURCE_ROWS,
        },
        "filter": {
            "decimal_rows": DECIMAL_ROWS,
            "fraction_rows_excluded": fraction,
            "history_overlap_excluded": overlap,
            "final_rows": len(output),
        },
        "separation": {
            "validation_used_for_training": False,
            "historical_validation_reused": False,
            "gate300_opened": False,
            "formal_full_opened": False,
        },
        "history_hashes": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in HISTORY},
        "output": {
            "path": VALIDATION.relative_to(ROOT).as_posix(),
            "rows": len(output),
            "sha256": file_digest(VALIDATION),
        },
        "errors": [],
    }
    audit["report_hash"] = text_digest(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(f"P0-A16 data passed rows={len(output)}/{SOURCE_ROWS}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    print(AUDIT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A16 build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
