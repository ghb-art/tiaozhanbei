#!/usr/bin/env python3
"""Freeze a decimal-only, ASDiv-disjoint Calc-MAWPS holdout for P0-A14."""

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
CONFIG = ROOT / "configs/p0a14_math_self_consistency.json"
OUTPUT_DIR = ROOT / "data/p0a14"
VALIDATION = OUTPUT_DIR / "math_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a14_data.json"
P0A12_TRAIN = ROOT / "data/p0a12/math_train.jsonl"
P0A13_VALIDATION = ROOT / "data/p0a13/math_validation.jsonl"
REVISION = "38c10053efeafd20ab6ff4e08c3ec17de26c19b7"
SOURCE_FILE = "data/validation-00000-of-00001-2ce28573971ca59f.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/MU-NLPC/Calc-mawps/resolve/{REVISION}/{SOURCE_FILE}"
README_URL = f"https://huggingface.co/datasets/MU-NLPC/Calc-mawps/resolve/{REVISION}/README.md"
SOURCE_SHA256 = "9d2ca9e33d8efdbc2527a84f1685d6dd5ec7defb201a290a836660c071f5b607"
README_SHA256 = "dc66dcea61b7cc9fa7380b31277ffef4d719f764a78f0a31b7b80dc825353c13"
SOURCE_ROWS = 1040
DECIMAL_ROWS = 734
VALIDATION_ROWS = 727
DECIMAL = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_number(value: str) -> str:
    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation as exc:
        raise BuildError(f"Invalid Calc-MAWPS result: {value!r}") from exc
    if not number.is_finite():
        raise BuildError(f"Non-finite Calc-MAWPS result: {value!r}")
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a14-data-builder/1.0"})
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


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BuildError("pyarrow is required") from exc
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A14-MATH-SELF-CONSISTENCY":
        raise BuildError("P0-A14 config identity mismatch")

    raw = download(SOURCE_URL, 1024 * 1024)
    readme = download(README_URL, 1024 * 1024)
    if sha256_bytes(raw) != SOURCE_SHA256 or sha256_bytes(readme) != README_SHA256:
        raise BuildError("Calc-MAWPS source hash mismatch")
    source = parquet.read_table(pa.BufferReader(raw)).to_pylist()
    if len(source) != SOURCE_ROWS:
        raise BuildError(f"Unexpected Calc-MAWPS rows: {len(source)}")

    asdiv_prompts = {
        normalize_text(str(row["prompt"])) for row in read_jsonl(P0A13_VALIDATION)
    }
    train_prompts = {
        normalize_text(str(row["messages"][-1]["content"]))
        for row in read_jsonl(P0A12_TRAIN)
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    fraction_rejected = 0
    asdiv_rejected = 0
    train_overlap = 0
    for index, item in enumerate(source):
        result = str(item.get("result", "")).strip().replace(",", "")
        if DECIMAL.fullmatch(result) is None:
            fraction_rejected += 1
            continue
        prompt = str(item.get("question", "")).strip()
        prompt_norm = normalize_text(prompt)
        if not prompt_norm or prompt_norm in seen:
            raise BuildError(f"Duplicate/missing Calc-MAWPS prompt at row {index}")
        seen.add(prompt_norm)
        if prompt_norm in asdiv_prompts:
            asdiv_rejected += 1
            continue
        if prompt_norm in train_prompts:
            train_overlap += 1
            continue
        sample_key = sha256_text(f"{REVISION}:{item.get('id')}:{prompt_norm}")
        rows.append(
            {
                "sample_id": f"calc-mawps/{sample_key[:24]}",
                "dataset_key": "calc_mawps",
                "domain": "math",
                "source": f"MU-NLPC/Calc-mawps@{REVISION}",
                "source_row": index,
                "split_role": "p0a14_external_validation",
                "prompt": prompt,
                "reference": normalize_number(result),
                "validator": "exact_numeric_answer",
                "unit_tests": [],
                "metadata": {
                    "source_id": str(item.get("id", "")),
                    "equation": str(item.get("equation", "")),
                },
            }
        )
    if (
        len(seen) != DECIMAL_ROWS
        or fraction_rejected != SOURCE_ROWS - DECIMAL_ROWS
        or asdiv_rejected != 7
        or train_overlap != 0
        or len(rows) != VALIDATION_ROWS
    ):
        raise BuildError(
            "Unexpected Calc-MAWPS filtering: "
            f"decimal={len(seen)} fraction={fraction_rejected} "
            f"asdiv={asdiv_rejected} train={train_overlap} final={len(rows)}"
        )

    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "calc_mawps_validation.parquet", raw)
    write_bytes(source_dir / "CALC_MAWPS_README.md", readme)
    write_jsonl(VALIDATION, rows)
    audit = {
        "gate": "P0-A14-CALC-MAWPS-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a14_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "MU-NLPC/Calc-mawps",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "source_rows": SOURCE_ROWS,
            "source_sha256": SOURCE_SHA256,
            "readme_sha256": README_SHA256,
            "declared_license": "mit",
        },
        "filter": {
            "decimal_rows": DECIMAL_ROWS,
            "fraction_rows_excluded": fraction_rejected,
            "asdiv_overlap_excluded": asdiv_rejected,
            "p0a12_train_overlap": train_overlap,
            "final_rows": len(rows),
        },
        "separation": {
            "calc_mawps_used_for_training": False,
            "asdiv_or_svamp_reused_for_selection": False,
            "gate300_opened": False,
            "formal_full_opened": False,
        },
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A12_TRAIN.relative_to(ROOT).as_posix(): sha256_file(P0A12_TRAIN),
            P0A13_VALIDATION.relative_to(ROOT).as_posix(): sha256_file(P0A13_VALIDATION),
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
    print(f"P0-A14 Calc-MAWPS data passed rows={len(rows)}/{SOURCE_ROWS}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A14 audit is missing")
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": data.get("status"), "filter": data.get("filter")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A14 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
