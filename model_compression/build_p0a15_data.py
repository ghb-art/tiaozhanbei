#!/usr/bin/env python3
"""Freeze P0-A15 Calc-MAWPS test holdout and a half-scale LoRA view."""

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
CONFIG = ROOT / "configs/p0a15_math_scaled_adapter.json"
OUTPUT_DIR = ROOT / "data/p0a15"
VALIDATION = OUTPUT_DIR / "math_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a15_data.json"
SOURCE_ADAPTER = ROOT / "models/checkpoints/p0a11/math-specialist/checkpoint-64"
HALF_ADAPTER = ROOT / "models/adapters/p0a15/math-step64-scale-0p5"
HISTORY = (
    ROOT / "data/p0a12/math_validation.jsonl",
    ROOT / "data/p0a13/math_validation.jsonl",
    ROOT / "data/p0a14/math_validation.jsonl",
)
REVISION = "38c10053efeafd20ab6ff4e08c3ec17de26c19b7"
SOURCE_FILE = "data/test-00000-of-00001-5a59f3fc4b0d9c98.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/MU-NLPC/Calc-mawps/resolve/{REVISION}/{SOURCE_FILE}"
SOURCE_SHA256 = "2ef6313cb811d5c5ebef422a909015f7db4750020f835fd95c2ac2ccbf343d82"
SOURCE_ROWS = 520
DECIMAL_ROWS = 350
FINAL_ROWS = 346
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
        raise BuildError(f"Invalid numeric answer: {value!r}") from exc
    normalized = format(number.normalize(), "f")
    return "0" if number == 0 else normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a15-data-builder/1.0"})
    with urlopen(request, timeout=120) as response:
        value = response.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise BuildError("Calc-MAWPS test download exceeded safety cap")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    value = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_text(path, value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_half_adapter() -> dict[str, Any]:
    config_path = SOURCE_ADAPTER / "adapter_config.json"
    weights_path = SOURCE_ADAPTER / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise BuildError("P0-A11 Math step64 adapter is incomplete")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_config.get("r") != 8 or source_config.get("lora_alpha") != 16:
        raise BuildError("Unexpected P0-A11 Math step64 rank/alpha")
    scaled_config = dict(source_config)
    scaled_config["lora_alpha"] = 8
    HALF_ADAPTER.mkdir(parents=True, exist_ok=True)
    target_config = HALF_ADAPTER / "adapter_config.json"
    target_weights = HALF_ADAPTER / "adapter_model.safetensors"
    expected_config = json.dumps(scaled_config, ensure_ascii=False, indent=2) + "\n"
    if target_config.exists():
        if target_config.read_text(encoding="utf-8") != expected_config:
            raise BuildError("Existing half-scale adapter config differs")
    else:
        atomic_text(target_config, expected_config)
    if target_weights.exists():
        if sha256_file(target_weights) != sha256_file(weights_path):
            raise BuildError("Existing half-scale adapter weights differ")
    else:
        os.link(weights_path, target_weights)
    return {
        "source": SOURCE_ADAPTER.relative_to(ROOT).as_posix(),
        "source_rank": 8,
        "source_alpha": 16,
        "scaled": HALF_ADAPTER.relative_to(ROOT).as_posix(),
        "scaled_rank": 8,
        "scaled_alpha": 8,
        "weights_sha256": sha256_file(weights_path),
        "weights_same_inode": weights_path.stat().st_ino == target_weights.stat().st_ino,
        "source_config_sha256": sha256_file(config_path),
        "scaled_config_sha256": sha256_file(target_config),
    }


def build() -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BuildError("pyarrow is required") from exc
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A15-MATH-SCALED-ADAPTER":
        raise BuildError("P0-A15 config identity mismatch")
    adapter = prepare_half_adapter()
    raw = download(SOURCE_URL, 1024 * 1024)
    if sha256_bytes(raw) != SOURCE_SHA256:
        raise BuildError("Calc-MAWPS test hash mismatch")
    source = parquet.read_table(pa.BufferReader(raw)).to_pylist()
    if len(source) != SOURCE_ROWS:
        raise BuildError(f"Unexpected source rows: {len(source)}")
    history_prompts = {
        normalize_text(str(row["prompt"]))
        for path in HISTORY
        for row in read_jsonl(path)
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    fraction_rejected = 0
    history_rejected = 0
    for index, item in enumerate(source):
        result = str(item.get("result", "")).strip().replace(",", "")
        if DECIMAL.fullmatch(result) is None:
            fraction_rejected += 1
            continue
        prompt = str(item.get("question", "")).strip()
        normalized = normalize_text(prompt)
        if not normalized or normalized in seen:
            raise BuildError(f"Duplicate/missing Calc-MAWPS test prompt: {index}")
        seen.add(normalized)
        if normalized in history_prompts:
            history_rejected += 1
            continue
        sample_hash = sha256_text(f"{REVISION}:test:{item.get('id')}:{normalized}")
        rows.append(
            {
                "sample_id": f"calc-mawps-test/{sample_hash[:24]}",
                "dataset_key": "calc_mawps",
                "domain": "math",
                "source": f"MU-NLPC/Calc-mawps@{REVISION}",
                "source_row": index,
                "split_role": "p0a15_external_validation",
                "prompt": prompt,
                "reference": normalize_number(result),
                "validator": "exact_numeric_answer",
                "unit_tests": [],
                "metadata": {"source_id": str(item.get("id", ""))},
            }
        )
    if (
        len(seen) != DECIMAL_ROWS
        or fraction_rejected != SOURCE_ROWS - DECIMAL_ROWS
        or history_rejected != 4
        or len(rows) != FINAL_ROWS
    ):
        raise BuildError(
            f"Unexpected P0-A15 filter decimal={len(seen)} fraction={fraction_rejected} "
            f"history={history_rejected} final={len(rows)}"
        )
    atomic_bytes(OUTPUT_DIR / "sources/calc_mawps_test.parquet", raw)
    write_jsonl(VALIDATION, rows)
    audit = {
        "gate": "P0-A15-DATA-AND-SCALED-ADAPTER",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a15_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": {
            "repo": "MU-NLPC/Calc-mawps",
            "revision": REVISION,
            "file": SOURCE_FILE,
            "source_sha256": SOURCE_SHA256,
            "source_rows": SOURCE_ROWS,
        },
        "filter": {
            "decimal_rows": DECIMAL_ROWS,
            "fraction_rows_excluded": fraction_rejected,
            "history_overlap_excluded": history_rejected,
            "final_rows": len(rows),
        },
        "adapter": adapter,
        "separation": {
            "validation_used_for_training": False,
            "historical_validation_reused": False,
            "gate300_opened": False,
            "formal_full_opened": False,
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
    print(f"P0-A15 data passed rows={len(rows)} scales=0.5,1.0")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A15 audit is missing")
    print(AUDIT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A15 build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
