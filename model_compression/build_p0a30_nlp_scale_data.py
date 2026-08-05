#!/usr/bin/env python3
"""Build untouched CMMLU-dev validation and two immutable NLP scale variants."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CMMLU_DEV = ROOT / "data/datasets/cmmlu/data/dev"
GATE = ROOT / "data/capability_v2/gate300.jsonl"
SOURCE_ADAPTER = ROOT / "models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
OUTPUT = ROOT / "data/p0a30/nlp_validation.jsonl"
ADAPTER_ROOT = ROOT / "models/adapters/p0a30"
AUDIT = ROOT / "reports/audit/gate_p0a30_data.json"
CHOICES = ("A", "B", "C", "D")


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cmmlu_rows() -> list[dict]:
    rows: list[dict] = []
    files = sorted(CMMLU_DEV.glob("*.csv"))
    if len(files) != 67:
        raise BuildError(f"CMMLU dev subject count changed: {len(files)}")
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                answer = str(row.get("Answer", "")).strip().upper()
                if answer not in CHOICES:
                    raise BuildError(f"Invalid CMMLU answer: {path}:{index}")
                prompt = (
                    f"问题：{row['Question']}\n"
                    f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
                    "请给出简短分析，并在最后一行严格输出“最终答案：X”。"
                )
                rows.append(
                    {
                        "sample_id": f"cmmlu/dev/{path.stem}/{index:05d}",
                        "dataset_key": "cmmlu",
                        "domain": "nlp",
                        "subject": path.stem,
                        "split_role": "p0a30_external_validation",
                        "prompt": prompt,
                        "reference": answer,
                        "validator": "choice_exact",
                        "unit_tests": [],
                    }
                )
    if len(rows) != 335:
        raise BuildError(f"CMMLU dev row count changed: {len(rows)}")
    return rows


def build_scaled_adapter(name: str, alpha: int) -> dict:
    destination = ADAPTER_ROOT / name
    if destination.exists():
        raise BuildError(f"Scale adapter already exists: {destination.relative_to(ROOT)}")
    destination.mkdir(parents=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        source = SOURCE_ADAPTER / filename
        if not source.is_file():
            raise BuildError(f"Missing source adapter file: {source.relative_to(ROOT)}")
        shutil.copy2(source, destination / filename)
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (int(config.get("r", 0)), int(config.get("lora_alpha", 0))) != (16, 32):
        raise BuildError("P0-A10 NLP adapter rank/alpha changed")
    config["lora_alpha"] = alpha
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": destination.relative_to(ROOT).as_posix(),
        "rank": 16,
        "alpha": alpha,
        "scale": alpha / 32,
        "weights_hash": sha256_file(destination / "adapter_model.safetensors"),
        "config_hash": sha256_file(config_path),
    }


def main() -> int:
    try:
        if OUTPUT.exists() or ADAPTER_ROOT.exists() or AUDIT.exists():
            raise BuildError("P0-A30 outputs already exist; overwrite refused")
        gate_rows = [row for row in read_jsonl(GATE) if row.get("domain") == "nlp"]
        gate_ids = {str(row["sample_id"]) for row in gate_rows}
        if len(gate_rows) != 100 or len(gate_ids) != 100:
            raise BuildError("Frozen CMMLU gate is not 100 unique rows")
        all_rows = cmmlu_rows()
        validation = [row for row in all_rows if row["sample_id"] not in gate_ids]
        if len(validation) != 235:
            raise BuildError(f"Expected 235 untouched CMMLU dev rows, found {len(validation)}")
        if gate_ids & {row["sample_id"] for row in validation}:
            raise BuildError("P0-A30 validation overlaps frozen gate")
        subject_counts = Counter(str(row["subject"]) for row in validation)
        if len(subject_counts) != 67 or min(subject_counts.values()) < 2:
            raise BuildError("P0-A30 validation no longer covers all 67 subjects")
        write_jsonl(OUTPUT, sorted(validation, key=lambda row: row["sample_id"]))
        variants = {
            "scale_0p75": build_scaled_adapter("nlp-step136-scale-0p75", 24),
            "scale_1p25": build_scaled_adapter("nlp-step136-scale-1p25", 40),
        }
        source_weight_hash = sha256_file(SOURCE_ADAPTER / "adapter_model.safetensors")
        if any(value["weights_hash"] != source_weight_hash for value in variants.values()):
            raise BuildError("Scaled adapter weights differ from P0-A10 source")
        report = {
            "gate": "P0-A30-NLP-SCALE-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0a30_nlp_scale_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "source_split": "CMMLU dev only",
            "formal_test_opened": False,
            "frozen_gate_rows_excluded": len(gate_ids),
            "validation_rows": len(validation),
            "validation_subjects": len(subject_counts),
            "minimum_rows_per_subject": min(subject_counts.values()),
            "validation": OUTPUT.relative_to(ROOT).as_posix(),
            "validation_hash": sha256_file(OUTPUT),
            "source_adapter": SOURCE_ADAPTER.relative_to(ROOT).as_posix(),
            "source_adapter_weights_hash": source_weight_hash,
            "variants": variants,
            "errors": [],
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json(AUDIT, report)
        print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(validation)}")
        print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
        return 0
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A30 data build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
