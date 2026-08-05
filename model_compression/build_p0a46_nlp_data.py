#!/usr/bin/env python3
"""Build the isolated P0-A46 NLP corpus from already verified train-only rows."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a46_nlp_isolated.json"
SOURCE = ROOT / "data/p0a45/train.jsonl"
SOURCE_AUDIT = ROOT / "reports/audit/gate_p0a45_data.json"
OUTPUT = ROOT / "data/p0a46/nlp_train.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a46_data.json"
VALIDATION = (
    ROOT / "data/p0a44/nlp_ceval_dev.jsonl",
    ROOT / "data/p0a44/nlp_cmmlu_dev.jsonl",
)
FINAL_RE = re.compile(r"(?:最终答案|FINAL)\s*[:：]\s*([ABCD])", re.I)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise BuildError(f"Non-object row: {path.relative_to(ROOT)}:{number}")
            rows.append(row)
    return rows


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    prompts = [
        str(message.get("content", "")).strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return prompts[-1] if prompts else ""


def prompt_identity(value: str) -> str:
    return sha256_text(" ".join(value.split()))


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if source_audit.get("status") != "passed":
        raise BuildError("P0-A45 verified-data audit is not passed")

    expected = int(config["training"]["rows"])
    kl_weight = float(config["training"]["kl_weight"])
    answer_weight = float(config["training"]["answer_token_weight"])
    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    source_counts: Counter[str] = Counter()
    for row in read_jsonl(SOURCE):
        if row.get("dataset_key") != "cmmlu" or row.get("domain") != "nlp":
            continue
        prompt = user_prompt(row)
        answer = str(row.get("answer", "")).strip()
        matches = FINAL_RE.findall(answer)
        identity = prompt_identity(prompt) if prompt else ""
        if not identity or identity in seen_prompts or not matches:
            raise BuildError(f"Invalid or duplicate verified NLP row: {row.get('sample_id')}")
        seen_prompts.add(identity)
        source_counts[str(row.get("source", "unknown"))] += 1
        selected.append(
            {
                "sample_id": f"p0a46/nlp/{identity[:24]}",
                "dataset_key": "cmmlu",
                "domain": "nlp",
                "source": str(row.get("source", "p0a45_verified_mcq")),
                "split_role": "train",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
                            "请将A替换为实际选项，只能使用A、B、C或D，禁止输出占位符。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "answer": answer,
                "answer_letter": matches[-1].upper(),
                "answer_token_weight": answer_weight,
                "training_weight": 1.0,
                "kl_weight": kl_weight,
                "distill_validation": "14b_choice_or_human_label_verified_upstream",
            }
        )
    if len(selected) != expected:
        raise BuildError(f"Unexpected NLP row count: {len(selected)} != {expected}")

    validation_ids: set[str] = set()
    validation_counts: Counter[str] = Counter()
    for path in VALIDATION:
        for row in read_jsonl(path):
            prompt = str(row.get("prompt", "")).strip()
            validation_ids.add(prompt_identity(prompt))
            validation_counts[path.stem] += 1
    overlap = seen_prompts & validation_ids
    if overlap:
        raise BuildError(f"Training/internal-validation prompt overlap: {len(overlap)}")

    atomic_jsonl(OUTPUT, selected)
    audit: dict[str, Any] = {
        "gate": "P0-A46-NLP-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a46_nlp_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_hash": sha256_file(SOURCE),
        "source_audit": SOURCE_AUDIT.relative_to(ROOT).as_posix(),
        "train_rows": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "validation_counts": dict(sorted(validation_counts.items())),
        "train_validation_prompt_overlap": 0,
        "formal_test_rows_loaded": 0,
        "settings": {"kl_weight": kl_weight, "answer_token_weight": answer_weight},
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_hash": sha256_file(OUTPUT),
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    atomic_json(AUDIT, audit)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(selected)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed overlap=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A46 data build failed: {exc}")
        raise SystemExit(1)
