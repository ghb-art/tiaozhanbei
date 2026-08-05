#!/usr/bin/env python3
"""Build fresh COIG Chinese MCQ Teacher requests and a C-Eval dev holdout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from build_p0a5_data import normalize_text


ROOT = Path(__file__).resolve().parents[1]
COIG = ROOT / "data/datasets/coig_cqia/COIG-CQIA-full.jsonl"
HISTORICAL = ROOT / "data/p0a6/train.jsonl"
CEVAL = ROOT / "data/datasets/ceval_exam"
REQUESTS = ROOT / "data/distill/p0a34_coig_mcq_requests.jsonl"
VALIDATION = ROOT / "data/p0a34/nlp_validation.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a34_requests.json"
SEED = 20260802

OPTION_RE = re.compile(r"(?:^|\n)\s*([A-E])[\.．、:：]\s*", re.I)
LABEL_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"(?:正确答案|答案)\s*(?:是|为|选择)?\s*[:：\-]?\s*([A-D])(?![A-Z])",
        r"(?:故|所以|因此)?\s*(?:本题)?\s*(?:选择|选|应选)\s*(?:错误的?)?\s*[:：]?\s*([A-D])(?![A-Z])",
        r"(?:只有|唯有)\s*([A-D])\s*(?:项)?\s*(?:是)?(?:正确|符合|对的|适合)",
        r"([A-D])\s*项?\s*(?:最)?(?:正确|符合(?:要求)?|是对的|适合)(?:。|，|,|$)",
        r"([A-D])\s*(?:才)?是\s*(?:正确|符合|对的|最佳)",
    )
)


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def historical_prompts() -> set[str]:
    prompts: set[str] = set()
    with HISTORICAL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("dataset_key")) != "coig_cqia":
                continue
            messages = row.get("messages") or []
            users = [item for item in messages if isinstance(item, dict) and item.get("role") == "user"]
            if len(users) == 1:
                prompts.add(normalize_text(str(users[0].get("content", ""))))
    if len(prompts) != 9500:
        raise BuildError(f"Historical COIG prompt count changed: {len(prompts)}")
    return prompts


def source_label(output: str) -> str | None:
    labels = {
        label.upper()
        for pattern in LABEL_PATTERNS
        for label in pattern.findall(output)
    }
    return next(iter(labels)) if len(labels) == 1 else None


def group_for(domains: list[Any]) -> str:
    combined = " ".join(str(value) for value in domains)
    if "法律" in combined:
        return "law"
    if any(value in combined for value in ("中国传统文化", "语文", "历史")):
        return "humanities"
    if any(value in combined for value in ("心理", "教育", "政治")):
        return "social_science"
    return "science_technical"


def teacher_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的中文学科考试助手。独立解答四选一题，不猜测题库标签。"
                "返回JSON对象，字段必须是reason_zh和final；reason_zh是一条简短中文理由，"
                "final只能是A、B、C或D。"
            ),
        },
        {"role": "user", "content": prompt},
    ]


def build_requests() -> tuple[list[dict[str, Any]], dict[str, int], Counter[str]]:
    used = historical_prompts()
    candidates: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    with COIG.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task_type") or {}
            major = " ".join(str(value) for value in (task.get("major") or []))
            minor = " ".join(str(value) for value in (task.get("minor") or []))
            if "试题" not in major or "多项" in minor:
                continue
            prompt = f"{str(row.get('instruction', '')).strip()}\n{str(row.get('input', '')).strip()}".strip()
            options = {value.upper() for value in OPTION_RE.findall(prompt)}
            if options != {"A", "B", "C", "D"}:
                rejected["not_exactly_four_options"] += 1
                continue
            label = source_label(str(row.get("output", "")))
            if label is None:
                rejected["source_label_ambiguous"] += 1
                continue
            identity = normalize_text(prompt)
            if identity in used:
                rejected["historical_prompt_overlap"] += 1
                continue
            if identity in candidates:
                rejected["duplicate_prompt"] += 1
                continue
            digest = sha256_text(identity)
            group = group_for(list(row.get("domain") or []))
            candidates[identity] = {
                "request_id": f"p0a34/coig/{group}/{digest[:20]}",
                "validation_group_id": f"coig-mcq/{digest[:20]}",
                "group": group,
                "domains": [str(value) for value in (row.get("domain") or [])],
                "expected_label": label,
                "prompt": prompt,
                "messages": teacher_messages(prompt),
                "source_prompt_hash": digest,
                "split_role": "train",
                "source_answer_from": str(row.get("answer_from", "")),
                "source_human_verified": bool(row.get("human_verified")),
            }
    rows = sorted(
        candidates.values(),
        key=lambda row: sha256_text(f"{SEED}:{row['request_id']}"),
    )
    return rows, dict(Counter(str(row["group"]) for row in rows)), rejected


def build_validation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CEVAL.glob("*/dev-*.parquet")):
        subject = path.parent.name
        for row in pq.read_table(path).to_pylist():
            prompt = (
                f"问题：{row['question']}\nA. {row['A']}\nB. {row['B']}\n"
                f"C. {row['C']}\nD. {row['D']}"
            )
            rows.append(
                {
                    "sample_id": f"ceval/dev/{subject}/{int(row['id']):05d}",
                    "dataset_key": "ceval_dev",
                    "domain": "nlp",
                    "subject": subject,
                    "prompt": prompt,
                    "reference": str(row["answer"]).strip().upper(),
                    "validator": "choice_exact",
                    "split_role": "p0a34_external_validation",
                }
            )
    rows.sort(key=lambda row: str(row["sample_id"]))
    if len(rows) != 260 or len({str(row["sample_id"]) for row in rows}) != 260:
        raise BuildError(f"Unexpected C-Eval dev rows: {len(rows)}")
    return rows


def main() -> int:
    if REQUESTS.exists() or VALIDATION.exists() or AUDIT.exists():
        raise BuildError("P0-A34 outputs already exist; overwrite refused")
    requests, group_counts, rejected = build_requests()
    validation = build_validation()
    if len(requests) < 1200 or len(group_counts) < 4:
        raise BuildError(f"Insufficient fresh COIG MCQ requests: {len(requests)} {group_counts}")
    write_jsonl(REQUESTS, requests)
    write_jsonl(VALIDATION, validation)
    report = {
        "gate": "P0-A34-FRESH-CHINESE-MCQ-REQUESTS",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a34_coig_requests.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "policy": {
            "teacher_prompt_contains_source_label": False,
            "formal_cmmlu_test_opened": False,
            "p0a32_validation_reused": False,
            "selection_manifest": "C-Eval dev only",
        },
        "historical_coig_prompt_count": 9500,
        "request_count": len(requests),
        "request_group_counts": group_counts,
        "rejections": dict(sorted(rejected.items())),
        "validation_rows": len(validation),
        "requests": REQUESTS.relative_to(ROOT).as_posix(),
        "requests_hash": sha256_file(REQUESTS),
        "validation": VALIDATION.relative_to(ROOT).as_posix(),
        "validation_hash": sha256_file(VALIDATION),
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (COIG, HISTORICAL)
        },
        "errors": [],
    }
    write_json(AUDIT, report)
    print(f"Wrote {REQUESTS.relative_to(ROOT)} rows={len(requests)} groups={group_counts}")
    print(f"Wrote {VALIDATION.relative_to(ROOT)} rows={len(validation)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A34 request build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
