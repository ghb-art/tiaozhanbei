#!/usr/bin/env python3
"""Build balanced COIG open-QA requests for deterministic MCQ conversion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_p0a5_data import normalize_text


ROOT = Path(__file__).resolve().parents[1]
COIG = ROOT / "data/datasets/coig_cqia/COIG-CQIA-full.jsonl"
HISTORICAL = ROOT / "data/p0a6/train.jsonl"
OUTPUT = ROOT / "data/distill/p0a36_human_verified_openqa_mcq_requests.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a36_human_requests.json"
SEED = 20260802
GROUPS = ("stem", "humanities", "social_law", "general")
REQUESTS_PER_GROUP = 500


class BuildError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            users = [
                item for item in (row.get("messages") or [])
                if isinstance(item, dict) and item.get("role") == "user"
            ]
            if len(users) == 1:
                prompts.add(normalize_text(str(users[0].get("content", ""))))
    if len(prompts) != 9500:
        raise BuildError(f"Historical COIG prompt count changed: {len(prompts)}")
    return prompts


def group_for(domains: list[Any]) -> str | None:
    combined = " ".join(str(value) for value in domains)
    if any(value in combined for value in ("医疗", "药物", "医学", "病症", "生物")):
        return "medicine"
    if any(value in combined for value in (
        "理学", "工学", "数学", "物理", "化学", "电子", "环境", "农学", "计算机"
    )):
        return "stem"
    if any(value in combined for value in (
        "中国传统文化", "语文", "历史", "文学", "艺术", "地理", "文言文"
    )):
        return "humanities"
    if any(value in combined for value in (
        "法律", "政治", "社会", "心理", "教育", "人类价值观", "哲学"
    )):
        return "social_law"
    if any(value in combined for value in ("经济", "金融", "管理")):
        return "business"
    if any(value in combined for value in ("通用", "百科", "逻辑", "常识", "多领域")):
        return "general"
    return None


def chinese_count(value: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", value))


def teacher_messages(question: str, trusted_answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你把可信的中文问答改写成客观四选一考试题。必须保持原知识事实，"
                "生成一个简洁正确项和三个同类型、看似合理但明确错误的干扰项。"
                "不得把正确项位置写成字母，不得包含‘以上都对’或‘以上都不对’。"
                "只返回JSON对象，字段为question、correct、distractors、reason_zh；"
                "distractors必须是恰好三个字符串。"
            ),
        },
        {
            "role": "user",
            "content": f"原问题：{question}\n可信参考答案：{trusted_answer}",
        },
    ]


def main() -> int:
    if OUTPUT.exists() or AUDIT.exists():
        raise BuildError("P0-A36 request outputs already exist; overwrite refused")
    used = historical_prompts()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    with COIG.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("human_verified")):
                rejected["source_not_human_verified"] += 1
                continue
            task = row.get("task_type") or {}
            major = " ".join(str(value) for value in (task.get("major") or []))
            if not any(value in major for value in ("问答", "知识问答", "名词解释", "文本生成")):
                rejected["unsupported_task_shape"] += 1
                continue
            question = (
                f"{str(row.get('instruction', '')).strip()}\n"
                f"{str(row.get('input', '')).strip()}"
            ).strip()
            answer = str(row.get("output", "")).strip()
            if not (5 <= chinese_count(question) <= 600 and 4 <= chinese_count(answer) <= 1000):
                rejected["length_or_language"] += 1
                continue
            identity = normalize_text(question)
            if not identity or identity in seen:
                rejected["duplicate_prompt"] += 1
                continue
            group = group_for(list(row.get("domain") or []))
            if group is None:
                rejected["unmapped_domain"] += 1
                continue
            seen.add(identity)
            digest = sha256_text(identity)
            historical = identity in used
            by_group[group].append(
                {
                    "request_id": f"p0a36/openqa/{group}/{digest[:20]}",
                    "validation_group_id": f"openqa-mcq/{digest[:20]}",
                    "group": group,
                    "domains": [str(value) for value in (row.get("domain") or [])],
                    "source_prompt_hash": digest,
                    "source_seen_as_open_qa": historical,
                    "messages": teacher_messages(question, answer),
                }
            )
    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, dict[str, int]] = {}
    for group in GROUPS:
        fresh = sorted(
            (row for row in by_group[group] if not row["source_seen_as_open_qa"]),
            key=lambda row: sha256_text(f"{SEED}:fresh:{row['request_id']}"),
        )
        replay = sorted(
            (row for row in by_group[group] if row["source_seen_as_open_qa"]),
            key=lambda row: sha256_text(f"{SEED}:replay:{row['request_id']}"),
        )
        chosen = (fresh + replay)[:REQUESTS_PER_GROUP]
        if len(chosen) != REQUESTS_PER_GROUP:
            raise BuildError(
                f"P0-A36 group {group} has only {len(chosen)} eligible rows"
            )
        selected.extend(chosen)
        selected_counts[group] = {
            "total": len(chosen),
            "fresh_source": sum(not row["source_seen_as_open_qa"] for row in chosen),
            "open_qa_replay_source": sum(row["source_seen_as_open_qa"] for row in chosen),
        }
    selected.sort(key=lambda row: sha256_text(f"{SEED}:request:{row['request_id']}"))
    if len(selected) != REQUESTS_PER_GROUP * len(GROUPS):
        raise BuildError("P0-A36 request total mismatch")
    write_jsonl(OUTPUT, selected)
    report = {
        "gate": "P0-A36-BALANCED-OPENQA-MCQ-REQUESTS",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a36_openqa_requests.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "request_count": len(selected),
        "requests_per_group": REQUESTS_PER_GROUP,
        "groups": list(GROUPS),
        "selected_counts": selected_counts,
        "rejections": dict(sorted(rejected.items())),
        "policy": {
            "teacher_receives_trusted_human_verified_answer": True,
            "teacher_does_not_choose_final_option_letter": True,
            "option_order": "assigned deterministically after generation",
            "formal_cmmlu_test_opened": False,
            "p0a34_validation_reused": False,
        },
        "inputs": {
            COIG.relative_to(ROOT).as_posix(): sha256_file(COIG),
            HISTORICAL.relative_to(ROOT).as_posix(): sha256_file(HISTORICAL),
        },
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_hash": sha256_file(OUTPUT),
        "errors": [],
    }
    write_json(AUDIT, report)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(selected)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A36 request build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
