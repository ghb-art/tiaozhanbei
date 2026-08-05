#!/usr/bin/env python3
"""Generate label-verified rationales for fresh P0-A34 Chinese MCQs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TARGET = 820
LAW_CAP = 680


class GenerateError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
        raise GenerateError(f"Missing input: {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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


def latest_trace(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for row in read_jsonl(path):
        latest[str(row["request_id"])] = row
    return latest


def teacher_request(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout: float,
    seed: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    choices = result.get("choices") or []
    if not choices:
        raise GenerateError("Teacher returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise GenerateError("Teacher returned empty content")
    return content.strip()


def parse_response(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise GenerateError("teacher_response_not_json")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise GenerateError("teacher_response_not_object")
    return value


def validate(row: dict[str, Any], response: str, model_id: str) -> tuple[dict[str, Any] | None, str]:
    try:
        value = parse_response(response)
    except (GenerateError, json.JSONDecodeError) as exc:
        return None, str(exc)
    reason = str(value.get("reason_zh", "")).strip()
    final = str(value.get("final", "")).strip().upper()
    if final not in {"A", "B", "C", "D"}:
        return None, "invalid_final"
    if final != str(row["expected_label"]):
        return None, "teacher_label_mismatch"
    if len(re.findall(r"[\u3400-\u9fff]", reason)) < 4:
        return None, "reason_too_short_or_not_chinese"
    return {
        "sample_id": str(row["request_id"]),
        "dataset_key": "coig_mcq_verified",
        "domain": "nlp",
        "task_id": "nlp",
        "source": "COIG-CQIA-Chinese-MCQ-14B-verified",
        "split_role": "train",
        "messages": [
            {
                "role": "system",
                "content": (
                    "请简要分析这道中文选择题，并在最后一行按“最终答案：A”的格式作答；"
                    "请将A替换为实际选项，只能使用A、B、C或D，禁止输出占位符。"
                ),
            },
            {"role": "user", "content": str(row["prompt"])},
        ],
        "answer": f"简短分析：{reason.rstrip('。')}。\n最终答案：{final}",
        "answer_letter": final,
        "answer_token_position": "last",
        "answer_token_weight": 2.0,
        "quality_weight": 1.0,
        "training_weight": 3.0,
        "kl_weight": 0.10,
        "validation_group_id": str(row["validation_group_id"]),
        "teacher_model_id": model_id,
        "distill_validation": "teacher_choice_matches_source_human_label",
        "metadata": {
            "coig_group": str(row["group"]),
            "source_domains": list(row.get("domains") or []),
            "source_prompt_hash": str(row["source_prompt_hash"]),
            "teacher_prompt_contains_source_label": False,
        },
    }, ""


def select_rows(accepted: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted.values():
        group = str((row.get("metadata") or {}).get("coig_group", ""))
        by_group[group].append(row)
    for group in by_group:
        by_group[group].sort(key=lambda row: sha256_text(f"20260802:{row['sample_id']}"))
    non_law = sorted(
        [row for group, rows in by_group.items() if group != "law" for row in rows],
        key=lambda row: sha256_text(f"20260802:nonlaw:{row['sample_id']}"),
    )
    law = by_group.get("law", [])[:LAW_CAP]
    available = [*non_law, *law]
    if len(non_law) < TARGET - LAW_CAP or len(available) < TARGET:
        return None
    selected_non_law = non_law[: max(TARGET - len(law), TARGET - LAW_CAP)]
    remaining = TARGET - len(selected_non_law)
    selected = [*selected_non_law, *law[:remaining]]
    if len(selected) != TARGET or sum(
        str((row.get("metadata") or {}).get("coig_group")) == "law" for row in selected
    ) > LAW_CAP:
        raise GenerateError("P0-A34 group cap selection failed")
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="data/distill/p0a34_coig_mcq_requests.jsonl")
    parser.add_argument("--trace", default="data/distill/p0a34_coig_mcq_trace.jsonl")
    parser.add_argument("--output", default="data/p0a34/coig_verified_train.jsonl")
    parser.add_argument("--audit", default="reports/audit/gate_p0a34_teacher_data.json")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=120)
    args = parser.parse_args()
    requests_path = resolve(args.requests)
    trace_path = resolve(args.trace)
    output_path = resolve(args.output)
    audit_path = resolve(args.audit)
    requests = read_jsonl(requests_path)
    previous = latest_trace(trace_path)
    accepted = {
        request_id: row["verified_row"]
        for request_id, row in previous.items()
        if isinstance(row.get("verified_row"), dict)
    }
    completed = set(previous)
    pending = [row for row in requests if str(row["request_id"]) not in completed]
    pending.sort(key=lambda row: sha256_text(f"20260802:pending:{row['request_id']}"))

    def process(row: dict[str, Any]) -> dict[str, Any]:
        response = ""
        reason = ""
        try:
            response = teacher_request(
                args.endpoint,
                args.model_id,
                list(row["messages"]),
                args.timeout_sec,
                20260802,
            )
            verified, reason = validate(row, response, args.model_id)
            return {
                "request_id": row["request_id"],
                "status": "accepted" if verified is not None else "rejected",
                "reason": reason,
                "model_id": args.model_id,
                "response": response,
                "verified_row": verified,
            }
        except (HTTPError, URLError, TimeoutError, GenerateError, json.JSONDecodeError) as exc:
            return {
                "request_id": row["request_id"],
                "status": "rejected",
                "reason": f"{type(exc).__name__}:{exc}",
                "model_id": args.model_id,
                "response": response,
                "verified_row": None,
            }

    selected = select_rows(accepted)
    processed = 0
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(0, len(pending), args.workers * 4):
                if selected is not None:
                    break
                batch = pending[offset : offset + args.workers * 4]
                futures = {executor.submit(process, row): row for row in batch}
                for future in as_completed(futures):
                    result = future.result()
                    stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    processed += 1
                    if isinstance(result.get("verified_row"), dict):
                        accepted[str(result["request_id"])] = result["verified_row"]
                    if processed % 50 == 0:
                        print(
                            f"[P0-A34 Teacher] {processed}/{len(pending)} "
                            f"accepted_total={len(accepted)}/{len(requests)}",
                            flush=True,
                        )
                selected = select_rows(accepted)
    if selected is None:
        raise GenerateError(
            f"Verified rows cannot satisfy target/cap: accepted={len(accepted)}"
        )
    write_jsonl(output_path, selected)
    latest = latest_trace(trace_path)
    selected_groups = Counter(
        str((row.get("metadata") or {}).get("coig_group")) for row in selected
    )
    report = {
        "gate": "P0-A34-COIG-MCQ-TEACHER-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/generate_p0a34_coig_mcq.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "teacher_model_id": args.model_id,
        "request_count": len(requests),
        "accepted_count": len(accepted),
        "selected_count": len(selected),
        "selected_group_counts": dict(sorted(selected_groups.items())),
        "law_cap": LAW_CAP,
        "rejection_counts": dict(
            Counter(
                str(row.get("reason", ""))
                for row in latest.values()
                if not isinstance(row.get("verified_row"), dict)
            )
        ),
        "policy": {
            "teacher_prompt_contains_source_label": False,
            "acceptance": "teacher final exactly matches parsed source human label",
            "formal_cmmlu_test_opened": False,
        },
        "requests": requests_path.relative_to(ROOT).as_posix(),
        "requests_hash": sha256_file(requests_path),
        "trace": trace_path.relative_to(ROOT).as_posix(),
        "trace_hash": sha256_file(trace_path),
        "output": output_path.relative_to(ROOT).as_posix(),
        "output_hash": sha256_file(output_path),
        "errors": [],
    }
    write_json(audit_path, report)
    print(
        f"P0-A34 verified={len(selected)}/{TARGET} groups={dict(selected_groups)} "
        "status=passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GenerateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A34 Teacher data failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
