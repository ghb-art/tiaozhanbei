#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4_remediation.json"
FINAL_RE = re.compile(
    r"(?:^|\s)FINAL\s*[:：]\s*([ABCD])\s*[。.!]?\s*$",
    re.IGNORECASE,
)
ANY_FINAL_RE = re.compile(r"\bFINAL\s*[:：]\s*([ABCD])\b", re.IGNORECASE)
FORBIDDEN_IDENTITY_MARKERS = (
    "cmmlu/test/",
    "gsm8k/test/",
    "humaneval/",
    "official_full",
    "final_test",
    "selection170",
    "smoke96",
)
RATIONALE_SYSTEM = (
    "You are a careful training-data teacher. Solve the multiple-choice question using a short, "
    "fact-focused reason. Do not mention an answer key, dataset label, or hidden reference."
)
RATIONALE_INSTRUCTIONS = (
    "请先给出一条简短、基于题目与选项内容的理由，最后一行严格写成 FINAL: X，其中X只能是A、B、C或D。",
    "请优先用排除法或关键概念给出简短理由，不要引用题库答案；最后一行严格写成 FINAL: X。",
)
RATIONALE_RETRY_INSTRUCTIONS = (
    "请重新独立核对题干和全部选项。只写一到两句、最多120个汉字（英文最多50词）的关键理由；最后单独一行严格写 FINAL: X。",
    "请换一种解法重新判断，重点检查容易混淆的选项。理由必须简短且自洽，最后单独一行严格写 FINAL: X。",
)


class RationaleError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RationaleError(f"Missing JSONL: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RationaleError(f"Expected object at {display_path(path)}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("validation_group_id") or row.get("sample_id") or "")


def source_identity(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "")).lower()
        for key in ("sample_id", "validation_group_id", "source", "split", "split_role")
    )


def load_unique_train_rows(path: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("dataset_key") != "cmmlu" or row.get("used_for_training") is not True:
            continue
        identity = source_identity(row)
        if any(marker in identity for marker in FORBIDDEN_IDENTITY_MARKERS):
            raise RationaleError(f"Forbidden evaluation identity in NLP source: {row.get('sample_id')}")
        group = group_id(row)
        messages = row.get("messages")
        expected = str(row.get("answer", "")).strip().upper()
        if not group or not isinstance(messages, list) or expected not in {"A", "B", "C", "D"}:
            raise RationaleError(f"Incomplete NLP training row: {row.get('sample_id')}")
        unique.setdefault(group, row)
    if not unique:
        raise RationaleError("No eligible train-only CMMLU rows")
    return [unique[key] for key in sorted(unique)]


def rationale_messages(
    row: dict[str, Any],
    variant: int,
    retry_attempt: int = 0,
) -> list[dict[str, str]]:
    source_messages = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in row["messages"]
        if isinstance(item, dict) and item.get("role") in {"system", "user"}
    ]
    user = "\n".join(item["content"] for item in source_messages if item["role"] == "user").strip()
    user = re.sub(
        r"(?:以下是单项选择题。)?只输出一个大写字母\s*A、B、C\s*或\s*D。[ \t]*",
        "以下是单项选择题。\n",
        user,
    )
    user = re.sub(
        r"(?:This is a single-choice question\.\s*)?Return only one uppercase letter A, B, C, or D\.\s*",
        "This is a single-choice question.\n",
        user,
        flags=re.IGNORECASE,
    )
    if retry_attempt >= 2:
        expected = str(row["answer"]).strip().upper()
        instruction = (
            f"这是训练集内部的解释修复任务，目标选项为 {expected}。"
            "请仅依据题干和选项内容写出支持该选项的一到两句自洽解释，"
            "不要提到答案标签、标准答案或隐藏参考；"
            f"最后单独一行严格写 FINAL: {expected}。"
        )
    elif retry_attempt > 0:
        instruction = RATIONALE_RETRY_INSTRUCTIONS[
            (retry_attempt + variant - 1) % len(RATIONALE_RETRY_INSTRUCTIONS)
        ]
    else:
        instruction = RATIONALE_INSTRUCTIONS[variant % len(RATIONALE_INSTRUCTIONS)]
    return [
        {"role": "system", "content": RATIONALE_SYSTEM},
        {"role": "user", "content": f"{user.strip()}\n\n{instruction}"},
    ]


def parse_verified_rationale(
    response: str,
    expected: str,
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str, str]:
    cleaned = response.replace("<think>", "").replace("</think>", "").strip()
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    match = FINAL_RE.search(cleaned)
    if not match:
        return False, "missing_final_marker", ""
    predicted = match.group(1).upper()
    if predicted != expected:
        return False, f"choice_mismatch expected={expected} predicted={predicted}", ""
    rationale = cleaned[: match.start()].strip()
    rationale = re.sub(r"^(?:理由|分析|Reason)\s*[:：]\s*", "", rationale, flags=re.IGNORECASE)
    if len(rationale) < min_chars:
        return False, "rationale_too_short", ""
    if len(rationale) > max_chars:
        return False, "rationale_too_long", ""
    lowered = rationale.lower()
    if any(marker in lowered for marker in ("标准答案", "题库答案", "给定答案", "answer key", "ground truth")):
        return False, "answer_key_leakage_phrase", ""
    if ANY_FINAL_RE.search(rationale):
        return False, "multiple_final_markers", ""
    return True, "choice_and_rationale_verified", f"理由：{rationale}\nFINAL: {expected}"


def trace_with_hash(trace: dict[str, Any]) -> dict[str, Any]:
    value = dict(trace)
    value.pop("trace_row_hash", None)
    value["trace_row_hash"] = sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return value


def revalidate_trace(
    trace: dict[str, Any],
    min_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    accepted, verification, normalized = parse_verified_rationale(
        str(trace.get("teacher_response", "")),
        str(trace.get("expected_choice", "")).strip().upper(),
        min_chars,
        max_chars,
    )
    accepted_for_training = bool(accepted and trace.get("dry_run") is False)
    if (
        trace.get("accepted_for_training") is accepted_for_training
        and trace.get("verification") == verification
        and trace.get("normalized_answer", "") == (normalized if accepted else "")
    ):
        return trace
    refreshed = dict(trace)
    refreshed["parser_revalidation"] = {
        "previous_accepted_for_training": trace.get("accepted_for_training"),
        "previous_verification": trace.get("verification"),
    }
    refreshed["accepted_for_training"] = accepted_for_training
    refreshed["verification"] = verification
    refreshed["normalized_answer"] = normalized if accepted else ""
    return trace_with_hash(refreshed)


def select_generation_tasks(
    train_sources: list[dict[str, Any]],
    variants: int,
    existing: dict[str, dict[str, Any]],
    retry_rejected: bool,
    min_accepted_variants: int,
    max_rejected_retry_rounds: int,
) -> list[tuple[dict[str, Any], int]]:
    accepted_by_group = Counter(
        str(trace.get("validation_group_id", ""))
        for trace in existing.values()
        if trace.get("accepted_for_training") is True
    )
    tasks: list[tuple[dict[str, Any], int]] = []
    for row in train_sources:
        group = group_id(row)
        group_complete = accepted_by_group[group] >= min_accepted_variants
        for variant in range(variants):
            key = trace_key(group, variant)
            previous = existing.get(key)
            if previous is None:
                tasks.append((row, variant))
                continue
            if (
                retry_rejected
                and not group_complete
                and previous.get("accepted_for_training") is not True
                and int(previous.get("generation_attempt", 0)) < max_rejected_retry_rounds
            ):
                tasks.append((row, variant))
    return tasks


def endpoint_model_ids(url: str, timeout_sec: float) -> set[str]:
    with urlopen(Request(f"{url.rstrip('/')}/v1/models", method="GET"), timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        str(item.get("id", ""))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def call_teacher(
    url: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_sec: float,
    retry_count: int,
) -> tuple[str, float]:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        started = time.perf_counter()
        try:
            request = Request(
                f"{url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout_sec) as response:
                value = json.loads(response.read().decode("utf-8"))
            text = str(value["choices"][0]["message"]["content"]).strip()
            if not text:
                raise RationaleError("Teacher returned an empty rationale")
            return text, (time.perf_counter() - started) * 1000
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError, RationaleError) as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(1.0)
    assert last_error is not None
    raise RationaleError(f"Teacher request failed: {last_error}")


def trace_key(group: str, variant: int) -> str:
    return f"{group}::rationale-{variant}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train-only verified NLP rationale distillation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--teacher-url")
    parser.add_argument("--teacher-model-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Retry only rejected slots in groups that still have no verified rationale.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["policy"].get("feedback_source") != "train_only_internal_validation":
            raise RationaleError("Remediation policy must be train-only")
        for key in ("smoke96_item_feedback_used", "selection170_feedback_used", "formal_full_feedback_used"):
            if config["policy"].get(key) is not False:
                raise RationaleError(f"Forbidden feedback policy enabled: {key}")
        settings = config["nlp_rationale"]
        data = config["data"]
        source_path = resolve_path(data["nlp_source"])
        trace_path = resolve_path(data["nlp_trace"])
        train_path = resolve_path(data["nlp_train"])
        validation_path = resolve_path(data["nlp_internal_validation"])
        audit_path = ROOT / "reports" / "audit" / "gate_p0a4r_nlp_rationale_data.json"
        teacher_url = args.teacher_url or config["models"]["teacher_url"]
        teacher_model_id = args.teacher_model_id or config["models"]["nlp_teacher_model_id"]
        rows = load_unique_train_rows(source_path)
        validation_count = int(settings["validation_groups"])
        if len(rows) <= validation_count:
            raise RationaleError("NLP source is too small for the frozen internal validation split")
        ordered = sorted(
            rows,
            key=lambda row: sha256_text(f"{config['seed']}:nlp-dev:{group_id(row)}"),
        )
        validation_groups = {group_id(row) for row in ordered[:validation_count]}
        train_sources = [row for row in rows if group_id(row) not in validation_groups]
        source_hash_by_group = {
            group_id(row): sha256_text(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            for row in train_sources
        }

        existing: dict[str, dict[str, Any]] = {}
        min_chars = int(settings["min_rationale_chars"])
        max_chars = int(settings["max_rationale_chars"])
        if args.resume and trace_path.is_file():
            for trace in read_jsonl(trace_path):
                key = str(trace.get("trace_key", ""))
                group = str(trace.get("validation_group_id", ""))
                if (
                    key
                    and trace.get("dry_run") is False
                    and trace.get("teacher_model_id") == teacher_model_id
                    and trace.get("source_row_hash") == source_hash_by_group.get(group)
                ):
                    existing[key] = revalidate_trace(trace, min_chars, max_chars)
        variants = int(settings["variants_per_group"])
        tasks = select_generation_tasks(
            train_sources,
            variants,
            existing,
            args.retry_rejected,
            int(settings["min_accepted_variants_per_train_group"]),
            int(settings.get("max_rejected_retry_rounds", 2)),
        )
        if not args.dry_run:
            models = endpoint_model_ids(teacher_url, min(float(settings["request_timeout_sec"]), 10.0))
            if teacher_model_id not in models:
                raise RationaleError(
                    f"Teacher model {teacher_model_id!r} is not served; available={sorted(models)}"
                )

        created_ts = datetime.now(timezone.utc).isoformat()

        def generate(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
            row, variant = item
            key = trace_key(group_id(row), variant)
            previous = existing.get(key)
            generation_attempt = int(previous.get("generation_attempt", 0)) + 1 if previous else 0
            # Label-conditioned repair is allowed only for the train-only Teacher request.
            # The Student input remains label-free and is stored separately as `messages`.
            messages = rationale_messages(row, variant, 0)
            teacher_generation_messages = rationale_messages(
                row,
                variant,
                generation_attempt,
            )
            expected = str(row["answer"]).strip().upper()
            if args.dry_run:
                response = f"该选项与题干中的关键条件一致。\nFINAL: {expected}"
                latency_ms = 0.0
            else:
                response, latency_ms = call_teacher(
                    teacher_url,
                    teacher_model_id,
                    teacher_generation_messages,
                    int(settings["max_new_tokens"]),
                    float(settings["request_timeout_sec"]),
                    int(settings["retry_count"]),
                )
            accepted, verification, normalized = parse_verified_rationale(
                response,
                expected,
                int(settings["min_rationale_chars"]),
                int(settings["max_rationale_chars"]),
            )
            trace = {
                "trace_version": "p0a4r-nlp-rationale-1.0",
                "created_by": "model_compression/generate_p0a4r_nlp_rationales.py",
                "created_ts": created_ts,
                "trace_key": trace_key(group_id(row), variant),
                "dataset_key": "cmmlu",
                "sample_id": str(row["sample_id"]),
                "validation_group_id": group_id(row),
                "variant": variant,
                "source_row_hash": sha256_text(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                ),
                "teacher_model_id": teacher_model_id,
                "messages": messages,
                "messages_hash": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                "teacher_generation_messages": teacher_generation_messages,
                "teacher_generation_messages_hash": sha256_text(
                    json.dumps(
                        teacher_generation_messages,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "teacher_response": response,
                "teacher_response_hash": sha256_text(response),
                "expected_choice": expected,
                "accepted_for_training": bool(accepted and not args.dry_run),
                "verification": verification,
                "normalized_answer": normalized if accepted else "",
                "latency_ms": latency_ms,
                "dry_run": bool(args.dry_run),
                "generation_attempt": generation_attempt,
                "rationale_generation_mode": (
                    "train_label_conditioned_repair"
                    if generation_attempt >= 2
                    else "independent_retry"
                    if generation_attempt > 0
                    else "independent"
                ),
            }
            if previous is not None:
                prior_attempts = list(previous.get("prior_attempts", []))
                prior_attempts.append(
                    {
                        "generation_attempt": int(previous.get("generation_attempt", 0)),
                        "teacher_response": previous.get("teacher_response", ""),
                        "teacher_response_hash": previous.get("teacher_response_hash", ""),
                        "accepted_for_training": previous.get("accepted_for_training"),
                        "verification": previous.get("verification", ""),
                        "latency_ms": previous.get("latency_ms"),
                        "messages_hash": previous.get("messages_hash", ""),
                    }
                )
                trace["prior_attempts"] = prior_attempts
            return trace_with_hash(trace)

        if tasks:
            with ThreadPoolExecutor(max_workers=int(settings["workers"])) as executor:
                futures = {executor.submit(generate, item): item for item in tasks}
                for completed, future in enumerate(as_completed(futures), start=1):
                    trace = future.result()
                    existing[str(trace["trace_key"])] = trace
                    print(
                        f"[{completed}/{len(tasks)}] {trace['sample_id']} variant={trace['variant']} "
                        f"accepted={trace['accepted_for_training']} verify={trace['verification']}",
                        flush=True,
                    )
                    if completed % 25 == 0:
                        write_jsonl(trace_path, [existing[key] for key in sorted(existing)])

        traces = [existing[key] for key in sorted(existing)]
        write_jsonl(trace_path, traces)
        traces_by_group: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            if trace.get("accepted_for_training") is True:
                traces_by_group.setdefault(str(trace["validation_group_id"]), []).append(trace)
        min_variants = int(settings["min_accepted_variants_per_train_group"])
        accepted_groups = {
            group for group, values in traces_by_group.items() if len(values) >= min_variants
        }
        train_rows: list[dict[str, Any]] = []
        source_by_group = {group_id(row): row for row in train_sources}
        for group in sorted(source_by_group):
            source = source_by_group[group]
            for trace in sorted(
                traces_by_group.get(group, []), key=lambda value: int(value["variant"])
            ):
                train_rows.append(
                    {
                        "remediation_version": "p0a4r-1.0",
                        "created_by": "model_compression/generate_p0a4r_nlp_rationales.py",
                        "source": "teacher_verified_nlp_rationale",
                        "dataset_key": "cmmlu",
                        "sample_id": f"{source['sample_id']}/rationale-{trace['variant']}",
                        "validation_group_id": group,
                        "messages": trace["messages"],
                        "answer": trace["normalized_answer"],
                        "teacher_trace_row_hash": trace["trace_row_hash"],
                        "supervision_type": "short_rationale_and_final_choice",
                        "used_for_training": True,
                        "used_for_validation": False,
                        "used_for_final_test": False,
                    }
                )
            for replay_index in range(int(settings["label_replay_per_group"])):
                train_rows.append(
                    {
                        "remediation_version": "p0a4r-1.0",
                        "created_by": "model_compression/generate_p0a4r_nlp_rationales.py",
                        "source": "nlp_direct_choice_replay",
                        "dataset_key": "cmmlu",
                        "sample_id": f"{source['sample_id']}/label-{replay_index}",
                        "validation_group_id": group,
                        "messages": source["messages"],
                        "answer": str(source["answer"]).strip().upper(),
                        "supervision_type": "direct_choice",
                        "used_for_training": True,
                        "used_for_validation": False,
                        "used_for_final_test": False,
                    }
                )
        validation_rows = []
        for source in sorted(
            (row for row in rows if group_id(row) in validation_groups),
            key=lambda row: group_id(row),
        ):
            validation_rows.append(
                {
                    "remediation_version": "p0a4r-1.0",
                    "created_by": "model_compression/generate_p0a4r_nlp_rationales.py",
                    "source": "nlp_train_only_internal_validation",
                    "dataset_key": "cmmlu",
                    "sample_id": f"p0a4r/internal-nlp/{sha256_text(group_id(source))[:16]}",
                    "validation_group_id": group_id(source),
                    "messages": source["messages"],
                    "answer": str(source["answer"]).strip().upper(),
                    "used_for_training": False,
                    "used_for_validation": True,
                    "used_for_final_test": False,
                }
            )
        write_jsonl(train_path, train_rows)
        write_jsonl(validation_path, validation_rows)
        accepted_variant_count = sum(
            trace.get("accepted_for_training") is True for trace in traces
        )
        errors = []
        if not args.dry_run and len(accepted_groups) < int(settings["min_train_groups"]):
            errors.append("insufficient_groups_with_verified_rationales")
        if len(validation_rows) != validation_count:
            errors.append("internal_validation_count_mismatch")
        if {group_id(row) for row in train_rows} & {group_id(row) for row in validation_rows}:
            errors.append("train_validation_group_overlap")
        audit = {
            "gate": "P0-A4R-NLP-RATIONALE-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/generate_p0a4r_nlp_rationales.py",
            "created_ts": created_ts,
            "status": "dry_run_passed" if args.dry_run and not errors else "passed" if not errors else "failed",
            "policy": config["policy"],
            "teacher_url": teacher_url,
            "teacher_model_id": teacher_model_id,
            "source": display_path(source_path),
            "source_hash": sha256_file(source_path),
            "source_unique_group_count": len(rows),
            "train_source_group_count": len(train_sources),
            "internal_validation_group_count": len(validation_rows),
            "requested_rationale_variant_count": len(train_sources) * variants,
            "completed_trace_count": len(traces),
            "generation_task_count_this_run": len(tasks),
            "retry_rejected_enabled": bool(args.retry_rejected),
            "retried_trace_count": sum(
                int(trace.get("generation_attempt", 0)) > 0 for trace in traces
            ),
            "label_conditioned_repair_trace_count": sum(
                int(trace.get("generation_attempt", 0)) >= 2 for trace in traces
            ),
            "accepted_rationale_variant_count": accepted_variant_count,
            "accepted_train_group_count": len(accepted_groups),
            "rejection_counts": dict(
                Counter(
                    str(trace.get("verification", ""))
                    for trace in traces
                    if trace.get("accepted_for_training") is not True
                )
            ),
            "training_row_count": len(train_rows),
            "training_supervision_counts": dict(
                Counter(str(row["supervision_type"]) for row in train_rows)
            ),
            "train_validation_group_overlap_count": len(
                {group_id(row) for row in train_rows}
                & {group_id(row) for row in validation_rows}
            ),
            "trace": display_path(trace_path),
            "trace_hash": sha256_file(trace_path),
            "train": display_path(train_path),
            "train_hash": sha256_file(train_path),
            "internal_validation": display_path(validation_path),
            "internal_validation_hash": sha256_file(validation_path),
            "formal_test_reference_count": 0,
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        write_json(audit_path, audit)
        print(f"Wrote {display_path(train_path)} rows={len(train_rows)}")
        print(f"Wrote {display_path(validation_path)} rows={len(validation_rows)}")
        print(f"Wrote {display_path(audit_path)} status={audit['status']}")
        return 0 if not errors else 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError, RationaleError) as exc:
        print(f"P0-A4R NLP rationale generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
