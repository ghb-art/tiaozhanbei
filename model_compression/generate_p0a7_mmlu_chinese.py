#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "datasets" / "mmlu" / "data" / "auxiliary_train"
DEFAULT_REQUESTS = ROOT / "data" / "distill" / "p0a7_nlp_teacher_requests.jsonl"
DEFAULT_TRACE = ROOT / "data" / "distill" / "p0a7_nlp_teacher_trace.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "distill" / "p0a7_nlp_verified_train.jsonl"
DEFAULT_VALIDATION = ROOT / "data" / "distill" / "p0a7_nlp_verified_validation.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a7_nlp_data.json"
CHOICES = ("A", "B", "C", "D")


class NlpDataError(RuntimeError):
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


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized_identity(question: str, options: list[str]) -> str:
    return sha256_text(" ".join([question, *options]).casefold())


def teacher_messages(question: str, options: list[str]) -> list[dict[str, str]]:
    rendered = "\n".join(
        [f"Question: {question}", *[f"{key}. {value}" for key, value in zip(CHOICES, options)]]
    )
    return [
        {
            "role": "system",
            "content": (
                "You create high-quality Chinese multiple-choice distillation data. "
                "Translate faithfully, solve independently, and never invent facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{rendered}\n\n"
                "Translate the question and all four options into fluent Chinese, then "
                "give one short Chinese reason and your independently solved final choice. "
                "Return exactly one JSON object with keys question_zh, options_zh "
                "(an object with A/B/C/D), reason_zh, and final. Do not use markdown."
            ),
        },
    ]


def prepare_requests(
    source_dir: Path,
    output: Path,
    train_target: int,
    validation_target: int,
    seed: int,
    oversample_factor: float,
) -> list[dict[str, Any]]:
    paths = sorted(source_dir.glob("*.csv"))
    if not paths:
        raise NlpDataError(f"No MMLU auxiliary CSV files: {display_path(source_dir)}")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        domain_seen: set[str] = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 6:
                    continue
                question = row[0].strip()
                options = [value.strip() for value in row[1:5]]
                expected = row[5].strip().upper()
                if (
                    not question
                    or any(not option for option in options)
                    or expected not in CHOICES
                ):
                    continue
                identity = normalized_identity(question, options)
                if identity in domain_seen:
                    continue
                domain_seen.add(identity)
                by_domain[path.stem].append(
                    {
                        "request_id": f"p0a7/nlp/{path.stem}/{identity[:20]}",
                        "validation_group_id": f"mmlu-aux/{identity[:20]}",
                        "domain": path.stem,
                        "source_row": row_number,
                        "expected_label": expected,
                        "messages": teacher_messages(question, options),
                        "source_question_hash": identity,
                    }
                )
    domains = sorted(by_domain)
    if len(domains) < 8:
        raise NlpDataError(f"Expected at least 8 domains, found {len(domains)}")
    candidate_train_target = math.ceil(train_target * oversample_factor)
    candidate_validation_target = math.ceil(
        validation_target * oversample_factor
    )
    train_base, train_remainder = divmod(candidate_train_target, len(domains))
    validation_base, validation_remainder = divmod(
        candidate_validation_target, len(domains)
    )
    selected: list[dict[str, Any]] = []
    selected_identities: set[str] = set()
    # Several auxiliary files intentionally aggregate questions from smaller
    # sources. Allocate the smallest domains first, then exclude their identities
    # from larger domains. This keeps all eight domains represented without
    # putting duplicate questions in different train/validation groups.
    allocation_order = sorted(domains, key=lambda name: (len(by_domain[name]), name))
    quota_by_domain: dict[str, tuple[int, int]] = {}
    for index, domain in enumerate(domains):
        train_count = train_base + int(index < train_remainder)
        validation_count = validation_base + int(index < validation_remainder)
        quota_by_domain[domain] = (train_count, validation_count)
    for domain in allocation_order:
        train_count, validation_count = quota_by_domain[domain]
        ordered = sorted(
            [
                row
                for row in by_domain[domain]
                if str(row["source_question_hash"]) not in selected_identities
            ],
            key=lambda row: sha256_text(f"{seed}:{row['request_id']}"),
        )
        required = train_count + validation_count
        if len(ordered) < required:
            raise NlpDataError(
                f"Domain {domain} has {len(ordered)} rows; requires {required}"
            )
        validation = ordered[:validation_count]
        train = ordered[validation_count:required]
        for row in validation:
            row["split_role"] = "new_train_only_validation"
        for row in train:
            row["split_role"] = "train"
        selected_identities.update(
            str(row["source_question_hash"]) for row in validation + train
        )
        selected.extend(validation)
        selected.extend(train)
    selected.sort(key=lambda row: str(row["request_id"]))
    write_jsonl(output, selected)
    return selected


def select_balanced_verified(
    rows: Iterable[dict[str, Any]],
    target: int,
    seed: int,
    split_name: str,
    minimum_equal_quota_ratio: float,
    minimum_domains: int = 8,
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", ""))].append(row)
    domains = sorted(domain for domain in by_domain if domain)
    if len(domains) < minimum_domains:
        raise NlpDataError(
            f"{split_name} has only {len(domains)} verified domains; "
            f"requires {minimum_domains}"
        )
    equal_quota = target // len(domains)
    minimum_quota = math.floor(equal_quota * minimum_equal_quota_ratio)
    ordered_by_domain: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    cursors: dict[str, int] = {}
    for domain in domains:
        ordered = sorted(
            by_domain[domain],
            key=lambda row: sha256_text(
                f"{seed}:{split_name}:{row.get('sample_id', '')}"
            ),
        )
        if len(ordered) < minimum_quota:
            raise NlpDataError(
                f"{split_name}/{domain} verified={len(ordered)} "
                f"requires_minimum={minimum_quota}"
            )
        ordered_by_domain[domain] = ordered
        selected.extend(ordered[:minimum_quota])
        cursors[domain] = minimum_quota
    while len(selected) < target:
        progressed = False
        for domain in domains:
            cursor = cursors[domain]
            ordered = ordered_by_domain[domain]
            if cursor >= len(ordered):
                continue
            selected.append(ordered[cursor])
            cursors[domain] = cursor + 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            raise NlpDataError(
                f"{split_name} verified_total={len(selected)} requires={target}"
            )
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def request_teacher(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout_sec: float,
    temperature: float = 0.0,
    seed: int = 20260727,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": 768,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        value = json.loads(response.read().decode("utf-8"))
    choices = value.get("choices", [])
    if not choices:
        raise NlpDataError("Teacher returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise NlpDataError("Teacher returned empty content")
    return content.strip()


def discover_fallback_model_id(
    endpoint: str,
    primary_model_id: str,
    configured: str,
) -> str:
    if configured and configured != "auto":
        return configured
    request = Request(
        endpoint.rstrip("/") + "/v1/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ids = [
        str(item.get("id", ""))
        for item in payload.get("data", [])
        if isinstance(item, dict) and str(item.get("id", ""))
    ]
    candidates = [model_id for model_id in ids if model_id != primary_model_id]
    return candidates[0] if candidates else primary_model_id


def parse_teacher_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise NlpDataError("teacher_response_not_json")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise NlpDataError("teacher_response_not_object")
    return value


def cjk_count(value: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", value))


def validate_translation(
    request_row: dict[str, Any],
    response: str,
) -> tuple[dict[str, Any] | None, str]:
    try:
        value = parse_teacher_json(response)
    except (json.JSONDecodeError, NlpDataError) as exc:
        return None, str(exc)
    question = str(value.get("question_zh", "")).strip()
    reason = str(value.get("reason_zh", "")).strip()
    final = str(value.get("final", "")).strip().upper()
    options_value = value.get("options_zh")
    if isinstance(options_value, dict):
        options = {key: str(options_value.get(key, "")).strip() for key in CHOICES}
    elif isinstance(options_value, list) and len(options_value) == 4:
        options = {key: str(item).strip() for key, item in zip(CHOICES, options_value)}
    else:
        return None, "invalid_options"
    if not question or any(not value for value in options.values()) or not reason:
        return None, "missing_translation_field"
    if final != str(request_row["expected_label"]):
        return None, "label_verification_failed"
    if cjk_count(question + reason + "".join(options.values())) < 20:
        return None, "insufficient_chinese_content"
    if len(reason) < 8 or len(reason) > 300:
        return None, "reason_length_out_of_range"
    rendered = "\n".join(
        [
            "以下是单项选择题。",
            "",
            f"题目: {question}",
            *[f"{key}. {options[key]}" for key in CHOICES],
            "",
            "请给出一条简短理由，最后一行严格写成 FINAL: X。",
        ]
    )
    split_role = str(request_row["split_role"])
    row = {
        "sample_id": str(request_row["request_id"]),
        "validation_group_id": str(request_row["validation_group_id"]),
        "dataset_key": "cmmlu",
        "source": "p0a7_mmlu_aux_teacher_verified_chinese",
        "origin": "mmlu_auxiliary_train",
        "domain": str(request_row["domain"]),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是严谨的中文边缘推理助手。依据题目内容作答，不引用题库答案。"
                ),
            },
            {"role": "user", "content": rendered},
        ],
        "answer": f"理由：{reason}\nFINAL: {final}",
        "supervision_type": "chinese_short_rationale_and_verified_choice",
        "teacher_verification": {
            "method": "independent_teacher_choice_matches_train_label",
            "source_question_hash": request_row["source_question_hash"],
        },
        "used_for_training": split_role == "train",
        "used_for_validation": split_role == "new_train_only_validation",
        "used_for_final_test": False,
    }
    return row, ""


def load_latest_trace(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        values[str(row.get("request_id", ""))] = row
    return values


def generate(args: argparse.Namespace) -> None:
    requests_path = resolve_path(args.requests)
    requests = read_jsonl(requests_path)
    if not requests:
        raise NlpDataError(f"No teacher requests: {display_path(requests_path)}")
    trace_path = resolve_path(args.trace)
    previous = load_latest_trace(trace_path)
    fallback_model_id = discover_fallback_model_id(
        args.endpoint,
        args.model_id,
        args.fallback_model_id,
    )
    accepted: dict[str, dict[str, Any]] = {
        key: value["verified_row"]
        for key, value in previous.items()
        if isinstance(value.get("verified_row"), dict)
    }
    for row in accepted.values():
        verification = row.get("teacher_verification")
        if isinstance(verification, dict):
            verification.setdefault("model_id", args.model_id)
            verification.setdefault("attempt", 1)
    completed_ids = set(previous)
    pending = [
        row
        for row in requests
        if (
            str(row["request_id"]) not in accepted
            if args.retry_rejected
            else str(row["request_id"]) not in completed_ids
        )
    ]
    pending.sort(
        key=lambda row: sha256_text(f"{args.seed}:pending:{row['request_id']}")
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    def select_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train_candidates = [
            row for row in accepted.values() if row["used_for_training"] is True
        ]
        validation_candidates = [
            row for row in accepted.values() if row["used_for_validation"] is True
        ]
        train_selected = select_balanced_verified(
            train_candidates,
            args.train_target,
            args.seed,
            "train",
            args.minimum_domain_equal_quota_ratio,
            args.minimum_domains,
        )
        validation_selected = select_balanced_verified(
            validation_candidates,
            args.validation_target,
            args.seed,
            "new_train_only_validation",
            args.minimum_domain_equal_quota_ratio,
            args.minimum_domains,
        )
        return train_selected, validation_selected

    def process(row: dict[str, Any]) -> dict[str, Any]:
        last_response = ""
        last_reason = ""
        for attempt in range(1, args.retries + 1):
            try:
                attempt_model_id = (
                    fallback_model_id
                    if args.retries > 1 and attempt == args.retries
                    else args.model_id
                )
                attempt_messages = list(row["messages"])
                if attempt > 1:
                    attempt_messages = [
                        *attempt_messages,
                        {
                            "role": "user",
                            "content": (
                                "Independently reconsider the problem, check every "
                                "option, and return the same required JSON schema. "
                                "Do not refer to any previous answer."
                            ),
                        },
                    ]
                last_response = request_teacher(
                    args.endpoint,
                    attempt_model_id,
                    attempt_messages,
                    args.timeout_sec,
                    temperature=(0.0, 0.2, 0.4)[min(attempt - 1, 2)],
                    seed=args.seed + attempt,
                )
                verified, last_reason = validate_translation(row, last_response)
                if verified is not None:
                    verified["teacher_verification"]["model_id"] = attempt_model_id
                    verified["teacher_verification"]["attempt"] = attempt
                    return {
                        "request_id": row["request_id"],
                        "attempt": attempt,
                        "status": "accepted",
                        "reason": "",
                        "model_id": attempt_model_id,
                        "response": last_response,
                        "verified_row": verified,
                    }
            except (HTTPError, URLError, TimeoutError, NlpDataError) as exc:
                last_reason = f"{type(exc).__name__}:{exc}"
            time.sleep(min(0.5 * attempt, 2.0))
        return {
            "request_id": row["request_id"],
            "attempt": args.retries,
            "status": "rejected",
            "reason": last_reason,
            "model_id": (
                fallback_model_id if args.retries > 1 else args.model_id
            ),
            "response": last_response,
            "verified_row": None,
        }

    selected_outputs: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None
    try:
        selected_outputs = select_outputs()
    except NlpDataError:
        pass
    processed = 0
    batch_size = max(args.workers, args.workers * 4)
    with trace_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(0, len(pending), batch_size):
                if selected_outputs is not None:
                    break
                batch = pending[offset : offset + batch_size]
                futures = {executor.submit(process, row): row for row in batch}
                for future in as_completed(futures):
                    result = future.result()
                    stream.write(stable_json(result) + "\n")
                    stream.flush()
                    processed += 1
                    if isinstance(result.get("verified_row"), dict):
                        accepted[str(result["request_id"])] = result["verified_row"]
                    if processed % 50 == 0:
                        print(
                            f"[NLP teacher] {processed}/{len(pending)} "
                            f"accepted_total={len(accepted)}/{len(requests)}",
                            flush=True,
                        )
                try:
                    selected_outputs = select_outputs()
                except NlpDataError:
                    pass
    if selected_outputs is None:
        selected_outputs = select_outputs()
    train, validation = selected_outputs
    train_candidates = [
        row for row in accepted.values() if row["used_for_training"] is True
    ]
    validation_candidates = [
        row for row in accepted.values() if row["used_for_validation"] is True
    ]
    train_path = resolve_path(args.train_output)
    validation_path = resolve_path(args.validation_output)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    expected_train = args.train_target
    expected_validation = args.validation_target
    errors: list[str] = []
    if len(train) != expected_train:
        errors.append("incomplete_verified_train")
    if len(validation) != expected_validation:
        errors.append("incomplete_verified_validation")
    train_groups = {str(row["validation_group_id"]) for row in train}
    validation_groups = {str(row["validation_group_id"]) for row in validation}
    if train_groups & validation_groups:
        errors.append("train_validation_group_overlap")
    audit = {
        "gate": "P0-A7-NLP-CHINESE-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/generate_p0a7_mmlu_chinese.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "policy": {
            "source_split": "MMLU auxiliary_train only",
            "teacher_prompt_contains_source_label": False,
            "acceptance": "teacher final must match train label",
            "required_language": "Chinese",
            "minimum_domain_equal_quota_ratio": (
                args.minimum_domain_equal_quota_ratio
            ),
            "formal_test_labels_used": False,
        },
        "teacher_endpoint": args.endpoint,
        "teacher_model_id": args.model_id,
        "teacher_fallback_model_id": fallback_model_id,
        "requests": display_path(requests_path),
        "requests_hash": sha256_file(requests_path),
        "request_count": len(requests),
        "accepted_count": len(accepted),
        "accepted_candidate_counts": {
            "train": len(train_candidates),
            "new_train_only_validation": len(validation_candidates),
        },
        "rejection_counts": dict(
            Counter(
                str(row.get("reason", ""))
                for row in load_latest_trace(trace_path).values()
                if not isinstance(row.get("verified_row"), dict)
            )
        ),
        "train": display_path(train_path),
        "train_hash": sha256_file(train_path),
        "train_unique_groups": len(train_groups),
        "train_domain_counts": dict(Counter(str(row["domain"]) for row in train)),
        "validation": display_path(validation_path),
        "validation_hash": sha256_file(validation_path),
        "validation_unique_groups": len(validation_groups),
        "validation_domain_counts": dict(
            Counter(str(row["domain"]) for row in validation)
        ),
        "train_validation_overlap": len(train_groups & validation_groups),
        "errors": errors,
    }
    audit["report_hash"] = sha256_text(stable_json(audit))
    write_json(resolve_path(args.audit), audit)
    print(
        f"NLP train={len(train)}/{expected_train} "
        f"validation={len(validation)}/{expected_validation} "
        f"status={audit['status']}",
        flush=True,
    )
    if errors:
        raise NlpDataError(str(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build label-verified Chinese train-only NLP distillation data."
    )
    parser.add_argument("command", choices=("prepare", "generate", "all", "status"))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN))
    parser.add_argument("--validation-output", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--train-target", type=int, default=3000)
    parser.add_argument("--validation-target", type=int, default=256)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--fallback-model-id",
        default="auto",
        help="Third-pass model id; auto selects the non-LoRA base from /v1/models.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Explicitly retry requests that already exhausted all Teacher attempts.",
    )
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.35,
        help="Candidate request multiplier before label verification.",
    )
    parser.add_argument(
        "--minimum-domain-equal-quota-ratio",
        type=float,
        default=0.8,
        help="Each of eight domains must retain this fraction of an equal quota.",
    )
    parser.add_argument(
        "--minimum-domains",
        type=int,
        default=8,
        help="Minimum distinct source domains required after Teacher verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if (
            args.train_target <= 0
            or args.validation_target <= 0
            or args.workers <= 0
            or args.retries <= 0
            or args.timeout_sec <= 0
            or args.oversample_factor < 1.0
            or not 0 < args.minimum_domain_equal_quota_ratio <= 1.0
            or args.minimum_domains <= 0
        ):
            raise NlpDataError("Invalid target/worker/retry/timeout argument")
        requests_path = resolve_path(args.requests)
        if args.command == "status":
            for value in (
                args.requests,
                args.trace,
                args.train_output,
                args.validation_output,
                args.audit,
            ):
                path = resolve_path(value)
                print(f"{display_path(path)} exists={path.is_file()}")
            return 0
        if args.command in {"prepare", "all"}:
            rows = prepare_requests(
                resolve_path(args.source_dir),
                requests_path,
                args.train_target,
                args.validation_target,
                args.seed,
                args.oversample_factor,
            )
            print(f"Wrote {display_path(requests_path)} rows={len(rows)}")
        if args.command in {"generate", "all"}:
            generate(args)
        return 0
    except (
        NlpDataError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"P0-A7 NLP data failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
