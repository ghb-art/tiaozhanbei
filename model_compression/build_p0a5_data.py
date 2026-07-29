#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a5_capability.json"
ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "copy",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
FORBIDDEN_TEXT = (
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "os.system",
    "eval(",
    "exec(",
    "__import__",
    "open(",
    "pathlib",
    "shutil",
    "pickle",
    "ctypes",
    "multiprocessing",
    "threading",
)
NLP_QUOTA_KEYS = (
    "exam",
    "science_encyclopedia",
    "humanities_social_science",
    "law_economics_management",
    "language_reasoning",
)
NLP_PRIMARY_PRIORITY = (
    "exam",
    "law_economics_management",
    "science_encyclopedia",
    "humanities_social_science",
    "language_reasoning",
)
CODE_EXECUTION_CACHE = ROOT / "data/capability_v2/code_execution_cache.jsonl"
CODE_VALIDATOR_VERSION = "p0a5-python-isolated-v1"


class DataBuildError(RuntimeError):
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


def stable_order(seed: int, identity: str) -> str:
    return sha256_text(f"{seed}:{identity}")


def normalize_text(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


def word_ngrams(value: str, width: int = 4) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|[\u4e00-\u9fff]", value.casefold())
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def too_similar(value: str, references: list[set[tuple[str, ...]]], threshold: float = 0.45) -> bool:
    grams = word_ngrams(value)
    if not grams:
        return False
    for reference in references:
        shared = len(grams & reference)
        if not shared:
            continue
        union = len(grams | reference)
        if union and shared / union >= threshold:
            return True
    return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DataBuildError(f"Non-object row {display_path(path)}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def download_sources(config: dict[str, Any]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DataBuildError("huggingface_hub is required to download P0-A5 sources") from exc

    code = config["datasets"]["code"]
    for shard in code["source_shards"]:
        filename = f"data/train-{int(shard):05d}-of-00050.parquet"
        path = hf_hub_download(
            repo_id=code["source"],
            repo_type="dataset",
            revision=code["source_revision"],
            filename=filename,
            local_dir=ROOT / "data/datasets/opencodeinstruct",
        )
        print(f"Downloaded {display_path(Path(path))}", flush=True)
    nlp = config["datasets"]["nlp"]
    path = hf_hub_download(
        repo_id=nlp["source"],
        repo_type="dataset",
        revision=nlp["source_revision"],
        filename="COIG-CQIA-full.jsonl",
        local_dir=ROOT / "data/datasets/coig_cqia",
    )
    print(f"Downloaded {display_path(Path(path))}", flush=True)


def training_row(
    sample_id: str,
    dataset_key: str,
    source: str,
    prompt: str,
    answer: str,
    split_role: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "sample_id": sample_id,
        "dataset_key": dataset_key,
        "source": source,
        "split_role": split_role,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a compact edge reasoning model. Give a concise, verifiable "
                    "answer and follow the requested output format."
                ),
            },
            {"role": "user", "content": prompt.strip()},
        ],
        "answer": answer.strip(),
    }
    if metadata:
        row["metadata"] = metadata
    return row


def build_math(
    config: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = resolve_path(config["datasets"]["math"]["train_source"])
    rows = read_jsonl(source)
    expected_total = sum(
        int(config["datasets"]["math"][key])
        for key in ("train_rows", "internal_validation_rows", "gate_rows")
    )
    if len(rows) != expected_total:
        raise DataBuildError(f"GSM8K train count changed: {len(rows)} != {expected_total}")
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: stable_order(seed, f"gsm8k:{item[0]}"))
    gate_count = int(config["datasets"]["math"]["gate_rows"])
    validation_count = int(config["datasets"]["math"]["internal_validation_rows"])
    gate_source = indexed[:gate_count]
    validation_source = indexed[gate_count : gate_count + validation_count]
    train_source = indexed[gate_count + validation_count :]

    def make(item: tuple[int, dict[str, Any]], role: str) -> dict[str, Any]:
        index, value = item
        return training_row(
            f"gsm8k/train/{index:05d}",
            "gsm8k",
            "GSM8K-train",
            str(value["question"]),
            str(value["answer"]),
            role,
            {"reference_answer": str(value["answer"]).rsplit("####", 1)[-1].strip()},
        )

    train = [make(item, "train") for item in train_source]
    validation = [make(item, "internal_validation") for item in validation_source]
    gate = [
        {
            "sample_id": f"gsm8k/train/{index:05d}",
            "domain": "math",
            "dataset_key": "gsm8k",
            "prompt": str(value["question"]),
            "reference": str(value["answer"]).rsplit("####", 1)[-1].strip(),
            "validator": "numeric_exact",
        }
        for index, value in gate_source
    ]
    return train, validation, gate, {"source_rows": len(rows)}


def extract_code(value: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else value).strip()


def parse_tests(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("unit_tests is not a non-empty list")
    return [str(item) for item in parsed]


def safe_python(source: str) -> bool:
    lowered = source.casefold()
    if any(marker in lowered for marker in FORBIDDEN_TEXT):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".", 1)[0] not in ALLOWED_IMPORTS:
                return False
    return True


def sandbox_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    os.setsid()


def execute_code_case(item: tuple[dict[str, Any], float]) -> tuple[dict[str, Any], str]:
    row, timeout = item
    code = extract_code(str(row["output"]))
    try:
        tests = parse_tests(str(row["unit_tests"]))
    except (ValueError, json.JSONDecodeError):
        return row, "invalid_tests"
    source = code + "\n\n" + "\n".join(tests) + "\n"
    if not safe_python(source):
        return row, "unsafe_or_invalid"
    with tempfile.TemporaryDirectory(prefix="p0a5-code-") as temp_dir:
        program = Path(temp_dir) / "main.py"
        program.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(program)],
                cwd=temp_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=sandbox_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return row, "timeout"
    return row, "passed" if completed.returncode == 0 else "execution_failed"


def load_humaneval_prompts() -> list[str]:
    path = ROOT / "data/datasets/humaneval/data/HumanEval.jsonl.gz"
    prompts: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prompts.append(str(json.loads(line)["prompt"]))
    if len(prompts) != 164:
        raise DataBuildError(f"HumanEval count changed: {len(prompts)}")
    return prompts


def code_prefilter(
    row: dict[str, Any],
    config: dict[str, Any],
    humaneval_grams: list[set[tuple[str, ...]]],
) -> str:
    prompt = str(row.get("input", ""))
    answer = str(row.get("output", ""))
    if float(row.get("average_test_score") or 0.0) < float(
        config["datasets"]["code"]["minimum_average_test_score"]
    ):
        return "reported_test_score"
    try:
        statuses = json.loads(str(row.get("tests_execution_status", "[]")))
        tests = parse_tests(str(row.get("unit_tests", "[]")))
    except (ValueError, json.JSONDecodeError):
        return "invalid_reported_tests"
    if len(statuses) != len(tests) or any(str(value).casefold() != "pass" for value in statuses):
        return "reported_test_failure"
    if len(prompt) > int(config["datasets"]["code"]["max_prompt_chars"]):
        return "prompt_too_long"
    if len(answer) > int(config["datasets"]["code"]["max_answer_chars"]):
        return "answer_too_long"
    if not re.search(r"\b(def|function|method|implement|return)\b", prompt, flags=re.IGNORECASE):
        return "not_function_style"
    if not re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", answer):
        return "missing_python_function"
    if any(marker in (prompt + "\n" + answer).casefold() for marker in FORBIDDEN_TEXT):
        return "unsafe_text"
    if too_similar(prompt, humaneval_grams):
        return "humaneval_near_duplicate"
    return ""


def code_fingerprint(row: dict[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            {
                "id": row.get("id"),
                "input": row.get("input"),
                "output": row.get("output"),
                "unit_tests": row.get("unit_tests"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_code_execution_cache(path: Path) -> dict[str, str]:
    allowed = {
        "passed",
        "invalid_tests",
        "unsafe_or_invalid",
        "timeout",
        "execution_failed",
    }
    cache: dict[str, str] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fingerprint = str(row.get("fingerprint", ""))
            status = str(row.get("status", ""))
            if (
                len(fingerprint) == 64
                and status in allowed
                and row.get("validator_version") == CODE_VALIDATOR_VERSION
            ):
                cache[fingerprint] = status
    return cache


def build_code(
    config: dict[str, Any], seed: int, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DataBuildError("pyarrow is required for OpenCodeInstruct") from exc
    code_config = config["datasets"]["code"]
    humaneval_grams = [word_ngrams(prompt) for prompt in load_humaneval_prompts()]
    reject_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for shard in code_config["source_shards"]:
        path = ROOT / (
            f"data/datasets/opencodeinstruct/data/train-{int(shard):05d}-of-00050.parquet"
        )
        if not path.is_file():
            raise DataBuildError(f"Missing OpenCodeInstruct shard: {display_path(path)}")
        for row in parquet.read_table(path).to_pylist():
            reason = code_prefilter(row, config, humaneval_grams)
            if reason:
                reject_counts[reason] += 1
                continue
            candidates.append(row)
    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        identity = sha256_text(normalize_text(str(row["input"])))
        unique.setdefault(identity, row)
    ordered = sorted(
        unique.values(),
        key=lambda row: stable_order(seed, f"opencode:{row.get('id', '')}"),
    )
    required = sum(
        int(code_config[key]) for key in ("train_rows", "internal_validation_rows", "gate_rows")
    )
    execution_pool = ordered[: max(required + 5000, required)]
    timeout = float(code_config["execution_timeout_sec"])
    accepted: list[dict[str, Any]] = []
    execution_cache = load_code_execution_cache(CODE_EXECUTION_CACHE)
    cache_hits = 0
    CODE_EXECUTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with (
        CODE_EXECUTION_CACHE.open("a", encoding="utf-8") as cache_handle,
        ThreadPoolExecutor(max_workers=max(1, workers)) as executor,
    ):
        batch_size = 512
        for start in range(0, len(execution_pool), batch_size):
            batch = execution_pool[start : start + batch_size]
            cached_before = {
                code_fingerprint(row)
                for row in batch
                if code_fingerprint(row) in execution_cache
            }
            pending = [
                row for row in batch if code_fingerprint(row) not in cached_before
            ]
            new_results = list(
                executor.map(
                    execute_code_case,
                    ((row, timeout) for row in pending),
                )
            )
            for row, status in new_results:
                fingerprint = code_fingerprint(row)
                execution_cache[fingerprint] = status
                cache_handle.write(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "status": status,
                            "validator_version": CODE_VALIDATOR_VERSION,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            cache_handle.flush()
            for row in batch:
                fingerprint = code_fingerprint(row)
                if fingerprint in cached_before:
                    cache_hits += 1
                result = execution_cache[fingerprint]
                if result == "passed":
                    accepted.append(row)
                else:
                    reject_counts[result] += 1
            print(
                f"Code execution verified={len(accepted)}/{required} "
                f"checked={min(start + len(batch), len(execution_pool))}",
                flush=True,
            )
            if len(accepted) >= required:
                break
    if len(accepted) < required:
        raise DataBuildError(f"Only {len(accepted)} independently verified Code rows; need {required}")
    accepted = accepted[:required]
    gate_count = int(code_config["gate_rows"])
    validation_count = int(code_config["internal_validation_rows"])
    gate_source = accepted[:gate_count]
    validation_source = accepted[gate_count : gate_count + validation_count]
    train_source = accepted[gate_count + validation_count :]

    def make(value: dict[str, Any], role: str) -> dict[str, Any]:
        return training_row(
            f"opencodeinstruct/train/{value['id']}",
            "opencodeinstruct",
            "nvidia/OpenCodeInstruct",
            str(value["input"]),
            str(value["output"]),
            role,
            {
                "unit_tests": parse_tests(str(value["unit_tests"])),
                "average_test_score": float(value["average_test_score"]),
                "independent_execution": "passed",
            },
        )

    train = [make(row, "train") for row in train_source]
    validation = [make(row, "internal_validation") for row in validation_source]
    gate = [
        {
            "sample_id": f"opencodeinstruct/train/{row['id']}",
            "domain": "code",
            "dataset_key": "opencodeinstruct",
            "prompt": str(row["input"]),
            "reference": "",
            "unit_tests": parse_tests(str(row["unit_tests"])),
            "validator": "python_unit_tests",
        }
        for row in gate_source
    ]
    return train, validation, gate, {
        "prefilter_candidates": len(candidates),
        "unique_candidates": len(unique),
        "independently_verified": len(accepted),
        "execution_cache_hits": cache_hits,
        "execution_cache": display_path(CODE_EXECUTION_CACHE),
        "rejections": dict(sorted(reject_counts.items())),
        "source_shards": list(code_config["source_shards"]),
    }


def nlp_major(row: dict[str, Any]) -> str:
    task = row.get("task_type") or {}
    values = task.get("major") if isinstance(task, dict) else []
    return " ".join(str(value) for value in (values or []))


def nlp_domains(row: dict[str, Any]) -> str:
    return " ".join(str(value) for value in (row.get("domain") or []))


def nlp_eligible_categories(row: dict[str, Any]) -> set[str]:
    domain = nlp_domains(row).casefold()
    major = nlp_major(row).casefold()
    combined = domain + " " + major
    categories: set[str] = set()
    if "试题" in major:
        categories.add("exam")
    if any(
        marker in combined
        for marker in (
            "百科",
            "理学",
            "工学",
            "生物",
            "医疗",
            "医学",
            "药物",
            "地理",
            "农学",
            "农业",
            "环境",
            "物理",
            "化学",
            "健康",
            "电子",
            "力学",
            "知识问答",
            "名词解释",
        )
    ):
        categories.add("science_encyclopedia")
    if any(
        marker in combined
        for marker in (
            "历史",
            "文化",
            "语文",
            "文学",
            "政治",
            "哲学",
            "社会",
            "艺术",
            "新闻",
            "语言学",
            "宗教",
            "民族",
            "人类价值观",
            "法学",
        )
    ):
        categories.add("humanities_social_science")
    if any(
        marker in combined
        for marker in ("法律", "法理", "law", "经济", "金融", "管理", "会计", "营销")
    ):
        categories.add("law_economics_management")
    if any(
        marker in combined
        for marker in (
            "逻辑推理",
            "自然语言推理",
            "自然语言推断",
            "阅读理解",
            "分类",
            "语义分析",
            "信息抽取",
            "纠错",
            "意图检测",
            "常识",
            "闭卷问答",
        )
    ):
        categories.add("language_reasoning")
    if not categories and "通用" in domain and any(
        marker in major for marker in ("问答", "知识问答", "文本生成")
    ):
        categories.add("language_reasoning")
    return categories


def load_cmmlu_prompts(split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    base = ROOT / f"data/datasets/cmmlu/data/{split}"
    for path in sorted(base.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                rows.append(
                    {
                        "sample_id": f"cmmlu/{split}/{path.stem}/{index:05d}",
                        "subject": path.stem,
                        "question": str(row["Question"]),
                        "A": str(row["A"]),
                        "B": str(row["B"]),
                        "C": str(row["C"]),
                        "D": str(row["D"]),
                        "answer": str(row["Answer"]).strip().upper(),
                    }
                )
    return rows


def build_cmmlu_prompt(row: dict[str, str]) -> str:
    return (
        f"问题：{row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "请给出简短分析，并在最后一行严格输出“最终答案：X”。"
    )


def select_cmmlu_gate(seed: int, count: int) -> list[dict[str, Any]]:
    rows = load_cmmlu_prompts("dev")
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    selected: list[dict[str, str]] = []
    for subject in sorted(by_subject):
        ordered = sorted(
            by_subject[subject],
            key=lambda row: stable_order(seed, row["sample_id"]),
        )
        selected.append(ordered[0])
    remaining = [row for row in rows if row not in selected]
    remaining.sort(key=lambda row: stable_order(seed + 1, row["sample_id"]))
    selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise DataBuildError(f"Cannot build {count}-row CMMLU dev gate")
    return [
        {
            "sample_id": row["sample_id"],
            "domain": "nlp",
            "dataset_key": "cmmlu",
            "subject": row["subject"],
            "prompt": build_cmmlu_prompt(row),
            "reference": row["answer"],
            "validator": "choice_exact",
        }
        for row in sorted(selected, key=lambda value: value["sample_id"])
    ]


def build_nlp(
    config: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = ROOT / "data/datasets/coig_cqia/COIG-CQIA-full.jsonl"
    if not path.is_file():
        raise DataBuildError(f"Missing COIG-CQIA: {display_path(path)}")
    cmmlu_rows = load_cmmlu_prompts("dev") + load_cmmlu_prompts("test")
    cmmlu_normalized = {
        normalize_text(row["question"])
        for row in cmmlu_rows
        if normalize_text(row["question"])
    }
    rows = read_jsonl(path)
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in NLP_QUOTA_KEYS}
    rejected: Counter[str] = Counter()
    unique: set[str] = set()
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        extra_input = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        combined = f"{instruction}\n{extra_input}".strip()
        domain = nlp_domains(row).casefold()
        major = nlp_major(row).casefold()
        if row.get("human_verified") is not True:
            rejected["not_human_verified"] += 1
            continue
        if (
            not combined
            or not answer
            or len(combined) > 3500
            or len(answer) > 2500
            or len(combined) + len(answer) > 5000
        ):
            rejected["length_or_empty"] += 1
            continue
        if any(marker in domain + " " + major for marker in ("代码", "javascript", "vue.js")):
            rejected["code"] += 1
            continue
        if "数学" in domain:
            rejected["math"] += 1
            continue
        identity = normalize_text(combined)
        if identity in cmmlu_normalized:
            rejected["cmmlu_exact_overlap"] += 1
            continue
        if identity in unique:
            rejected["duplicate"] += 1
            continue
        categories = nlp_eligible_categories(row)
        if not categories:
            rejected["outside_target_domains"] += 1
            continue
        unique.add(identity)
        enriched = dict(row)
        enriched["_combined_prompt"] = combined
        enriched["_identity"] = identity
        primary_category = next(
            category for category in NLP_PRIMARY_PRIORITY if category in categories
        )
        enriched["_eligible_categories"] = sorted(categories)
        enriched["_primary_category"] = primary_category
        pools[primary_category].append(enriched)

    quotas = {
        key: int(value)
        for key, value in config["datasets"]["nlp"]["category_train_quotas"].items()
    }
    if set(quotas) != set(NLP_QUOTA_KEYS):
        raise DataBuildError(f"COIG-CQIA category keys changed: {sorted(quotas)}")
    if sum(quotas.values()) != int(config["datasets"]["nlp"]["train_rows"]):
        raise DataBuildError(
            "COIG-CQIA train_rows does not match category_train_quotas"
        )
    validation_total = int(config["datasets"]["nlp"]["internal_validation_rows"])
    validation_quotas = {
        key: validation_total * quotas[key] // sum(quotas.values()) for key in NLP_QUOTA_KEYS
    }
    while sum(validation_quotas.values()) < validation_total:
        for key in NLP_QUOTA_KEYS:
            validation_quotas[key] += 1
            if sum(validation_quotas.values()) == validation_total:
                break

    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    category_order = sorted(
        NLP_QUOTA_KEYS,
        key=lambda key: (len(pools[key]) / (quotas[key] + validation_quotas[key]), key),
    )
    for category in category_order:
        ordered = sorted(
            pools[category],
            key=lambda row: stable_order(seed, f"{category}:{row['_identity']}"),
        )
        needed = quotas[category] + validation_quotas[category]
        selected = [row for row in ordered if row["_identity"] not in used][:needed]
        if len(selected) != needed:
            raise DataBuildError(
                f"COIG-CQIA category {category} has {len(selected)} unused rows; need {needed}"
            )
        selected_by_category[category] = selected
        used.update(row["_identity"] for row in selected)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for category in NLP_QUOTA_KEYS:
        selected = selected_by_category[category]
        validation_count = validation_quotas[category]
        validation_source = selected[:validation_count]
        train_source = selected[validation_count:]
        for role, source_rows, destination in (
            ("train", train_source, train),
            ("internal_validation", validation_source, validation),
        ):
            for row in source_rows:
                destination.append(
                    training_row(
                        f"coig_cqia/{role}/{sha256_text(row['_identity'])[:20]}",
                        "cmmlu",
                        "m-a-p/COIG-CQIA",
                        row["_combined_prompt"],
                        str(row["output"]),
                        role,
                        {
                            "nlp_category": category,
                            "domain": row.get("domain", []),
                            "task_type": row.get("task_type", {}),
                            "answer_from": row.get("answer_from", ""),
                            "human_verified": True,
                        },
                    )
                )
    train.sort(key=lambda row: row["sample_id"])
    validation.sort(key=lambda row: row["sample_id"])
    gate = select_cmmlu_gate(seed, int(config["datasets"]["nlp"]["gate_rows"]))
    return train, validation, gate, {
        "source_rows": len(rows),
        "pool_counts": {key: len(value) for key, value in pools.items()},
        "train_category_counts": dict(
            Counter(row["metadata"]["nlp_category"] for row in train)
        ),
        "validation_category_counts": dict(
            Counter(row["metadata"]["nlp_category"] for row in validation)
        ),
        "rejections": dict(sorted(rejected.items())),
    }


def add_task_weights(rows: list[dict[str, Any]], mass: dict[str, float]) -> None:
    counts = Counter(str(row["dataset_key"]) for row in rows)
    total = len(rows)
    if set(counts) != set(mass):
        raise DataBuildError(f"Task weight/count mismatch: counts={counts}, mass={mass}")
    if abs(sum(mass.values()) - 1.0) > 1e-9:
        raise DataBuildError("Student task loss mass must sum to 1")
    for row in rows:
        task = str(row["dataset_key"])
        row["training_weight"] = mass[task] * total / counts[task]
        row["preserve_math"] = task == "gsm8k"


def rows_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text(
        "\n".join(
            sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
            for row in rows
        )
        + "\n"
    )


def build_all(config_path: Path, workers: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    math_train, math_validation, math_gate, math_stats = build_math(config, seed)
    print("Math data prepared", flush=True)
    code_train, code_validation, code_gate, code_stats = build_code(config, seed, workers)
    print("Code data prepared", flush=True)
    nlp_train, nlp_validation, nlp_gate, nlp_stats = build_nlp(config, seed)
    print("NLP data prepared", flush=True)

    train = math_train + code_train + nlp_train
    validation = math_validation + code_validation + nlp_validation
    gate = math_gate + code_gate + nlp_gate
    add_task_weights(
        train,
        {
            key: float(value)
            for key, value in config["student_training"]["task_loss_mass"].items()
        },
    )
    train.sort(key=lambda row: stable_order(seed, row["sample_id"]))
    validation.sort(key=lambda row: stable_order(seed + 1, row["sample_id"]))
    gate.sort(key=lambda row: (row["domain"], row["sample_id"]))

    train_ids = {row["sample_id"] for row in train}
    validation_ids = {row["sample_id"] for row in validation}
    gate_ids = {row["sample_id"] for row in gate}
    overlaps = {
        "train_validation": len(train_ids & validation_ids),
        "train_gate": len(train_ids & gate_ids),
        "validation_gate": len(validation_ids & gate_ids),
    }
    if any(overlaps.values()):
        raise DataBuildError(f"Split overlap detected: {overlaps}")

    artifacts = config["artifacts"]
    train_path = resolve_path(artifacts["source_train"])
    validation_path = resolve_path(artifacts["source_validation"])
    gate_path = resolve_path(artifacts["gate_manifest"])
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    write_jsonl(gate_path, gate)
    counts = {
        "train": dict(sorted(Counter(row["dataset_key"] for row in train).items())),
        "validation": dict(
            sorted(Counter(row["dataset_key"] for row in validation).items())
        ),
        "gate": dict(sorted(Counter(row["domain"] for row in gate).items())),
    }
    manifest = {
        "protocol": "P0-A5",
        "created_by": "model_compression/build_p0a5_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "counts": counts,
        "overlaps": overlaps,
        "source_train": display_path(train_path),
        "source_train_hash": sha256_file(train_path),
        "source_validation": display_path(validation_path),
        "source_validation_hash": sha256_file(validation_path),
        "gate_manifest": display_path(gate_path),
        "gate_manifest_hash": sha256_file(gate_path),
        "formal_references": {
            "gsm8k": "official test 1319; never loaded by this builder",
            "humaneval": "official 164 used only for decontamination",
            "cmmlu": "official test 11582 used only for exact decontamination",
        },
        "source_stats": {
            "math": math_stats,
            "code": code_stats,
            "nlp": nlp_stats,
        },
    }
    manifest_path = resolve_path(artifacts["split_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "gate": "P0-A5-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a5_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "config": display_path(config_path),
        "config_hash": sha256_file(config_path),
        "manifest": display_path(manifest_path),
        "manifest_hash": sha256_file(manifest_path),
        "counts": counts,
        "overlaps": overlaps,
        "formal_training_reference_count": 0,
        "data_licenses": {
            "GSM8K": "MIT repository license",
            "OpenCodeInstruct": "CC BY 4.0",
            "COIG-CQIA": "dataset card currently says More Information Needed; raw rows are not committed",
            "CMMLU": "evaluation only; follow upstream non-commercial terms",
            "HumanEval": "evaluation/decontamination only; MIT repository license",
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    audit_path = resolve_path(artifacts["data_audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {display_path(manifest_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"counts={counts}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leak-safe P0-A5 training and gate data.")
    parser.add_argument("command", choices=("download", "build", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.command in {"download", "all"}:
            download_sources(config)
        if args.command in {"build", "all"}:
            build_all(config_path, args.workers)
    except (DataBuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 data build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
