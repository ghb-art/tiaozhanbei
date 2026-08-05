#!/usr/bin/env python3
"""Build leak-safe P0-A11 Code and Math accuracy-repair corpora."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/capability_v2/distill_train.jsonl"
INTERNAL = ROOT / "data/capability_v2/internal_validation.jsonl"
TOKENIZER_DIR = ROOT / "models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR = ROOT / "data/p0a11"
AUDIT = ROOT / "reports/audit/gate_p0a11_data.json"
MATH_TRACE = ROOT / "reports/audit/p0a11/math_mining_trace.jsonl"
HISTORY = (
    ROOT / "data/p0a6/quick_validation.jsonl",
    ROOT / "data/p0a6/full_validation.jsonl",
    ROOT / "data/p0a8/code_internal_validation.jsonl",
    ROOT / "data/p0a10/math_validation.jsonl",
    ROOT / "data/p0a10/code_validation.jsonl",
)
SEED = 20260802
CODE_TARGET = 16000
MATH_VALIDATION_ROWS = 300
MAX_SEQUENCE_LENGTH = 1536
MAX_GENERATION = {"math": 512, "code": 768}
PROMPTS = {
    "math": (
        "Solve the problem concisely. End with one line formatted as `#### 42`, "
        "where 42 is replaced by the actual numeric answer."
    ),
    "code": (
        "Return only a complete Python function implementation in one python code block. "
        "Do not use files, network access, third-party packages, or explanatory prose."
    ),
}
BANNED_CODE_NAMES = {
    "input", "open", "exec", "eval", "compile", "__import__", "breakpoint",
}
BANNED_MODULES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "pathlib",
    "numpy", "pandas", "torch", "tensorflow",
}


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


def stable_key(sample_id: str, namespace: str) -> str:
    return sha256_text(f"{SEED}:{namespace}:{sample_id}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    users = [
        item for item in messages or []
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(users) != 1:
        raise BuildError(f"Invalid messages: {row.get('sample_id')}")
    return str(users[0].get("content", ""))


def training_row(row: dict[str, Any], domain: str) -> dict[str, Any]:
    copied = dict(row)
    copied["messages"] = [
        {"role": "system", "content": PROMPTS[domain]},
        {"role": "user", "content": user_prompt(row)},
    ]
    copied["domain"] = domain
    copied["task_id"] = domain
    copied["split_role"] = "train"
    copied["answer_token_weight"] = 1.0
    copied["quality_weight"] = 1.0
    copied["training_weight"] = 1.0
    copied["kl_weight"] = 0.02 if domain == "code" else 0.20
    return copied


def final_number(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    expected = str(metadata.get("reference_answer", "")).replace(",", "").strip()
    if expected:
        return expected
    suffix = str(row.get("answer", "")).rsplit("####", 1)[-1]
    values = re.findall(r"-?\d+(?:\.\d+)?", suffix.replace(",", ""))
    if not values:
        raise BuildError(f"Missing math reference: {row.get('sample_id')}")
    return values[-1]


def validation_row(row: dict[str, Any], domain: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "dataset_key": str(row["dataset_key"]),
        "domain": domain,
        "source": str(row.get("source", "")),
        "split_role": "p0a11_internal_validation",
        "prompt": user_prompt(row),
        "validator": "exact_numeric_answer" if domain == "math" else "python_unit_tests",
    }
    if domain == "math":
        value.update({"reference": final_number(row), "unit_tests": []})
    else:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        tests = metadata.get("unit_tests") or row.get("unit_tests")
        if not isinstance(tests, list) or len(tests) != 10:
            raise BuildError(f"Code row lacks 10 tests: {row.get('sample_id')}")
        value.update({"reference": "unit_tests", "unit_tests": [str(x) for x in tests]})
    return value


def extract_python(answer: str) -> str:
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", answer, re.I | re.S)
    return (match.group(1) if match else answer).strip()


def human_eval_shape(row: dict[str, Any]) -> tuple[bool, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("independent_execution") != "passed":
        return False, "not_execution_verified"
    if float(metadata.get("average_test_score", 0.0)) != 1.0:
        return False, "not_full_test_score"
    tests = metadata.get("unit_tests")
    if not isinstance(tests, list) or len(tests) != 10:
        return False, "missing_ten_tests"
    prompt = user_prompt(row)
    answer = str(row.get("answer", ""))
    if len(prompt) > 3000 or len(answer) > 2400:
        return False, "too_long"
    code = extract_python(answer)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "invalid_python"
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions or any(isinstance(node, ast.ClassDef) for node in tree.body):
        return False, "not_function_task"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".", 1)[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "").split(".", 1)[0]]
            )
            if set(names) & BANNED_MODULES:
                return False, "banned_import"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CODE_NAMES:
                return False, "banned_call"
    lowered = prompt.casefold()
    if any(marker in lowered for marker in ("standard input", "stdin", "read from a file")):
        return False, "io_task"
    return True, "passed"


def token_lengths(tokenizer: Any, row: dict[str, Any]) -> tuple[int, int]:
    prompt_ids = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": str(row["answer"])}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False,
    )
    return len(prompt_ids), len(full_ids)


def historical_ids(domain: str) -> set[str]:
    used: set[str] = set()
    for path in HISTORY:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            row_domain = str(row.get("domain", ""))
            dataset = str(row.get("dataset_key", ""))
            if row_domain == domain or (domain == "code" and dataset == "opencodeinstruct") or (
                domain == "math" and dataset == "gsm8k"
            ):
                used.add(str(row.get("sample_id", "")))
    return used


def prepare() -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("transformers is required") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, trust_remote_code=True
    )
    source = read_jsonl(SOURCE)
    internal = read_jsonl(INTERNAL)
    gsm = [row for row in source if row.get("dataset_key") == "gsm8k"]
    code_source = [row for row in source if row.get("dataset_key") == "opencodeinstruct"]
    if len(gsm) != 7173 or len(code_source) != 20000:
        raise BuildError("Unexpected source counts")

    math_used = historical_ids("math")
    math_candidates = [row for row in gsm if str(row["sample_id"]) not in math_used]
    math_candidates.sort(key=lambda row: stable_key(str(row["sample_id"]), "math-val"))
    if len(math_candidates) < MATH_VALIDATION_ROWS + 1000:
        raise BuildError(f"Insufficient Math candidates: {len(math_candidates)}")
    math_validation_source = math_candidates[:MATH_VALIDATION_ROWS]
    math_validation_ids = {str(row["sample_id"]) for row in math_validation_source}
    # Mine only rows that have never been used for checkpoint selection.  The
    # P0-A11 holdout is also excluded so neither old nor new validation labels
    # can flow back into training.
    math_pool_source = [
        row for row in math_candidates
        if str(row["sample_id"]) not in math_validation_ids
    ]

    code_used = historical_ids("code")
    fresh_internal = [
        row for row in internal
        if row.get("dataset_key") == "opencodeinstruct"
        and str(row.get("sample_id", "")) not in code_used
    ]
    fresh_internal.sort(key=lambda row: stable_key(str(row["sample_id"]), "code-val"))
    code_validation: list[dict[str, Any]] = []
    code_validation_rejections: Counter[str] = Counter()
    for row in fresh_internal:
        prepared = training_row(row, "code")
        prompt_tokens, _full_tokens = token_lengths(tokenizer, prepared)
        if prompt_tokens + MAX_GENERATION["code"] > MAX_SEQUENCE_LENGTH:
            code_validation_rejections["prompt_budget"] += 1
            continue
        code_validation.append(validation_row(row, "code"))

    code_rejections: Counter[str] = Counter()
    eligible_code: list[dict[str, Any]] = []
    for row in code_source:
        sample_id = str(row["sample_id"])
        if sample_id in code_used:
            code_rejections["historical_validation"] += 1
            continue
        accepted, reason = human_eval_shape(row)
        if not accepted:
            code_rejections[reason] += 1
            continue
        prepared = training_row(row, "code")
        _prompt_tokens, full_tokens = token_lengths(tokenizer, prepared)
        if full_tokens > MAX_SEQUENCE_LENGTH:
            code_rejections["sequence_budget"] += 1
            continue
        eligible_code.append(prepared)
    eligible_code.sort(key=lambda row: stable_key(str(row["sample_id"]), "code-train"))
    if len(eligible_code) < CODE_TARGET:
        raise BuildError(f"Only {len(eligible_code)} Code rows passed; need {CODE_TARGET}")
    code_train = eligible_code[:CODE_TARGET]

    math_validation = [validation_row(row, "math") for row in math_validation_source]
    math_pool = [validation_row(row, "math") for row in math_pool_source]
    outputs = {
        "code_train": code_train,
        "code_validation": code_validation,
        "math_validation": math_validation,
        "math_mining_pool": math_pool,
    }
    output_meta: dict[str, Any] = {}
    for name, rows in outputs.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        output_meta[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    train_ids = {str(row["sample_id"]) for row in code_train}
    validation_ids = {str(row["sample_id"]) for row in code_validation}
    if train_ids & validation_ids:
        raise BuildError("Code train-validation overlap")
    audit = {
        "gate": "P0-A11-ACCURACY-REPAIR-DATA",
        "check_version": "1.0-prepared",
        "created_by": "model_compression/build_p0a11_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "policy": {
            "gate300_opened": False,
            "formal_full_opened": False,
            "formal_test_rows_used": 0,
            "p0a9_per_question_feedback_used": False,
            "p0a10_validation_feedback": "domain aggregates and format diagnostics only",
        },
        "counts": {
            "code_train": len(code_train),
            "code_validation": len(code_validation),
            "math_validation": len(math_validation),
            "math_mining_pool": len(math_pool),
        },
        "code_filter": {
            "eligible_before_cap": len(eligible_code),
            "selected": len(code_train),
            "rejections": dict(sorted(code_rejections.items())),
            "validation_rejections": dict(sorted(code_validation_rejections.items())),
            "independent_tests_per_row": 10,
        },
        "history_hashes": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in HISTORY if path.is_file()
        },
        "inputs": {
            SOURCE.relative_to(ROOT).as_posix(): sha256_file(SOURCE),
            INTERNAL.relative_to(ROOT).as_posix(): sha256_file(INTERNAL),
        },
        "outputs": output_meta,
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(
        f"P0-A11 prepared code_train={len(code_train)} code_validation={len(code_validation)} "
        f"math_validation={len(math_validation)} math_pool={len(math_pool)}"
    )
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=prepared")
    return 0


def finalize() -> int:
    if not AUDIT.is_file() or not MATH_TRACE.is_file():
        raise BuildError("Run prepare and Math mining first")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") not in {"prepared", "passed"}:
        raise BuildError("Prepared data audit is not usable")
    pool = read_jsonl(OUTPUT_DIR / "math_mining_pool.jsonl")
    trace = read_jsonl(MATH_TRACE)
    if len(trace) != len(pool):
        raise BuildError(f"Math trace rows {len(trace)} != pool rows {len(pool)}")
    by_id = {str(row["sample_id"]): row for row in trace}
    if len(by_id) != len(trace):
        raise BuildError("Duplicate Math mining trace ids")
    if any(str(row.get("generation_error", "")) for row in trace):
        raise BuildError("Math mining has generation errors")
    pool_ids = [str(row["sample_id"]) for row in pool]
    if set(pool_ids) != set(by_id):
        raise BuildError("Math mining trace identity mismatch")
    wrong_ids = [sample_id for sample_id in pool_ids if not bool(by_id[sample_id]["correct"])]
    wrong_id_set = set(wrong_ids)
    correct_ids = [sample_id for sample_id in pool_ids if bool(by_id[sample_id]["correct"])]
    replay_count = min(len(correct_ids), math.ceil(len(wrong_ids) / 2))
    correct_ids.sort(key=lambda sample_id: stable_key(sample_id, "math-replay"))
    replay_ids = correct_ids[:replay_count]
    selected_ids = wrong_id_set | set(replay_ids)
    source_rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(SOURCE)
        if row.get("dataset_key") == "gsm8k"
    }
    math_train: list[dict[str, Any]] = []
    for sample_id in sorted(selected_ids, key=lambda value: stable_key(value, "math-train")):
        row = training_row(source_rows[sample_id], "math")
        is_error = sample_id in wrong_id_set
        row["training_weight"] = 1.0 if is_error else 0.5
        row["hard_mining_role"] = "base_error" if is_error else "correct_replay"
        math_train.append(row)
    path = OUTPUT_DIR / "math_train.jsonl"
    write_jsonl(path, math_train)
    validation_ids = {
        str(row["sample_id"]) for row in read_jsonl(OUTPUT_DIR / "math_validation.jsonl")
    }
    if selected_ids & validation_ids:
        raise BuildError("Math train-validation overlap")
    audit["check_version"] = "1.0"
    audit["status"] = "passed"
    audit["created_ts"] = datetime.now(timezone.utc).isoformat()
    audit["math_mining"] = {
        "pool_rows": len(pool),
        "base_error_rows": len(wrong_ids),
        "correct_rows": len(correct_ids),
        "correct_replay_rows": len(replay_ids),
        "training_rows": len(math_train),
        "error_to_replay_ratio": (
            len(wrong_ids) / len(replay_ids) if replay_ids else None
        ),
        "trace": MATH_TRACE.relative_to(ROOT).as_posix(),
        "trace_hash": sha256_file(MATH_TRACE),
    }
    audit["counts"]["math_train"] = len(math_train)
    audit["outputs"]["math_train"] = {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(math_train),
        "sha256": sha256_file(path),
    }
    audit.pop("report_hash", None)
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(
        f"P0-A11 finalized math_errors={len(wrong_ids)} replay={len(replay_ids)} "
        f"math_train={len(math_train)}"
    )
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize", "status"))
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "finalize":
        return finalize()
    if not AUDIT.is_file():
        raise BuildError("P0-A11 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "counts": value.get("counts")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A11 data build failed: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
