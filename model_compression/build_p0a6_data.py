#!/usr/bin/env python3
"""Build the accuracy-focused, train-only P0-A6 datasets.

This builder deliberately reads only the P0-A5 train/internal-validation
artifacts and the labelled C-Eval ``val``/``dev`` splits.  It never discovers
or reads C-Eval ``test`` or anything below ``data/eval``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SOURCE = ROOT / "data/capability_v2/distill_train.jsonl"
DEFAULT_VALIDATION_SOURCE = ROOT / "data/capability_v2/internal_validation.jsonl"
DEFAULT_CEVAL_ROOT = ROOT / "data/datasets/ceval_exam"
DEFAULT_OUTPUT_DIR = ROOT / "data/p0a6"
DEFAULT_AUDIT = ROOT / "reports/audit/gate_p0a6_data.json"

SYSTEM_PROMPT = "Give a concise, verifiable answer in the requested format."
TASK_MASS = {"math": 0.35, "code": 0.30, "nlp": 0.35}
KL_WEIGHTS = {"math": 0.60, "code": 0.15, "nlp": 0.20}
CHOICES = ("A", "B", "C", "D")
CEVAL_DOWNLOAD_HELP = (
    "Download the official labelled C-Eval data, for example: "
    "git clone https://github.com/hkust-nlp/ceval.git data/datasets/ceval; "
    "then ensure labelled CSV/JSON files are present below dev and val directories, "
    "or download ceval/ceval-exam Parquet shards into "
    "data/datasets/ceval_exam/<subject>/{val,dev}-*.parquet. "
    "The unlabelled test split must not be supplied to this builder."
)


class DataBuildError(RuntimeError):
    """Raised when a source violates the P0-A6 data protocol."""


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


def stable_key(seed: int, identity: str) -> str:
    return sha256_text(f"{seed}:{identity}")


def normalized_question(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def guard_non_evaluation_source(path: Path) -> None:
    if _is_below(path, ROOT / "data/eval"):
        raise DataBuildError(f"Forbidden evaluation source: {display_path(path)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    guard_non_evaluation_source(path)
    if not path.is_file():
        raise DataBuildError(f"Missing JSONL source: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataBuildError(
                    f"Invalid JSON at {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DataBuildError(
                    f"Non-object row at {display_path(path)}:{line_number}"
                )
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def valid_training_schema(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("sample_id", "")).strip()
        and user_prompt(row)
        and str(row.get("answer", "")).strip()
    )


def extract_code(value: str) -> str:
    match = re.search(
        r"```(?:python)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL
    )
    return (match.group(1) if match else value).strip()


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def humaneval_style_code_reason(
    row: dict[str, Any],
    *,
    min_unit_tests: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> str:
    """Return an empty string only for a short, executable single-function task."""

    if not valid_training_schema(row):
        return "invalid_training_schema"
    prompt = user_prompt(row)
    answer = str(row["answer"])
    if len(prompt) > max_prompt_chars:
        return "prompt_too_long"
    if len(answer) > max_answer_chars:
        return "answer_too_long"
    metadata = row.get("metadata")
    tests = metadata.get("unit_tests") if isinstance(metadata, dict) else None
    if not isinstance(tests, list) or len(tests) < min_unit_tests:
        return "insufficient_unit_tests"
    if any(not isinstance(item, str) or not item.strip() for item in tests):
        return "invalid_unit_tests"

    source = extract_code(answer)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "invalid_python"
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        return "not_one_sync_function"
    function = functions[0]
    if function not in tree.body:
        return "nested_function_only"
    if any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
        return "contains_class"
    allowed_top_level = (ast.Import, ast.ImportFrom, ast.FunctionDef)
    for node in tree.body:
        if isinstance(node, allowed_top_level):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        return "top_level_execution"
    unsafe_calls = {
        _call_name(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    } & {"input", "print", "open", "exec", "eval", "compile", "__import__"}
    if unsafe_calls:
        return "io_or_dynamic_execution"
    if not re.search(rf"\b{re.escape(function.name)}\s*\(", prompt):
        return "signature_not_in_prompt"

    referenced = False
    for test in tests:
        try:
            test_tree = ast.parse(test)
        except SyntaxError:
            return "invalid_unit_tests"
        if not any(isinstance(node, ast.Assert) for node in ast.walk(test_tree)):
            return "unit_test_without_assert"
        if any(
            isinstance(node, ast.Call) and _call_name(node) == function.name
            for node in ast.walk(test_tree)
        ):
            referenced = True
    if not referenced:
        return "tests_do_not_call_function"
    if isinstance(metadata, dict) and metadata.get("independent_execution") not in {
        None,
        "passed",
    }:
        return "not_execution_verified"
    return ""


def copy_training_row(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    copied["task_id"] = task_id
    copied["split_role"] = "train"
    copied["kl_weight"] = KL_WEIGHTS[task_id]
    copied["answer_token_weight"] = 1.0
    copied.pop("preserve_math", None)
    return copied


def code_training_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = copy_training_row(row, "code")
    metadata = copied.get("metadata")
    assert isinstance(metadata, dict)
    copied["metadata"] = dict(metadata)
    copied["metadata"]["p0a6_filter"] = "humaneval_style_single_function"
    return copied


def validation_row(
    *,
    sample_id: str,
    domain: str,
    dataset_key: str,
    prompt: str,
    reference: str,
    unit_tests: list[str],
    validator: str,
    source: str,
    split_role: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "task_id": domain,
        "dataset_key": dataset_key,
        "prompt": prompt,
        "reference": reference,
        "unit_tests": unit_tests,
        "validator": validator,
        "source": source,
        "split_role": split_role,
    }


def extract_math_reference(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and str(metadata.get("reference_answer", "")).strip():
        return str(metadata["reference_answer"]).strip()
    answer = str(row.get("answer", ""))
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return ""


def adapt_internal_validation(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    adapted: dict[str, list[dict[str, Any]]] = {"math": [], "code": []}
    rejected: Counter[str] = Counter()
    for row in rows:
        dataset_key = str(row.get("dataset_key", ""))
        if dataset_key == "gsm8k":
            prompt = user_prompt(row)
            reference = extract_math_reference(row)
            if not prompt or not reference:
                rejected["math_invalid_schema"] += 1
                continue
            adapted["math"].append(
                validation_row(
                    sample_id=str(row["sample_id"]),
                    domain="math",
                    dataset_key="gsm8k",
                    prompt=prompt,
                    reference=reference,
                    unit_tests=[],
                    validator="numeric_exact",
                    source=str(row.get("source", "GSM8K-train")),
                    split_role="full_validation",
                )
            )
        elif dataset_key == "opencodeinstruct":
            reason = humaneval_style_code_reason(
                row,
                min_unit_tests=args.min_unit_tests,
                max_prompt_chars=args.max_code_prompt_chars,
                max_answer_chars=args.max_code_answer_chars,
            )
            if reason:
                rejected[f"code_{reason}"] += 1
                continue
            metadata = row["metadata"]
            adapted["code"].append(
                validation_row(
                    sample_id=str(row["sample_id"]),
                    domain="code",
                    dataset_key="opencodeinstruct",
                    prompt=user_prompt(row),
                    reference="",
                    unit_tests=[str(item) for item in metadata["unit_tests"]],
                    validator="python_unit_tests",
                    source=str(row.get("source", "nvidia/OpenCodeInstruct")),
                    split_role="full_validation",
                )
            )
    return adapted, rejected


def _ceval_file_is_safe(path: Path) -> bool:
    lowered_parts = {part.casefold() for part in path.parts}
    return "test" not in lowered_parts and not path.stem.casefold().endswith("_test")


def discover_ceval_files(root: Path, split: str) -> list[Path]:
    if split not in {"dev", "val"}:
        raise DataBuildError(f"Forbidden C-Eval split request: {split}")
    if not root.is_dir():
        raise DataBuildError(
            f"Missing C-Eval root: {display_path(root)}. {CEVAL_DOWNLOAD_HELP}"
        )
    extensions = {".csv", ".json", ".jsonl", ".parquet"}
    discovered: set[Path] = set()
    for directory in root.rglob(split):
        if directory.is_dir() and directory.name.casefold() == split:
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.casefold() in extensions:
                    discovered.add(path.resolve())
    for path in root.rglob(f"*_{split}.*"):
        if path.is_file() and path.suffix.casefold() in extensions:
            discovered.add(path.resolve())
    # Hugging Face ceval/ceval-exam layout:
    #   <root>/<subject>/val-00000-of-00001.parquet
    #   <root>/<subject>/dev-00000-of-00001.parquet
    for path in root.rglob(f"{split}-*.parquet"):
        if path.is_file():
            discovered.add(path.resolve())
    safe = sorted(path for path in discovered if _ceval_file_is_safe(path))
    if not safe:
        raise DataBuildError(
            f"Missing labelled C-Eval {split} files below {display_path(root)}. "
            f"{CEVAL_DOWNLOAD_HELP}"
        )
    return safe


def _raw_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise DataBuildError(
                "pyarrow is required to read the downloaded C-Eval Parquet shards"
            ) from exc
        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.casefold() == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(value.get("data"), list):
        records = value["data"]
    else:
        raise DataBuildError(f"Unsupported C-Eval JSON structure: {display_path(path)}")
    if not all(isinstance(item, dict) for item in records):
        raise DataBuildError(f"Non-object C-Eval record: {display_path(path)}")
    return [dict(item) for item in records]


def _casefold_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in row.items()}


def _ceval_choice(value: Any) -> str:
    if isinstance(value, int) and 0 <= value < 4:
        return CHOICES[value]
    text = str(value).strip().upper()
    match = re.fullmatch(r"(?:OPTION\s*)?([ABCD])", text)
    return match.group(1) if match else ""


def _ceval_options(row: dict[str, Any], lowered: dict[str, Any]) -> dict[str, str]:
    direct = {choice: str(lowered.get(choice.casefold(), "")).strip() for choice in CHOICES}
    if all(direct.values()):
        return direct
    raw = lowered.get("options", lowered.get("choices"))
    if isinstance(raw, dict):
        folded = _casefold_keys(raw)
        direct = {choice: str(folded.get(choice.casefold(), "")).strip() for choice in CHOICES}
    elif isinstance(raw, list) and len(raw) == 4:
        direct = {choice: str(raw[index]).strip() for index, choice in enumerate(CHOICES)}
    return direct


def format_mcq_prompt(question: str, options: dict[str, str]) -> str:
    return "\n".join([f"问题：{question}"] + [f"{key}. {options[key]}" for key in CHOICES])


def load_ceval_split(
    root: Path, split: str
) -> tuple[list[dict[str, Any]], list[Path], Counter[str]]:
    files = discover_ceval_files(root, split)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for path in files:
        if not _ceval_file_is_safe(path):
            raise DataBuildError(f"Refusing C-Eval test-like file: {display_path(path)}")
        if path.suffix.casefold() == ".parquet" and path.stem.casefold().startswith(
            f"{split}-"
        ):
            subject = path.parent.name
        else:
            subject = re.sub(rf"_{split}$", "", path.stem, flags=re.IGNORECASE)
        for index, raw in enumerate(_raw_records(path)):
            lowered = _casefold_keys(raw)
            question = str(
                lowered.get("question", lowered.get("prompt", lowered.get("input", "")))
            ).strip()
            answer = _ceval_choice(
                lowered.get("answer", lowered.get("label", lowered.get("reference", "")))
            )
            options = _ceval_options(raw, lowered)
            if not question or not answer or not all(options.values()):
                rejected["invalid_labelled_mcq"] += 1
                continue
            identity = normalized_question(question)
            if not identity or identity in seen:
                rejected["duplicate_question"] += 1
                continue
            seen.add(identity)
            explanation = str(
                lowered.get("explanation", lowered.get("rationale", lowered.get("analysis", "")))
            ).strip()
            raw_id = str(lowered.get("id", lowered.get("index", index))).strip()
            rows.append(
                {
                    "sample_id": f"ceval/{split}/{subject}/{raw_id}",
                    "dataset_key": "ceval",
                    "task_id": "nlp",
                    "source": "C-Eval-labelled",
                    "split_role": "train" if split == "val" else "full_validation",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": format_mcq_prompt(question, options)},
                    ],
                    "answer": (
                        f"简短分析：{explanation}\n最终答案：{answer}"
                        if explanation
                        else f"最终答案：{answer}"
                    ),
                    "metadata": {
                        "subject": subject,
                        "reference_answer": answer,
                        "options": options,
                        "human_labelled": True,
                        "ceval_split": split,
                    },
                    "kl_weight": KL_WEIGHTS["nlp"],
                    "answer_token_weight": 2.0,
                    "_identity": identity,
                }
            )
    if not rows:
        raise DataBuildError(
            f"No valid labelled C-Eval {split} rows were found. {CEVAL_DOWNLOAD_HELP}"
        )
    return rows, files, rejected


def ceval_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        output.append(
            validation_row(
                sample_id=str(row["sample_id"]),
                domain="nlp",
                dataset_key="ceval",
                prompt=user_prompt(row),
                reference=str(metadata["reference_answer"]),
                unit_tests=[],
                validator="choice_exact",
                source="C-Eval-labelled-dev",
                split_role="full_validation",
            )
        )
    return output


def _quality_weight(row: dict[str, Any], task_id: str) -> float:
    validation = str(row.get("distill_validation", ""))
    if task_id == "code" and "fallback" in validation:
        return 0.50
    if task_id == "math" and "fallback" in validation:
        return 0.75
    return 1.0


def assign_training_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row["task_id"])
        if task_id == "nlp":
            group = "nlp_mcq" if row["dataset_key"] == "ceval" else "nlp_open"
        else:
            group = task_id
        row["quality_weight"] = _quality_weight(row, task_id)
        groups[group].append(row)
    required = {"math", "code", "nlp_open", "nlp_mcq"}
    if set(groups) != required:
        raise DataBuildError(
            f"Training groups are incomplete: got={sorted(groups)}, need={sorted(required)}"
        )
    group_mass = {
        "math": TASK_MASS["math"],
        "code": TASK_MASS["code"],
        "nlp_open": TASK_MASS["nlp"] / 2,
        "nlp_mcq": TASK_MASS["nlp"] / 2,
    }
    total_rows = len(rows)
    for group, selected in groups.items():
        quality_total = sum(float(row["quality_weight"]) for row in selected)
        if quality_total <= 0:
            raise DataBuildError(f"Non-positive quality mass for {group}")
        for row in selected:
            row["training_weight"] = (
                group_mass[group]
                * total_rows
                * float(row["quality_weight"])
                / quality_total
            )
    effective: Counter[str] = Counter()
    for row in rows:
        effective[str(row["task_id"])] += float(row["training_weight"]) / total_rows
    for task_id, expected in TASK_MASS.items():
        if abs(effective[task_id] - expected) > 1e-9:
            raise DataBuildError(
                f"Task mass mismatch for {task_id}: {effective[task_id]} != {expected}"
            )
    return dict(sorted(effective.items()))


def validate_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    counts = Counter(str(row.get("sample_id", "")) for row in rows)
    duplicates = sorted(key for key, value in counts.items() if not key or value > 1)
    if duplicates:
        raise DataBuildError(f"Duplicate/empty sample ids in {label}: {duplicates[:5]}")


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    train_source = resolve_path(args.train_source)
    validation_source = resolve_path(args.validation_source)
    ceval_root = resolve_path(args.ceval_root)
    output_dir = resolve_path(args.output_dir)
    audit_path = resolve_path(args.audit)
    guard_non_evaluation_source(train_source)
    guard_non_evaluation_source(validation_source)

    # Discover both labelled C-Eval splits before scanning the larger JSONL inputs,
    # so a missing dependency fails immediately and with an actionable message.
    ceval_val, ceval_val_files, ceval_val_rejected = load_ceval_split(ceval_root, "val")
    ceval_dev, ceval_dev_files, ceval_dev_rejected = load_ceval_split(ceval_root, "dev")
    dev_identities = {str(row.pop("_identity")) for row in ceval_dev}
    retained_ceval_val: list[dict[str, Any]] = []
    ceval_split_overlap = 0
    for row in ceval_val:
        identity = str(row.pop("_identity"))
        if identity in dev_identities:
            ceval_split_overlap += 1
            continue
        retained_ceval_val.append(row)
    ceval_val = retained_ceval_val
    if not ceval_val:
        raise DataBuildError("All C-Eval val rows overlap dev; refusing validation leakage")

    source_train = read_jsonl(train_source)
    source_validation = read_jsonl(validation_source)
    rejected: Counter[str] = Counter()
    train: list[dict[str, Any]] = []
    for row in source_train:
        dataset_key = str(row.get("dataset_key", ""))
        if dataset_key == "gsm8k":
            if not valid_training_schema(row):
                rejected["math_invalid_training_schema"] += 1
                continue
            train.append(copy_training_row(row, "math"))
        elif dataset_key == "opencodeinstruct":
            reason = humaneval_style_code_reason(
                row,
                min_unit_tests=args.min_unit_tests,
                max_prompt_chars=args.max_code_prompt_chars,
                max_answer_chars=args.max_code_answer_chars,
            )
            if reason:
                rejected[f"code_{reason}"] += 1
                continue
            train.append(code_training_row(row))
        elif dataset_key in {"cmmlu", "coig_cqia"}:
            if not valid_training_schema(row):
                rejected["nlp_invalid_training_schema"] += 1
                continue
            copied = copy_training_row(row, "nlp")
            copied["dataset_key"] = "coig_cqia"
            copied["answer_token_weight"] = 1.0
            train.append(copied)
        else:
            rejected["unsupported_dataset_key"] += 1
    train.extend(ceval_val)

    counts = Counter(str(row["task_id"]) for row in train)
    minimums = {
        "math": args.min_math_train,
        "code": args.min_code_train,
        "nlp": args.min_nlp_train,
    }
    for task_id, minimum in minimums.items():
        if counts[task_id] < minimum:
            raise DataBuildError(
                f"Only {counts[task_id]} P0-A6 {task_id} training rows; need at least {minimum}"
            )
    effective_mass = assign_training_weights(train)
    train.sort(key=lambda row: stable_key(args.seed, str(row["sample_id"])))
    validate_unique_ids(train, "train")

    validation, validation_rejected = adapt_internal_validation(source_validation, args)
    validation["nlp"] = ceval_validation_rows(ceval_dev)
    rejected.update(validation_rejected)
    for domain in ("math", "code", "nlp"):
        if len(validation[domain]) < args.quick_per_domain:
            raise DataBuildError(
                f"Only {len(validation[domain])} {domain} validation rows; "
                f"need {args.quick_per_domain} for quick validation"
            )
    if len(validation["code"]) < args.min_code_validation:
        raise DataBuildError(
            f"Only {len(validation['code'])} strict Code validation rows; "
            f"need at least {args.min_code_validation}"
        )

    full_validation: list[dict[str, Any]] = []
    quick_validation: list[dict[str, Any]] = []
    for domain in ("math", "code", "nlp"):
        ordered = sorted(
            validation[domain],
            key=lambda row: stable_key(args.seed + 1, str(row["sample_id"])),
        )
        full_validation.extend(ordered)
        for row in ordered[: args.quick_per_domain]:
            copied = dict(row)
            copied["split_role"] = "quick_validation"
            quick_validation.append(copied)
    full_validation.sort(key=lambda row: (str(row["domain"]), str(row["sample_id"])))
    quick_validation.sort(key=lambda row: (str(row["domain"]), str(row["sample_id"])))
    validate_unique_ids(full_validation, "full_validation")
    validate_unique_ids(quick_validation, "quick_validation")

    train_ids = {str(row["sample_id"]) for row in train}
    validation_ids = {str(row["sample_id"]) for row in full_validation}
    overlap_count = len(train_ids & validation_ids)
    if overlap_count:
        raise DataBuildError(f"Train/full-validation sample id overlap: {overlap_count}")

    train_path = output_dir / "train.jsonl"
    quick_path = output_dir / "quick_validation.jsonl"
    full_path = output_dir / "full_validation.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(quick_path, quick_validation)
    write_jsonl_atomic(full_path, full_validation)

    train_dataset_counts = dict(sorted(Counter(str(row["dataset_key"]) for row in train).items()))
    train_task_counts = dict(sorted(Counter(str(row["task_id"]) for row in train).items()))
    quick_counts = dict(
        sorted(Counter(str(row["domain"]) for row in quick_validation).items())
    )
    full_counts = dict(
        sorted(Counter(str(row["domain"]) for row in full_validation).items())
    )
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "protocol": "P0-A6",
        "created_by": "model_compression/build_p0a6_data.py",
        "created_ts": now,
        "train": {
            "path": display_path(train_path),
            "rows": len(train),
            "sha256": sha256_file(train_path),
        },
        "validation": {
            "quick": {
                "path": display_path(quick_path),
                "rows": len(quick_validation),
                "sha256": sha256_file(quick_path),
                "expected_counts": quick_counts,
            },
            "full": {
                "path": display_path(full_path),
                "rows": len(full_validation),
                "sha256": sha256_file(full_path),
                "expected_counts": full_counts,
            },
        },
    }
    write_json_atomic(manifest_path, manifest)
    audit: dict[str, Any] = {
        "gate": "P0-A6-ACCURACY-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a6_data.py",
        "created_ts": now,
        "status": "passed",
        "policy": {
            "task_mass": TASK_MASS,
            "nlp_internal_mass": {"coig_open_qa": 0.50, "ceval_mcq": 0.50},
            "kl_weight": KL_WEIGHTS,
            "mcq_answer_token_weight": 2.0,
            "code_filter": "HumanEval-style one synchronous top-level Python function",
            "formal_test_loaded": False,
            "data_eval_loaded": False,
        },
        "sources": {
            "distill_train": {
                "path": display_path(train_source),
                "sha256": sha256_file(train_source),
                "rows": len(source_train),
            },
            "internal_validation": {
                "path": display_path(validation_source),
                "sha256": sha256_file(validation_source),
                "rows": len(source_validation),
            },
            "ceval_val": [
                {"path": display_path(path), "sha256": sha256_file(path)}
                for path in ceval_val_files
            ],
            "ceval_dev": [
                {"path": display_path(path), "sha256": sha256_file(path)}
                for path in ceval_dev_files
            ],
        },
        "counts": {
            "train_by_task": train_task_counts,
            "train_by_dataset": train_dataset_counts,
            "quick_validation": quick_counts,
            "full_validation": full_counts,
        },
        "effective_task_mass": effective_mass,
        "rejections": dict(sorted(rejected.items())),
        "ceval": {
            "val_rejections": dict(sorted(ceval_val_rejected.items())),
            "dev_rejections": dict(sorted(ceval_dev_rejected.items())),
            "val_dev_question_overlap_removed_from_train": ceval_split_overlap,
        },
        "overlaps": {
            "train_full_validation_sample_ids": overlap_count,
            "ceval_val_dev_questions_removed": ceval_split_overlap,
        },
        "outputs": {
            "train": {
                "path": display_path(train_path),
                "rows": len(train),
                "sha256": sha256_file(train_path),
            },
            "quick_validation": {
                "path": display_path(quick_path),
                "rows": len(quick_validation),
                "sha256": sha256_file(quick_path),
            },
            "full_validation": {
                "path": display_path(full_path),
                "rows": len(full_validation),
                "sha256": sha256_file(full_path),
            },
            "manifest": {
                "path": display_path(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    write_json_atomic(audit_path, audit)
    print(f"Wrote {display_path(train_path)} rows={len(train)}")
    print(f"Wrote {display_path(quick_path)} counts={quick_counts}")
    print(f"Wrote {display_path(full_path)} counts={full_counts}")
    print(f"Wrote {display_path(manifest_path)}")
    print(f"Wrote {display_path(audit_path)} status=passed")
    return audit


def write_failure_audit(args: argparse.Namespace, error: Exception) -> None:
    audit_path = resolve_path(args.audit)
    audit = {
        "gate": "P0-A6-ACCURACY-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a6_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "errors": [str(error)],
        "formal_test_loaded": False,
        "data_eval_loaded": False,
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    write_json_atomic(audit_path, audit)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leak-safe P0-A6 accuracy train and internal-validation data."
    )
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--validation-source", default=str(DEFAULT_VALIDATION_SOURCE))
    parser.add_argument("--ceval-root", default=str(DEFAULT_CEVAL_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--quick-per-domain", type=int, default=100)
    parser.add_argument("--min-unit-tests", type=int, default=3)
    parser.add_argument("--max-code-prompt-chars", type=int, default=4000)
    parser.add_argument("--max-code-answer-chars", type=int, default=4000)
    parser.add_argument("--min-math-train", type=int, default=1000)
    parser.add_argument("--min-code-train", type=int, default=3000)
    parser.add_argument("--min-nlp-train", type=int, default=1000)
    parser.add_argument("--min-code-validation", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_outputs(args)
    except (DataBuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        try:
            write_failure_audit(args, exc)
        except OSError:
            pass
        print(f"P0-A6 data build failed: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
