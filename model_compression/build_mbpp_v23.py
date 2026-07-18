#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_chapter2_capability import build_messages, extract_function_block, run_assert_source_check  # noqa: E402
from generate_teacher_capability_distill import (  # noqa: E402
    code_source_candidates,
    normalize_code_body,
    safe_code_ast,
)


EXPECTED_MBPP_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
DEFAULT_INPUT = ROOT / "data" / "datasets" / "mbpp" / "mbpp.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "distill" / "mbpp_v23_train_source.jsonl"
DEFAULT_DEV_SELECT = ROOT / "data" / "distill" / "mbpp_v23_dev_select.jsonl"
DEFAULT_DEV_GATE = ROOT / "data" / "distill" / "mbpp_v23_dev_gate.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_mbpp_v23_code_data.json"
HUMANEVAL_PATH = ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
ALLOWED_SETUP_NODES = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)


class MbppBuildError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MbppBuildError(f"Invalid JSON at {display_path(path)}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise MbppBuildError(f"Expected object at {display_path(path)}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def module_setup_code(module: ast.Module, function: ast.FunctionDef) -> tuple[str, str]:
    setup_nodes: list[ast.stmt] = []
    for node in module.body:
        if node is function:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, ALLOWED_SETUP_NODES):
            return "", f"unsupported_top_level_{type(node).__name__}"
        setup_nodes.append(node)
    return "\n".join(ast.unparse(node) for node in setup_nodes).strip(), ""


def function_prompt(function: ast.FunctionDef, description: str, visible_tests: list[str]) -> str:
    contract_lines = [description.strip(), "", "Examples that must hold:"]
    contract_lines.extend(str(test).strip() for test in visible_tests if str(test).strip())
    contract = "\n".join(contract_lines).strip()
    prompt_function = ast.FunctionDef(
        name=function.name,
        args=function.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=function.returns,
        type_comment=function.type_comment,
    )
    ast.fix_missing_locations(prompt_function)
    signature = ast.unparse(prompt_function).splitlines()[0]
    return f"{signature}\n    {json.dumps(contract, ensure_ascii=False)}\n"


def normalized_tests(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    visible = [str(test).strip() for test in row.get("test_list", []) if str(test).strip()]
    challenge = [str(test).strip() for test in row.get("challenge_test_list", []) if str(test).strip()]
    combined = list(dict.fromkeys(visible + challenge))
    return visible, combined


def build_row(row: dict[str, Any], role: str, timeout_sec: float) -> tuple[dict[str, Any] | None, str]:
    task_id = int(row.get("task_id", -1))
    code = str(row.get("code", ""))
    try:
        module = ast.parse(code)
    except (IndentationError, SyntaxError):
        return None, "canonical_syntax_error"
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    async_functions = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef)]
    if len(functions) != 1 or classes or async_functions:
        return None, "not_single_top_level_function"
    function = functions[0]
    setup_code, setup_error = module_setup_code(module, function)
    if setup_error:
        return None, setup_error
    test_setup = str(row.get("test_setup_code", "")).strip()
    setup_parts = [part for part in [setup_code, test_setup] if part]
    combined_setup = "\n".join(dict.fromkeys(setup_parts))
    visible_tests, execution_tests = normalized_tests(row)
    if not visible_tests or not execution_tests:
        return None, "missing_assert_tests"
    prompt_source = function_prompt(function, str(row.get("text", "")), visible_tests)
    full_prompt_source = f"{combined_setup}\n\n{prompt_source}" if combined_setup else prompt_source
    extracted_prompt = extract_function_block(prompt_source, function.name)
    if extracted_prompt != prompt_source:
        return None, "prompt_extraction_mismatch"
    try:
        answer = normalize_code_body(code, function.name)
    except (StopIteration, ValueError, SyntaxError, MbppBuildError):
        return None, "canonical_body_normalization_failed"
    candidates = code_source_candidates(prompt_source, function.name, answer)
    canonical_source = next(
        (
            candidate
            for candidate in candidates
            if safe_code_ast(candidate)
            and run_assert_source_check(candidate, execution_tests, timeout_sec, combined_setup)[0]
        ),
        None,
    )
    if canonical_source is None:
        return None, "canonical_assertions_failed"
    sample = {
        "dataset_key": "humaneval",
        "sample_id": f"mbpp/{role}/{task_id:04d}",
        "prompt": full_prompt_source,
        "entry_point": function.name,
    }
    messages, prompt_hash = build_messages(sample, "v11")
    output = {
        "rehearsal_version": "23.0-mbpp-independent",
        "created_by": "model_compression/build_mbpp_v23.py",
        "source": f"mbpp_official_{role}",
        "dataset_key": "humaneval",
        "sample_id": sample["sample_id"],
        "validation_group_id": f"mbpp/task/{task_id:04d}",
        "mbpp_task_id": task_id,
        "split_role": role,
        "messages": messages,
        "answer": answer,
        "prompt_style": "v11",
        "prompt_hash": prompt_hash,
        "code_eval": {
            "kind": "mbpp_assert_tests_v1",
            "entry_point": function.name,
            "prompt_source": prompt_source,
            "full_prompt_source": full_prompt_source,
            "setup_code": combined_setup,
            "tests": execution_tests,
            "visible_test_count": len(visible_tests),
            "hidden_challenge_test_count": len(execution_tests) - len(visible_tests),
            "completion_protocol": "formal_humaneval_v11_body_or_full_function",
            "execution_protocol": "isolated_python_asserts",
        },
        "used_for_training": role == "train",
    }
    output["rehearsal_row_hash"] = sha256_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return output, ""


def formal_humaneval_prompt_hashes() -> set[str]:
    if not HUMANEVAL_PATH.is_file():
        raise MbppBuildError(f"Missing formal HumanEval data: {display_path(HUMANEVAL_PATH)}")
    hashes: set[str] = set()
    with gzip.open(HUMANEVAL_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                hashes.add(sha256_text(str(row.get("prompt", "")).strip()))
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze independent MBPP train/dev data under the formal HumanEval prompt and scorer protocol."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--output-dev-select", default=str(DEFAULT_DEV_SELECT))
    parser.add_argument("--output-dev-gate", default=str(DEFAULT_DEV_GATE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--code-timeout-sec", type=float, default=5.0)
    parser.add_argument("--min-train-count", type=int, default=300)
    parser.add_argument("--min-dev-count", type=int, default=35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.code_timeout_sec <= 0 or args.min_train_count <= 0 or args.min_dev_count <= 0:
        print("Timeout and minimum counts must be positive.", file=sys.stderr)
        return 2
    input_path = resolve_path(args.input)
    train_path = resolve_path(args.output_train)
    select_path = resolve_path(args.output_dev_select)
    gate_path = resolve_path(args.output_dev_gate)
    audit_path = resolve_path(args.audit)
    if not input_path.is_file():
        print(f"Missing MBPP source: {display_path(input_path)}", file=sys.stderr)
        return 1
    input_hash = sha256_file(input_path)
    if input_hash != EXPECTED_MBPP_SHA256:
        print(
            f"MBPP source hash mismatch: expected={EXPECTED_MBPP_SHA256} actual={input_hash}",
            file=sys.stderr,
        )
        return 1
    try:
        source_rows = read_jsonl(input_path)
        formal_prompt_hashes = formal_humaneval_prompt_hashes()
    except MbppBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    by_id = {int(row.get("task_id", -1)): row for row in source_rows}
    if sorted(by_id) != list(range(1, 975)):
        print("MBPP source does not contain exactly task IDs 1-974.", file=sys.stderr)
        return 1
    rejected = Counter()
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for task_id in range(511, 975):
        role = "train" if task_id >= 601 else "development"
        built, reason = build_row(by_id[task_id], role, args.code_timeout_sec)
        if built is None:
            rejected[f"{role}:{reason}"] += 1
            continue
        if role == "train":
            train_rows.append(built)
        else:
            dev_rows.append(built)
    train_rows.sort(key=lambda row: int(row["mbpp_task_id"]))
    dev_rows.sort(key=lambda row: int(row["mbpp_task_id"]))
    select_rows = []
    gate_rows = []
    for index, row in enumerate(dev_rows):
        target_role = "dev_select" if index % 2 == 0 else "dev_gate"
        copied = dict(row)
        copied["split_role"] = target_role
        copied["source"] = f"mbpp_official_{target_role}"
        copied["sample_id"] = f"mbpp/{target_role}/{int(row['mbpp_task_id']):04d}"
        copied["rehearsal_row_hash"] = sha256_text(
            json.dumps(
                {key: value for key, value in copied.items() if key != "rehearsal_row_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        (select_rows if index % 2 == 0 else gate_rows).append(copied)

    all_output_rows = train_rows + select_rows + gate_rows
    sample_ids = [str(row["sample_id"]) for row in all_output_rows]
    duplicate_sample_count = len(sample_ids) - len(set(sample_ids))
    train_task_ids = {int(row["mbpp_task_id"]) for row in train_rows}
    select_task_ids = {int(row["mbpp_task_id"]) for row in select_rows}
    gate_task_ids = {int(row["mbpp_task_id"]) for row in gate_rows}
    split_overlap_count = sum(
        len(left & right)
        for left, right in [
            (train_task_ids, select_task_ids),
            (train_task_ids, gate_task_ids),
            (select_task_ids, gate_task_ids),
        ]
    )
    formal_prompt_overlap_count = sum(
        sha256_text(str(row["code_eval"]["full_prompt_source"]).strip()) in formal_prompt_hashes
        for row in all_output_rows
    )
    formal_id_overlap_count = sum(str(row["sample_id"]).startswith("HumanEval/") for row in all_output_rows)
    formal_test_overlap_count = formal_prompt_overlap_count + formal_id_overlap_count
    errors: list[str] = []
    if len(train_rows) < args.min_train_count:
        errors.append("insufficient_train_rows")
    if len(select_rows) < args.min_dev_count:
        errors.append("insufficient_dev_select_rows")
    if len(gate_rows) < args.min_dev_count:
        errors.append("insufficient_dev_gate_rows")
    if duplicate_sample_count:
        errors.append("duplicate_sample_ids")
    if split_overlap_count:
        errors.append("train_dev_task_overlap")
    if formal_test_overlap_count:
        errors.append("formal_humaneval_overlap")
    if any(row.get("used_for_training") is not False for row in select_rows + gate_rows):
        errors.append("dev_rows_marked_for_training")

    write_jsonl(train_path, train_rows)
    write_jsonl(select_path, select_rows)
    write_jsonl(gate_path, gate_rows)
    report = {
        "gate": "G-KD-v23-independent-code-data",
        "check_version": "1.0",
        "created_by": "model_compression/build_mbpp_v23.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "input_path": display_path(input_path),
        "input_sha256": input_hash,
        "official_split_policy": {
            "train_task_ids": "601-974",
            "development_task_ids": "511-600",
            "excluded_prompt_and_test_task_ids": "1-510",
            "development_partition": "valid single-function rows sorted by task_id, alternating select/gate",
        },
        "prompt_style": "v11",
        "completion_protocol": "formal_humaneval_v11_body_or_full_function",
        "scoring_protocol": "formal_humaneval_response_plus_mbpp_assert_execution_v1",
        "train_count": len(train_rows),
        "dev_select_count": len(select_rows),
        "dev_gate_count": len(gate_rows),
        "excluded_mbpp_task_count": 510,
        "rejection_counts": dict(sorted(rejected.items())),
        "duplicate_sample_count": duplicate_sample_count,
        "split_overlap_count": split_overlap_count,
        "formal_prompt_overlap_count": formal_prompt_overlap_count,
        "formal_id_overlap_count": formal_id_overlap_count,
        "formal_test_overlap_count": formal_test_overlap_count,
        "human_eval_test_used_for_training": False,
        "human_eval_test_used_for_selection": False,
        "human_eval_test_used_for_gate": False,
        "outputs": {
            "train": {"path": display_path(train_path), "sha256": sha256_file(train_path)},
            "dev_select": {"path": display_path(select_path), "sha256": sha256_file(select_path)},
            "dev_gate": {"path": display_path(gate_path), "sha256": sha256_file(gate_path)},
        },
        "errors": errors,
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {display_path(train_path)} rows={len(train_rows)}")
    print(f"Wrote {display_path(select_path)} rows={len(select_rows)}")
    print(f"Wrote {display_path(gate_path)} rows={len(gate_rows)}")
    print(f"Wrote {display_path(audit_path)} status={report['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
