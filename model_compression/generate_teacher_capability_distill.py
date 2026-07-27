from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_chapter2_capability import (  # noqa: E402
    HUMANEVAL_EXEC_PREAMBLE,
    extract_choice,
    extract_function_block,
    extract_gsm8k_prediction,
    extract_gsm8k_reference,
    health_status,
    humaneval_candidate_sources,
    generate_text_endpoint,
    run_assert_source_check,
)


DEFAULT_TEACHER_URL = "http://127.0.0.1:8000"
DEFAULT_TEACHER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_TRACE = ROOT / "data" / "distill" / "teacher_capability_trace_v21.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "distill" / "teacher_capability_distill_v21.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_teacher_capability_distill_v21.json"
SUPPORTED_DATASETS = {"gsm8k", "humaneval", "cmmlu"}
LEGACY_VERIFIER_VERSION = "1.0"
CODE_VERIFIER_VERSION = "1.2"


class CapabilityDistillError(RuntimeError):
    pass


def verifier_version(dataset_key: str) -> str:
    return CODE_VERIFIER_VERSION if dataset_key == "humaneval" else LEGACY_VERIFIER_VERSION


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CapabilityDistillError(
                    f"Invalid JSONL at {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CapabilityDistillError(
                    f"JSONL row must be an object at {display_path(path)}:{line_number}"
                )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_comma_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def parse_teacher_model_id_map(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in parse_comma_values(values):
        if "=" not in item:
            raise CapabilityDistillError(f"Invalid teacher model map entry: {item}")
        dataset, model_id = item.split("=", 1)
        dataset = dataset.strip()
        model_id = model_id.strip()
        if dataset not in SUPPORTED_DATASETS:
            raise CapabilityDistillError(f"Unsupported teacher model map dataset: {dataset}")
        if not model_id:
            raise CapabilityDistillError(f"Empty Teacher model ID for dataset: {dataset}")
        parsed[dataset] = model_id
    return parsed


def parse_dataset_threshold_map(
    values: list[str],
    value_type: type[int] | type[float],
    name: str,
) -> dict[str, int | float]:
    parsed: dict[str, int | float] = {}
    for item in parse_comma_values(values):
        if "=" not in item:
            raise CapabilityDistillError(f"Invalid {name} entry: {item}")
        dataset, raw_value = item.split("=", 1)
        dataset = dataset.strip()
        if dataset not in SUPPORTED_DATASETS:
            raise CapabilityDistillError(f"Unsupported dataset in {name}: {dataset}")
        try:
            parsed[dataset] = value_type(raw_value.strip())
        except ValueError as exc:
            raise CapabilityDistillError(f"Invalid {name} value: {item}") from exc
    return parsed


def input_key(row: dict[str, Any]) -> str:
    payload = {
        "dataset_key": str(row.get("dataset_key", "")),
        "messages": row.get("messages", []),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def row_hash(row: dict[str, Any]) -> str:
    payload = {
        "source": str(row.get("source", "")),
        "dataset_key": str(row.get("dataset_key", "")),
        "sample_id": str(row.get("sample_id", "")),
        "validation_group_id": str(row.get("validation_group_id", "")),
        "messages": row.get("messages", []),
        "answer": str(row.get("answer", "")),
        "code_eval": row.get("code_eval", {}),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def final_test_ids() -> set[str]:
    selected: set[str] = set()
    for path in sorted((ROOT / "data" / "splits").glob("*_test.txt")):
        selected.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return selected


def matches_final_id(sample_id: str, final_ids: set[str]) -> bool:
    return any(sample_id == final_id or sample_id.startswith(f"{final_id}/") for final_id in final_ids)


def load_source_rows(paths: list[Path], datasets: set[str]) -> tuple[list[dict[str, Any]], int]:
    final_ids = final_test_ids()
    selected: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for path in paths:
        if not path.is_file():
            raise CapabilityDistillError(f"Missing input JSONL: {display_path(path)}")
        for row in read_jsonl(path):
            if row.get("used_for_training") is not True:
                continue
            dataset_key = str(row.get("dataset_key", ""))
            if dataset_key not in datasets:
                continue
            if dataset_key not in SUPPORTED_DATASETS:
                raise CapabilityDistillError(f"Unsupported dataset: {dataset_key}")
            sample_id = str(row.get("sample_id", ""))
            messages = row.get("messages")
            answer = str(row.get("answer", "")).strip()
            if not sample_id or not isinstance(messages, list) or not messages or not answer:
                raise CapabilityDistillError(f"Incomplete source row: {sample_id or '<missing sample_id>'}")
            if matches_final_id(sample_id, final_ids):
                raise CapabilityDistillError(f"Final/test sample is forbidden in teacher distillation: {sample_id}")
            key = input_key(row)
            if key in selected:
                duplicate_count += 1
                continue
            copied = dict(row)
            copied["teacher_distill_input_key"] = key
            copied["teacher_distill_input_row_hash"] = row_hash(row)
            copied["teacher_distill_input_path"] = display_path(path)
            selected[key] = copied
    rows = list(selected.values())
    rows.sort(key=lambda row: (str(row.get("dataset_key", "")), str(row.get("sample_id", ""))))
    if not rows:
        raise CapabilityDistillError("No eligible non-final source rows selected")
    return rows, duplicate_count


def list_served_models(url: str, timeout_sec: float) -> list[str]:
    try:
        with urlopen(Request(f"{url.rstrip('/')}/v1/models", method="GET"), timeout=timeout_sec) as response:
            value = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise CapabilityDistillError(f"Cannot list Teacher models at {url}: {exc}") from exc
    model_ids = [
        str(item.get("id", ""))
        for item in value.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if not model_ids:
        raise CapabilityDistillError(f"Teacher endpoint returned no model IDs: {url}")
    return model_ids


def probe_endpoints(
    urls: list[str],
    timeout_sec: float,
    default_model: str,
    model_id_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    model_id_map = dict(model_id_map or {})
    required_models = {default_model, *model_id_map.values()}
    endpoints: list[dict[str, Any]] = []
    for url in urls:
        status = health_status(url, timeout_sec)
        if status != 200:
            raise CapabilityDistillError(f"Teacher endpoint health check failed: {url} status={status}")
        available_models = list_served_models(url, timeout_sec)
        missing = sorted(required_models - set(available_models))
        if missing:
            raise CapabilityDistillError(
                f"Teacher endpoint {url} is missing routed models {missing}; available={available_models}"
            )
        endpoints.append(
            {
                "teacher_url": url,
                "health_status": status,
                "teacher_model_id": default_model,
                "served_model_id": default_model,
                "served_model_id_map": model_id_map,
                "available_model_ids": available_models,
            }
        )
    return endpoints


def route_endpoint(endpoint: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    routed = dict(endpoint)
    model_id_map = endpoint.get("served_model_id_map", {})
    if not isinstance(model_id_map, dict):
        raise CapabilityDistillError("Endpoint served_model_id_map must be an object")
    model_id = str(model_id_map.get(dataset_key, endpoint["served_model_id"]))
    routed["teacher_model_id"] = model_id
    routed["served_model_id"] = model_id
    return routed


def call_teacher(
    endpoint: dict[str, Any],
    messages: list[dict[str, str]],
    timeout_sec: float,
    max_new_tokens: int,
    retry_count: int,
    retry_sleep_sec: float,
) -> tuple[str, float]:
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            return generate_text_endpoint(
                str(endpoint["teacher_url"]),
                str(endpoint["served_model_id"]),
                messages,
                timeout_sec,
                max_new_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_sleep_sec)
    assert last_error is not None
    raise last_error


def extract_prompt_function(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )
    matches = list(re.finditer(r"(?m)^def\s+([A-Za-z_]\w*)\s*\(", user_text))
    if not matches:
        raise CapabilityDistillError("Code prompt does not contain a top-level function definition")
    entry_point = matches[-1].group(1)
    prompt_source = extract_function_block(user_text[matches[-1].start() :], entry_point)
    if not prompt_source:
        raise CapabilityDistillError(f"Could not extract code prompt for {entry_point}")
    return entry_point, prompt_source


def code_source_candidates(prompt_source: str, entry_point: str, completion: str) -> list[str]:
    candidates: list[str] = []
    for candidate in humaneval_candidate_sources(prompt_source, entry_point, completion):
        if not candidate or candidate in candidates:
            continue
        try:
            module = ast.parse(candidate)
        except (IndentationError, SyntaxError):
            continue
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == entry_point]
        if len(functions) == 1:
            candidates.append(candidate)
    return candidates


FORBIDDEN_CODE_NODES = (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)
FORBIDDEN_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
SAFE_IMPORT_MODULES = {
    "bisect",
    "collections",
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
}


def safe_code_ast(source: str) -> bool:
    try:
        module = ast.parse(source)
    except (IndentationError, SyntaxError):
        return False
    for node in ast.walk(module):
        if isinstance(node, FORBIDDEN_CODE_NODES):
            return False
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in SAFE_IMPORT_MODULES for alias in node.names):
                return False
        if isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".", 1)[0]
            if node.level != 0 or module_name not in SAFE_IMPORT_MODULES:
                return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
    return True


EQUIVALENCE_CHECK = r'''
import copy
import inspect
import itertools
import math

TASK_NAME = VALIDATION_GROUP_ID.rsplit("/", 1)[-1]
EDGE_CASES = [
    [],
    [("a", "b")],
    [("a", "b"), ("b", "c")],
    [("a", "b"), ("c", "d")],
]
INT_PAIR_CASES = [[], [(1, 2)], [(2, 1), (1, 2), (1, 1)]]
INTERVAL_CASES = [[], [[1, 2]], [[1, 2], [3, 4]], [[-1, 0], [2, 3], [5, 8]]]
SORTED_UNIQUE_CASES = [[], [1], [-1, 1, 2, 4], [0, 3, 5, 9]]
SORTED_WITH_DUPLICATES_CASES = [[], [1], [-1, 1, 2, 2], [0, 3, 5, 9]]
MATRIX_CASES = [[], [[1]], [[1, 2], [3, 4]], [[-1, 0, 2], [3, 5, 8]]]


def values_for(parameter, position):
    annotation = str(parameter.annotation).lower()
    name = parameter.name.lower()
    if TASK_NAME == "evaluate_rpn" and position == 0:
        return [["2", "3", "+"], ["4", "2", "/"], ["5", "1", "2", "+", "4", "*", "+", "3", "-"]]
    if TASK_NAME == "balanced_parentheses" and position == 0:
        return ["", "()", "(())", "()()", "(()", ")("]
    if TASK_NAME == "valid_parentheses" and position == 0:
        return ["", "()", "([])", "{[]}", "([)]", "(("]
    if TASK_NAME == "roman_to_int" and position == 0:
        return ["I", "III", "IV", "IX", "LVIII", "MCMXCIV"]
    if TASK_NAME in {"topological_layers", "shortest_path_length"} and position == 0:
        return EDGE_CASES
    if TASK_NAME == "tree_levels_from_edges" and position == 1:
        return EDGE_CASES
    if TASK_NAME == "connected_components":
        return [[], ["a"], ["a", "b", "c"], ["a", "b", "c", "d"]] if position == 0 else EDGE_CASES
    if TASK_NAME == "rle_decode" and position == 0:
        return [[], [("a", 1)], [("a", 2), ("b", 1)], [("x", 0), ("y", 3)]]
    if TASK_NAME == "sort_by_second_then_first" and position == 0:
        return INT_PAIR_CASES
    if TASK_NAME in {"merge_intervals", "interval_intersection"}:
        return INTERVAL_CASES
    if TASK_NAME in {"spiral_order", "rotate_matrix_clockwise"} and position == 0:
        return MATRIX_CASES
    if TASK_NAME == "compress_ranges" and "list" in annotation:
        return SORTED_UNIQUE_CASES
    if TASK_NAME in {"binary_search", "merge_sorted_lists"} and "list" in annotation:
        return SORTED_WITH_DUPLICATES_CASES
    if TASK_NAME == "min_coins" and position == 0:
        return [[], [1], [1, 2, 5], [2, 3, 7]]
    if "bool" in annotation:
        return [False, True]
    if "int" in annotation and "list" not in annotation and "dict" not in annotation and "tuple" not in annotation:
        if TASK_NAME in {"factorial", "pascal_row"}:
            return [0, 1, 2, 3, 5, 10]
        if TASK_NAME in {"chunk", "window_max", "sliding_average"} and position == 1:
            return [1, 2, 3, 5]
        if TASK_NAME in {"top_k", "k_closest_values"} and position == len(inspect.signature(reference_fn).parameters) - 1:
            return [0, 1, 2, 3, 5]
        if TASK_NAME == "min_coins" and position == 1:
            return [0, 1, 2, 3, 5, 10]
        return [0, 1, 2, 3, 5, -1, 10]
    if "float" in annotation and "list" not in annotation and "dict" not in annotation and "tuple" not in annotation:
        return [0.0, 0.5, 1.0, -1.5, 3.25]
    if "str" in annotation and "list" not in annotation and "dict" not in annotation and "tuple" not in annotation:
        return ["", "a", "abba", "hello world", "()()", "a b a"]
    if "dict" in annotation:
        return [{}, {"a": 1}, {"a": 1, "b": 2}]
    if "list" in annotation or name in {"values", "items", "numbers", "words", "matrix", "pairs"}:
        if "tuple[str" in annotation:
            return EDGE_CASES
        if "tuple[int" in annotation or name == "pairs":
            return INT_PAIR_CASES
        if "str" in annotation or name == "words":
            return [[], ["a"], ["a", "bb", "a"], ["car", "cat", "dog"]]
        if "list[list" in annotation or name == "matrix":
            return MATRIX_CASES
        if "float" in annotation:
            return [[], [0.0], [1.0, 2.0, -1.0], [0.5, 3.25, -1.5, 2.0]]
        return [[], [1], [1, 2, 2, -1], [5, 3, 9, 0]]
    if "tuple" in annotation or "pair" in name:
        return [(), (1,), (1, 2), ((1, 2), (2, 1))]
    if "str" in annotation or any(token in name for token in ("text", "word", "prefix", "suffix")):
        return ["", "a", "abba", "hello world", "()()", "a b a"]
    return [0, 1, 2, 3, 5, -1, 10]

def valid_args(args):
    if TASK_NAME == "clamp":
        return args[1] <= args[2]
    if TASK_NAME == "connected_components":
        nodes, edges = args
        node_set = set(nodes)
        return all(left in node_set and right in node_set for left, right in edges)
    if TASK_NAME == "median_sorted":
        return bool(args[0])
    return True


def normalized_nested_groups(value):
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return sorted(tuple(sorted(group)) for group in value)
    except (TypeError, ValueError):
        return None


def same(left, right, args):
    if TASK_NAME == "binary_search":
        values, target = args
        if left == -1:
            return right == -1
        return (
            isinstance(right, int)
            and not isinstance(right, bool)
            and 0 <= right < len(values)
            and values[right] == target
        )
    if TASK_NAME == "two_sum_indices":
        if left is None:
            return right is None
        if not isinstance(right, (list, tuple)) or len(right) != 2:
            return False
        first, second = right
        values, target = args
        return (
            isinstance(first, int)
            and isinstance(second, int)
            and first != second
            and 0 <= first < len(values)
            and 0 <= second < len(values)
            and values[first] + values[second] == target
        )
    if TASK_NAME in {"connected_components", "group_anagrams"}:
        return normalized_nested_groups(left) == normalized_nested_groups(right)
    if TASK_NAME == "topological_layers":
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return [sorted(layer) for layer in left] == [sorted(layer) for layer in right]
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return left == right

safe_names = {
    name: globals()[name]
    for name in (
        "Any", "Callable", "Counter", "DefaultDict", "Deque", "Dict", "Iterable", "Iterator",
        "List", "Optional", "Sequence", "Set", "Tuple", "Union", "bisect", "collections",
        "decimal", "fractions", "functools", "heapq", "itertools", "math", "operator", "re",
        "statistics", "string"
    )
    if name in globals()
}
reference_ns = dict(safe_names)
candidate_ns = dict(safe_names)
exec(REFERENCE_SOURCE, reference_ns)
exec(CANDIDATE_SOURCE, candidate_ns)
reference_fn = reference_ns[ENTRY_POINT]
candidate_fn = candidate_ns[ENTRY_POINT]
signature = inspect.signature(reference_fn)
parameters = list(signature.parameters.values())
if any(item.kind not in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD) for item in parameters):
    raise AssertionError("unsupported signature")
pools = [values_for(item, index) for index, item in enumerate(parameters)]
successful = 0
for args in itertools.product(*pools):
    if not valid_args(args):
        continue
    try:
        expected = reference_fn(*copy.deepcopy(args))
    except Exception:
        continue
    try:
        actual = candidate_fn(*copy.deepcopy(args))
    except Exception as exc:
        raise AssertionError(f"candidate raised {type(exc).__name__} for args={args!r}") from exc
    if not same(expected, actual, args):
        raise AssertionError(f"mismatch for args={args!r}: expected={expected!r} actual={actual!r}")
    successful += 1
    if successful >= 96:
        break
if successful == 0:
    raise AssertionError("no valid differential cases")
'''


def run_code_equivalence(
    reference_source: str,
    candidate_source: str,
    entry_point: str,
    validation_group_id: str,
    timeout_sec: float,
) -> tuple[bool, str]:
    if not safe_code_ast(reference_source) or not safe_code_ast(candidate_source):
        return False, "unsafe_or_invalid_ast"
    script = (
        HUMANEVAL_EXEC_PREAMBLE
        + "\nREFERENCE_SOURCE = "
        + repr(reference_source)
        + "\nCANDIDATE_SOURCE = "
        + repr(candidate_source)
        + "\nENTRY_POINT = "
        + repr(entry_point)
        + "\nVALIDATION_GROUP_ID = "
        + repr(validation_group_id)
        + "\n"
        + EQUIVALENCE_CHECK
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "code_equivalence_timeout"
    if completed.returncode == 0:
        return True, "differential_equivalence_passed"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1][:240] if detail else f"returncode={completed.returncode}"


def normalize_code_body(source: str, entry_point: str) -> str:
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == entry_point
    )
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    normalized = "\n".join(ast.unparse(node) for node in body).strip()
    if not normalized:
        raise CapabilityDistillError("Teacher code has an empty function body")
    return textwrap.indent(normalized, "    ")


def validate_code_row(row: dict[str, Any], response: str, timeout_sec: float) -> tuple[bool, str, str]:
    entry_point, prompt_source = extract_prompt_function(row["messages"])
    reference_candidates = code_source_candidates(prompt_source, entry_point, str(row["answer"]))
    teacher_candidates = code_source_candidates(prompt_source, entry_point, response)
    if not reference_candidates:
        raise CapabilityDistillError(f"Canonical code target is invalid for {row['sample_id']}")
    if not teacher_candidates:
        return False, "teacher_code_not_parseable", ""
    code_eval = row.get("code_eval")
    if isinstance(code_eval, dict) and code_eval:
        test_kind = str(code_eval.get("kind", ""))
        supported_test_kinds = {
            "mbpp_assert_tests_v1",
            "apps_call_tests_v1",
            "code_contests_io_tests_v1",
        }
        if test_kind not in supported_test_kinds:
            raise CapabilityDistillError(
                f"Unsupported code_eval kind for {row['sample_id']}: {test_kind}"
            )
        declared_entry_point = str(code_eval.get("entry_point", ""))
        declared_prompt_source = str(code_eval.get("prompt_source", ""))
        tests = code_eval.get("tests", [])
        setup_code = str(code_eval.get("setup_code", ""))
        if declared_entry_point != entry_point or declared_prompt_source != prompt_source:
            raise CapabilityDistillError(f"code_eval prompt metadata mismatch for {row['sample_id']}")
        if not isinstance(tests, list) or not tests:
            raise CapabilityDistillError(f"code_eval tests are missing for {row['sample_id']}")
        reference_passed = any(
            run_assert_source_check(reference_source, tests, timeout_sec, setup_code)[0]
            for reference_source in reference_candidates
        )
        if not reference_passed:
            raise CapabilityDistillError(
                f"Canonical code target fails {test_kind} assertions for {row['sample_id']}"
            )
        first_detail = ""
        for teacher_source in teacher_candidates:
            passed, detail = run_assert_source_check(
                teacher_source,
                tests,
                timeout_sec,
                setup_code,
            )
            if passed:
                return True, f"{test_kind}_passed", normalize_code_body(teacher_source, entry_point)
            if not first_detail:
                first_detail = detail
        return False, first_detail or f"teacher_code_failed_{test_kind}", ""
    first_detail = ""
    validation_group_id = str(row.get("validation_group_id", row["sample_id"]))
    for reference_source, teacher_source in itertools.product(reference_candidates, teacher_candidates):
        passed, detail = run_code_equivalence(
            reference_source,
            teacher_source,
            entry_point,
            validation_group_id,
            timeout_sec,
        )
        if passed:
            return True, detail, normalize_code_body(teacher_source, entry_point)
        if not first_detail:
            first_detail = detail
    return False, first_detail or "teacher_code_failed_equivalence", ""


def validate_response(row: dict[str, Any], response: str, code_timeout_sec: float) -> tuple[bool, str, str]:
    dataset_key = str(row["dataset_key"])
    canonical_answer = str(row["answer"])
    if dataset_key == "gsm8k":
        expected = extract_gsm8k_reference(canonical_answer)
        predicted = extract_gsm8k_prediction(response)
        if not expected or predicted != expected:
            return False, f"numeric_mismatch expected={expected} predicted={predicted}", ""
        reasoning = response.rsplit("####", 1)[0].strip() if "####" in response else response.strip()
        normalized = f"{reasoning}\n#### {expected}" if reasoning else f"#### {expected}"
        return True, "exact_numeric_match", normalized
    if dataset_key == "cmmlu":
        expected = extract_choice(canonical_answer) or canonical_answer.strip().upper()[:1]
        predicted = extract_choice(response)
        if expected not in {"A", "B", "C", "D"} or predicted != expected:
            return False, f"choice_mismatch expected={expected} predicted={predicted}", ""
        return True, "choice_match", expected
    if dataset_key == "humaneval":
        return validate_code_row(row, response, code_timeout_sec)
    raise CapabilityDistillError(f"Unsupported dataset: {dataset_key}")


def load_prior_traces(path: Path, source_by_key: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise CapabilityDistillError(f"Missing repair parent trace: {display_path(path)}")
    selected: dict[str, dict[str, Any]] = {}
    for trace in read_jsonl(path):
        key = str(trace.get("input_key", ""))
        source_row = source_by_key.get(key)
        if source_row is None:
            continue
        if trace.get("input_row_hash") != source_row.get("teacher_distill_input_row_hash"):
            raise CapabilityDistillError(
                f"Repair parent trace/source hash mismatch for {source_row['sample_id']}"
            )
        selected[key] = trace
    missing = [row["sample_id"] for key, row in source_by_key.items() if key not in selected]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise CapabilityDistillError(
            f"Repair parent trace is incomplete: missing={len(missing)} examples={preview}"
        )
    return selected


def function_contract(messages: list[dict[str, Any]]) -> str:
    try:
        _, prompt_source = extract_prompt_function(messages)
        module = ast.parse(prompt_source)
    except (CapabilityDistillError, IndentationError, SyntaxError):
        return ""
    function = next((node for node in module.body if isinstance(node, ast.FunctionDef)), None)
    return ast.get_docstring(function, clean=True) if function is not None else ""


CONTRACT_CLARIFICATIONS = {
    "binary_search": "Any valid index containing the target is correct when duplicate values exist.",
    "common_items": "Return each shared value exactly once, in sorted order.",
    "compress_ranges": "Input values are sorted and unique.",
    "connected_components": "Every edge endpoint belongs to the supplied node list.",
    "interval_intersection": "Each input list contains closed intervals that are internally disjoint.",
    "median_sorted": "The input may be unsorted; compute the statistical median after ordering it.",
}


def canonical_contracts(source_rows: list[dict[str, Any]]) -> dict[str, str]:
    contracts: dict[str, str] = {}
    ordered = sorted(
        source_rows,
        key=lambda row: (not str(row.get("sample_id", "")).endswith("/r0"), str(row.get("sample_id", ""))),
    )
    for row in ordered:
        group = str(row.get("validation_group_id", row["sample_id"]))
        contract = function_contract(row["messages"])
        if contract and group not in contracts:
            task_name = group.rsplit("/", 1)[-1]
            clarification = CONTRACT_CLARIFICATIONS.get(task_name, "")
            contracts[group] = f"{contract} {clarification}".strip()
    return contracts


def build_repair_messages(
    source_row: dict[str, Any],
    previous_response: str,
    verification: str,
    canonical_contract: str,
) -> list[dict[str, str]]:
    messages = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in source_row["messages"]
        if isinstance(message, dict) and message.get("role") in {"system", "user"}
    ]
    contract_text = canonical_contract or "Follow the current function docstring exactly."
    feedback = (
        "The previous implementation failed executable verification. Repair it.\n"
        f"Canonical behavior contract: {contract_text}\n"
        f"Verification failure: {verification}\n"
        "The function signature in the original prompt is authoritative. Preserve duplicate values and "
        "the exact annotated return container type when required. Handle empty and boundary inputs that "
        "are valid under the contract. Use only built-ins or safe Python standard-library modules. "
        "Return only valid Python code with no markdown or explanation."
    )
    messages.append({"role": "assistant", "content": previous_response})
    messages.append({"role": "user", "content": feedback})
    return messages


def build_trace_row(
    row: dict[str, Any],
    response: str,
    latency_ms: float,
    endpoint: dict[str, Any],
    accepted: bool,
    verification: str,
    normalized_answer: str,
    created_ts: str,
    dry_run: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = {
        "trace_version": "21.0",
        "verifier_version": verifier_version(str(row["dataset_key"])),
        "created_by": "model_compression/generate_teacher_capability_distill.py",
        "created_ts": created_ts,
        "dataset_key": row["dataset_key"],
        "sample_id": row["sample_id"],
        "validation_group_id": row.get("validation_group_id", row["sample_id"]),
        "input_key": row["teacher_distill_input_key"],
        "input_row_hash": row["teacher_distill_input_row_hash"],
        "input_path": row["teacher_distill_input_path"],
        "source": row.get("source", ""),
        "teacher_model_id": endpoint.get("teacher_model_id", DEFAULT_TEACHER_MODEL_ID),
        "served_model_id": endpoint["served_model_id"],
        "teacher_url": endpoint["teacher_url"],
        "messages_hash": sha256_text(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)),
        "teacher_response": response,
        "teacher_response_hash": sha256_text(response),
        "accepted_for_training": bool(accepted and not dry_run),
        "verification": verification,
        "normalized_answer": normalized_answer if accepted else "",
        "latency_ms": latency_ms,
        "dry_run": dry_run,
    }
    if extra:
        trace.update(extra)
    trace["trace_row_hash"] = sha256_text(json.dumps(trace, ensure_ascii=False, sort_keys=True))
    return trace


def build_distill_row(
    source_row: dict[str, Any],
    trace: dict[str, Any],
    rehearsal_version: str | None = None,
) -> dict[str, Any]:
    repair_mode = trace.get("repair_mode") is True
    code_verifier = str(trace.get("verifier_version", ""))
    repaired_version = "21.2-teacher-repaired" if code_verifier == "1.2" else "21.1-teacher-repaired"
    row = {
        "rehearsal_version": rehearsal_version or (
            repaired_version if repair_mode else "21.0-teacher-filtered"
        ),
        "created_by": "model_compression/generate_teacher_capability_distill.py",
        "created_ts": trace["created_ts"],
        "source": (
            f"teacher_capability_repair_{trace['dataset_key']}"
            if repair_mode
            else f"teacher_capability_distill_{trace['dataset_key']}"
        ),
        "dataset_key": trace["dataset_key"],
        "sample_id": trace["sample_id"],
        "validation_group_id": trace["validation_group_id"],
        "messages": source_row["messages"],
        "answer": trace["normalized_answer"],
        "teacher_model_id": trace["teacher_model_id"],
        "served_model_id": trace["served_model_id"],
        "teacher_trace_row_hash": trace["trace_row_hash"],
        "teacher_verification": trace["verification"],
        "used_for_training": True,
    }
    if isinstance(source_row.get("code_eval"), dict):
        row["code_eval"] = source_row["code_eval"]
    row["rehearsal_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def default_partial_audit_path(audit_path: Path) -> Path:
    return audit_path.with_name(f"{audit_path.stem}.partial{audit_path.suffix or '.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate correctness-filtered non-final 14B capability supervision."
    )
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--output-distill", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--partial-audit", default=None)
    parser.add_argument("--teacher-url", action="append", default=[])
    parser.add_argument("--teacher-model-id", default=DEFAULT_TEACHER_MODEL_ID)
    parser.add_argument(
        "--teacher-model-id-map",
        action="append",
        default=[],
        help="Dataset-specific served Teacher IDs as dataset=model_id; repeat or comma-separate entries.",
    )
    parser.add_argument("--trace-version", default=None)
    parser.add_argument("--rehearsal-version", default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--code-timeout-sec", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--min-accept-rate", type=float, default=0.5)
    parser.add_argument("--min-group-coverage-rate", type=float, default=0.0)
    parser.add_argument("--min-selected-count-map", action="append", default=[])
    parser.add_argument("--min-accepted-count-map", action="append", default=[])
    parser.add_argument("--min-accept-rate-map", action="append", default=[])
    parser.add_argument("--repair-from-trace", default=None)
    parser.add_argument("--repair-rounds", type=int, default=0)
    parser.add_argument("--retry-rejected-on-resume", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Recompute outputs and gate status from a complete resumed trace without calling Teacher.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs using canonical answers; emitted rows are never marked for training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.max_new_tokens <= 0 or args.timeout_sec <= 0 or args.code_timeout_sec <= 0:
        print("Worker, token, and timeout values must be positive.", file=sys.stderr)
        return 2
    if (
        args.retry_count < 0
        or args.checkpoint_interval < 0
        or args.repair_rounds < 0
        or not 0 <= args.min_accept_rate <= 1
        or not 0 <= args.min_group_coverage_rate <= 1
    ):
        print("Retry/checkpoint values and min accept rate are invalid.", file=sys.stderr)
        return 2
    if bool(args.repair_from_trace) != bool(args.repair_rounds > 0):
        print("Repair mode requires both --repair-from-trace and --repair-rounds > 0.", file=sys.stderr)
        return 2
    if args.audit_only and (not args.resume or args.retry_rejected_on_resume):
        print("Audit-only mode requires --resume and forbids --retry-rejected-on-resume.", file=sys.stderr)
        return 2

    input_paths = [resolve_path(value) for value in parse_comma_values(args.input_jsonl)]
    datasets = set(parse_comma_values(args.dataset)) or set(SUPPORTED_DATASETS)
    unknown = datasets - SUPPORTED_DATASETS
    if unknown:
        print(f"Unsupported datasets: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    source_rows, duplicate_count = load_source_rows(input_paths, datasets)
    try:
        teacher_model_id_map = parse_teacher_model_id_map(args.teacher_model_id_map)
        min_selected_counts = parse_dataset_threshold_map(
            args.min_selected_count_map, int, "--min-selected-count-map"
        )
        min_accepted_counts = parse_dataset_threshold_map(
            args.min_accepted_count_map, int, "--min-accepted-count-map"
        )
        min_accept_rates = parse_dataset_threshold_map(
            args.min_accept_rate_map, float, "--min-accept-rate-map"
        )
    except CapabilityDistillError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if (
        any(int(value) < 0 for value in (*min_selected_counts.values(), *min_accepted_counts.values()))
        or any(not 0 <= float(value) <= 1 for value in min_accept_rates.values())
    ):
        print("Per-task count/rate thresholds are invalid.", file=sys.stderr)
        return 2
    source_by_key = {str(row["teacher_distill_input_key"]): row for row in source_rows}
    repair_mode = bool(args.repair_from_trace)
    if repair_mode and datasets != {"humaneval"}:
        print("Repair mode currently supports --dataset humaneval only.", file=sys.stderr)
        return 2
    repair_parent_path = resolve_path(args.repair_from_trace) if args.repair_from_trace else None
    repair_parent_by_key = (
        load_prior_traces(repair_parent_path, source_by_key) if repair_parent_path is not None else {}
    )
    contract_by_group = canonical_contracts(source_rows) if repair_mode else {}

    trace_path = resolve_path(args.output_trace)
    output_path = resolve_path(args.output_distill)
    audit_path = resolve_path(args.audit)
    partial_audit_path = resolve_path(args.partial_audit) if args.partial_audit else default_partial_audit_path(audit_path)
    teacher_urls = parse_comma_values(args.teacher_url) or [DEFAULT_TEACHER_URL]
    teacher_urls = list(dict.fromkeys(url.rstrip("/") for url in teacher_urls))
    if args.audit_only:
        endpoints: list[dict[str, Any]] = []
    elif args.dry_run:
        endpoints = [
            {
                "teacher_url": "dry-run",
                "health_status": None,
                "teacher_model_id": args.teacher_model_id,
                "served_model_id": args.teacher_model_id,
                "served_model_id_map": teacher_model_id_map,
            }
        ]
    else:
        endpoints = probe_endpoints(
            teacher_urls,
            args.timeout_sec,
            args.teacher_model_id,
            teacher_model_id_map,
        )

    existing_by_key: dict[str, dict[str, Any]] = {}
    retry_seed_by_key: dict[str, dict[str, Any]] = {}
    if args.resume and trace_path.is_file():
        for trace in read_jsonl(trace_path):
            key = str(trace.get("input_key", ""))
            source_row = source_by_key.get(key)
            expected_teacher_model_id = (
                teacher_model_id_map.get(str(source_row.get("dataset_key", "")), args.teacher_model_id)
                if source_row is not None
                else args.teacher_model_id
            )
            if (
                source_row is not None
                and trace.get("input_row_hash") == source_row.get("teacher_distill_input_row_hash")
                and trace.get("dry_run") == bool(args.dry_run)
                and trace.get("teacher_model_id") == expected_teacher_model_id
                and trace.get("verifier_version") == verifier_version(str(source_row["dataset_key"]))
            ):
                if args.retry_rejected_on_resume and trace.get("accepted_for_training") is not True:
                    retry_seed_by_key[key] = trace
                else:
                    existing_by_key[key] = trace
    resumed_count = len(existing_by_key)
    tasks = [row for row in source_rows if row["teacher_distill_input_key"] not in existing_by_key]
    if args.audit_only:
        if tasks:
            print(
                f"Audit-only trace is incomplete: resumed={resumed_count} missing={len(tasks)}.",
                file=sys.stderr,
            )
            return 2
        endpoint_by_url: dict[str, dict[str, Any]] = {}
        for trace in existing_by_key.values():
            url = str(trace.get("teacher_url", ""))
            if not url:
                continue
            endpoint_by_url[url] = {
                "teacher_url": url,
                "health_status": None,
                "teacher_model_id": str(trace.get("teacher_model_id", args.teacher_model_id)),
                "served_model_id": str(trace.get("served_model_id", args.teacher_model_id)),
                "served_model_id_map": {},
            }
        endpoints = [endpoint_by_url[url] for url in sorted(endpoint_by_url)]
    created_ts = datetime.now(timezone.utc).isoformat()
    trace_by_key = dict(existing_by_key)
    errors: list[dict[str, str]] = []
    completed = 0

    def generate_one(index: int, source_row: dict[str, Any]) -> dict[str, Any]:
        endpoint = route_endpoint(
            endpoints[index % len(endpoints)], str(source_row["dataset_key"])
        )
        if repair_mode:
            key = str(source_row["teacher_distill_input_key"])
            parent_trace = retry_seed_by_key.get(key, repair_parent_by_key[key])
            parent_response = str(parent_trace.get("teacher_response", ""))
            if not parent_response:
                raise CapabilityDistillError(
                    f"Repair parent response is empty for {source_row['sample_id']}"
                )
            accepted, verification, normalized = validate_response(
                source_row,
                parent_response,
                args.code_timeout_sec,
            )
            revalidated_parent_verification = verification
            response = parent_response
            total_latency_ms = 0.0
            final_endpoint = {
                "teacher_url": str(parent_trace.get("teacher_url", "repair-parent")),
                "health_status": None,
                "teacher_model_id": str(parent_trace.get("teacher_model_id", args.teacher_model_id)),
                "served_model_id": str(parent_trace.get("served_model_id", args.teacher_model_id)),
            }
            attempts = list(parent_trace.get("repair_attempts", []))
            initial_attempt_count = len(attempts)
            group = str(source_row.get("validation_group_id", source_row["sample_id"]))
            contract = contract_by_group.get(group, "")
            for repair_offset in range(args.repair_rounds):
                if accepted:
                    break
                final_endpoint = route_endpoint(
                    endpoints[(index + repair_offset) % len(endpoints)],
                    str(source_row["dataset_key"]),
                )
                repair_messages = build_repair_messages(
                    source_row,
                    response,
                    verification,
                    contract,
                )
                if args.dry_run:
                    response, latency_ms = str(source_row["answer"]), 0.0
                else:
                    response, latency_ms = call_teacher(
                        final_endpoint,
                        repair_messages,
                        args.timeout_sec,
                        args.max_new_tokens,
                        args.retry_count,
                        args.retry_sleep_sec,
                    )
                total_latency_ms += latency_ms
                accepted, verification, normalized = validate_response(
                    source_row,
                    response,
                    args.code_timeout_sec,
                )
                attempts.append(
                    {
                        "repair_round": len(attempts) + 1,
                        "teacher_url": final_endpoint["teacher_url"],
                        "messages_hash": sha256_text(
                            json.dumps(repair_messages, ensure_ascii=False, sort_keys=True)
                        ),
                        "teacher_response_hash": sha256_text(response),
                        "verification": verification,
                        "accepted": bool(accepted and not args.dry_run),
                        "latency_ms": latency_ms,
                    }
                )
            new_attempt_count = len(attempts) - initial_attempt_count
            outcome = (
                "parent_revalidated"
                if accepted and new_attempt_count == 0
                else "repaired"
                if accepted
                else "repair_exhausted"
            )
            return build_trace_row(
                source_row,
                response,
                total_latency_ms,
                final_endpoint,
                accepted,
                verification,
                normalized,
                created_ts,
                args.dry_run,
                {
                    "trace_version": args.trace_version or "21.1",
                    "repair_mode": True,
                    "repair_outcome": outcome,
                    "repair_parent_trace_hash": parent_trace.get("trace_row_hash", ""),
                    "repair_parent_verifier_version": parent_trace.get("verifier_version", ""),
                    "repair_parent_accepted": parent_trace.get("accepted_for_training") is True,
                    "repair_parent_verification": parent_trace.get("verification", ""),
                    "repair_revalidated_verification": revalidated_parent_verification,
                    "canonical_contract_hash": sha256_text(contract),
                    "repair_attempt_count": len(attempts),
                    "repair_attempts": attempts,
                },
            )
        if args.dry_run:
            response, latency_ms = str(source_row["answer"]), 0.0
        else:
            response, latency_ms = call_teacher(
                endpoint,
                source_row["messages"],
                args.timeout_sec,
                args.max_new_tokens,
                args.retry_count,
                args.retry_sleep_sec,
            )
        accepted, verification, normalized = validate_response(source_row, response, args.code_timeout_sec)
        return build_trace_row(
            source_row,
            response,
            latency_ms,
            endpoint,
            accepted,
            verification,
            normalized,
            created_ts,
            args.dry_run,
            {"trace_version": args.trace_version} if args.trace_version else None,
        )

    def current_traces() -> list[dict[str, Any]]:
        return [trace_by_key[row["teacher_distill_input_key"]] for row in source_rows if row["teacher_distill_input_key"] in trace_by_key]

    def task_gate_snapshot(traces: list[dict[str, Any]]) -> dict[str, Any]:
        selected_counts = Counter(str(row["dataset_key"]) for row in source_rows)
        accepted_counts = Counter(
            str(trace["dataset_key"])
            for trace in traces
            if trace.get("accepted_for_training") is True
        )
        accept_rates = {
            task: accepted_counts[task] / selected_counts[task] if selected_counts[task] else 0.0
            for task in sorted(datasets)
        }
        failures: dict[str, dict[str, dict[str, int | float]]] = {}
        for task in sorted(datasets):
            task_failures: dict[str, dict[str, int | float]] = {}
            if selected_counts[task] < int(min_selected_counts.get(task, 0)):
                task_failures["selected_unique_count"] = {
                    "actual": selected_counts[task],
                    "required": int(min_selected_counts[task]),
                }
            if accepted_counts[task] < int(min_accepted_counts.get(task, 0)):
                task_failures["accepted_unique_count"] = {
                    "actual": accepted_counts[task],
                    "required": int(min_accepted_counts[task]),
                }
            if accept_rates[task] < float(min_accept_rates.get(task, 0.0)):
                task_failures["accept_rate"] = {
                    "actual": accept_rates[task],
                    "required": float(min_accept_rates[task]),
                }
            if task_failures:
                failures[task] = task_failures
        return {
            "selected_unique_prompt_counts": dict(sorted(selected_counts.items())),
            "accepted_unique_prompt_counts": dict(sorted(accepted_counts.items())),
            "accept_rate_by_dataset": accept_rates,
            "min_selected_count_by_dataset": dict(sorted(min_selected_counts.items())),
            "min_accepted_count_by_dataset": dict(sorted(min_accepted_counts.items())),
            "min_accept_rate_by_dataset": dict(sorted(min_accept_rates.items())),
            "task_gate_failures": failures,
        }

    def write_outputs(target_audit: Path, status: str, is_partial: bool) -> dict[str, Any]:
        traces = current_traces()
        task_gate = task_gate_snapshot(traces)
        distill_rows = [
            build_distill_row(
                source_by_key[str(trace["input_key"])],
                trace,
                args.rehearsal_version,
            )
            for trace in traces
            if trace.get("accepted_for_training") is True
        ]
        write_jsonl(trace_path, traces)
        write_jsonl(output_path, distill_rows)
        accepted_counts = Counter(str(row["dataset_key"]) for row in distill_rows)
        trace_counts = Counter(str(row["dataset_key"]) for row in traces)
        rejection_counts = Counter(
            str(trace.get("verification", ""))
            for trace in traces
            if trace.get("accepted_for_training") is not True
        )
        endpoint_counts = Counter(str(trace.get("teacher_url", "")) for trace in traces)
        accepted_count = len(distill_rows)
        completed_count = len(traces)
        accept_rate = accepted_count / completed_count if completed_count else 0.0
        selected_groups = {
            str(row.get("validation_group_id", row["sample_id"])) for row in source_rows
        }
        accepted_groups = {
            str(trace.get("validation_group_id", trace["sample_id"]))
            for trace in traces
            if trace.get("accepted_for_training") is True
        }
        group_coverage_rate = len(accepted_groups) / len(selected_groups) if selected_groups else 0.0
        fully_rejected_groups = sorted(selected_groups - accepted_groups)
        repair_attempt_total = sum(int(trace.get("repair_attempt_count", 0)) for trace in traces)
        repaired_accept_count = sum(
            1
            for trace in traces
            if trace.get("accepted_for_training") is True
            and trace.get("repair_outcome") == "repaired"
        )
        parent_revalidated_accept_count = sum(
            1
            for trace in traces
            if trace.get("accepted_for_training") is True
            and trace.get("repair_outcome") == "parent_revalidated"
        )
        trace_gate_version = str(args.trace_version or "21").removeprefix("v")
        audit = {
            "gate": f"G-KD-TRACE-teacher-capability-distill-v{trace_gate_version}",
            "check_version": (
                verifier_version(next(iter(datasets))) if len(datasets) == 1 else "dataset-specific"
            ),
            "verifier_version": (
                verifier_version(next(iter(datasets))) if len(datasets) == 1 else "dataset-specific"
            ),
            "verifier_versions": {
                dataset_key: verifier_version(dataset_key) for dataset_key in sorted(datasets)
            },
            "created_by": "model_compression/generate_teacher_capability_distill.py",
            "created_ts": created_ts,
            "updated_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "is_partial": is_partial,
            "dry_run": bool(args.dry_run),
            "clean_training_policy": not args.dry_run,
            "allow_final_eval_labels": False,
            "final_test_overlap_count": 0,
            "teacher_model_id": args.teacher_model_id,
            "teacher_model_id_map": teacher_model_id_map,
            "teacher_model_id_map_hash": sha256_text(
                json.dumps(teacher_model_id_map, ensure_ascii=False, sort_keys=True)
            ),
            "trace_version_override": args.trace_version,
            "rehearsal_version_override": args.rehearsal_version,
            "teacher_endpoints": endpoints,
            "input_paths": [display_path(path) for path in input_paths],
            "input_hashes": {display_path(path): sha256_file(path) for path in input_paths},
            "selected_dataset_keys": sorted(datasets),
            "selected_unique_prompt_count": len(source_rows),
            "deduplicated_input_count": duplicate_count,
            "resumed_trace_count": resumed_count,
            "attempted_trace_count": len(tasks),
            "completed_trace_count": completed_count,
            "pending_trace_count": len(source_rows) - completed_count - len(errors),
            "accepted_training_count": accepted_count,
            "rejected_trace_count": completed_count - accepted_count,
            "accept_rate": accept_rate,
            "min_accept_rate": args.min_accept_rate,
            "selected_validation_group_count": len(selected_groups),
            "accepted_validation_group_count": len(accepted_groups),
            "group_coverage_rate": group_coverage_rate,
            "min_group_coverage_rate": args.min_group_coverage_rate,
            "fully_rejected_validation_groups": fully_rejected_groups,
            "repair_mode": repair_mode,
            "repair_from_trace": display_path(repair_parent_path) if repair_parent_path else None,
            "repair_from_trace_hash": sha256_file(repair_parent_path) if repair_parent_path else None,
            "repair_rounds_per_invocation": args.repair_rounds,
            "repair_attempt_count": repair_attempt_total,
            "repaired_accept_count": repaired_accept_count,
            "parent_revalidated_accept_count": parent_revalidated_accept_count,
            "retry_rejected_on_resume": bool(args.retry_rejected_on_resume),
            "audit_only": bool(args.audit_only),
            "trace_dataset_counts": dict(sorted(trace_counts.items())),
            "accepted_dataset_counts": dict(sorted(accepted_counts.items())),
            **task_gate,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
            "workers": args.workers,
            "max_new_tokens": args.max_new_tokens,
            "retry_count": args.retry_count,
            "resume": bool(args.resume),
            "checkpoint_interval": args.checkpoint_interval,
            "trace_path": display_path(trace_path),
            "trace_hash": sha256_file(trace_path),
            "distill_path": display_path(output_path),
            "distill_hash": sha256_file(output_path),
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(
            json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
        )
        write_json(target_audit, audit)
        return audit

    def record_trace(trace: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        trace_by_key[str(trace["input_key"])] = trace
        print(
            f"[{'ACCEPT' if trace['accepted_for_training'] else 'REJECT'}] "
            f"{completed}/{len(tasks)} {trace['sample_id']} dataset={trace['dataset_key']} "
            f"verify={trace['verification']} endpoint={trace['teacher_url']} latency_ms={trace['latency_ms']:.1f}",
            flush=True,
        )

    def maybe_checkpoint() -> None:
        if args.checkpoint_interval <= 0 or completed <= 0 or completed % args.checkpoint_interval:
            return
        audit = write_outputs(partial_audit_path, "running", True)
        print(
            f"[CHECKPOINT] completed={audit['completed_trace_count']}/{len(source_rows)} "
            f"accepted={audit['accepted_training_count']} audit={display_path(partial_audit_path)}",
            flush=True,
        )

    if tasks:
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for batch_start in range(0, len(tasks), args.workers):
                    batch = tasks[batch_start : batch_start + args.workers]
                    futures = {
                        executor.submit(generate_one, batch_start + index, row): row
                        for index, row in enumerate(batch)
                    }
                    for future in as_completed(futures):
                        source_row = futures[future]
                        try:
                            record_trace(future.result())
                        except Exception as exc:
                            completed += 1
                            errors.append(
                                {
                                    "sample_id": str(source_row.get("sample_id", "")),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            print(f"[ERROR] {source_row.get('sample_id', '')}: {exc}", flush=True)
                        maybe_checkpoint()
        except KeyboardInterrupt:
            audit = write_outputs(partial_audit_path, "interrupted", True)
            print(
                f"[INTERRUPTED] completed={audit['completed_trace_count']}/{len(source_rows)} "
                f"accepted={audit['accepted_training_count']} resume with the same command.",
                file=sys.stderr,
                flush=True,
            )
            return 130
    else:
        print(f"[OK] resume found all {resumed_count} selected prompts; no teacher calls needed.", flush=True)

    traces = current_traces()
    accepted_count = sum(1 for trace in traces if trace.get("accepted_for_training") is True)
    accept_rate = accepted_count / len(traces) if traces else 0.0
    selected_groups = {
        str(row.get("validation_group_id", row["sample_id"])) for row in source_rows
    }
    accepted_groups = {
        str(trace.get("validation_group_id", trace["sample_id"]))
        for trace in traces
        if trace.get("accepted_for_training") is True
    }
    group_coverage_rate = len(accepted_groups) / len(selected_groups) if selected_groups else 0.0
    task_gate_failures = task_gate_snapshot(traces)["task_gate_failures"]
    final_status = (
        "passed"
        if not errors
        and len(traces) == len(source_rows)
        and not task_gate_failures
        and (
            args.dry_run
            or (
                accepted_count > 0
                and accept_rate >= args.min_accept_rate
                and group_coverage_rate >= args.min_group_coverage_rate
            )
        )
        else "failed"
    )
    audit = write_outputs(audit_path, final_status, False)
    if args.checkpoint_interval > 0:
        write_outputs(partial_audit_path, "completed" if final_status == "passed" else "completed_with_errors", True)
    print(f"Wrote {display_path(trace_path)}")
    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(
        f"status={audit['status']} accepted={audit['accepted_training_count']}/"
        f"{audit['completed_trace_count']} accept_rate={audit['accept_rate']:.4f} "
        f"group_coverage={audit['accepted_validation_group_count']}/"
        f"{audit['selected_validation_group_count']}"
    )
    return 0 if final_status == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapabilityDistillError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
