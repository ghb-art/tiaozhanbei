#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "datasets" / "apps" / "APPS.tar.gz"
DEFAULT_DATASET_DIR = ROOT / "data" / "datasets" / "apps"
DEFAULT_TRAIN_DIR = DEFAULT_DATASET_DIR / "APPS" / "train"
DEFAULT_OUTPUT = ROOT / "data" / "distill" / "p0a4r_apps_verified_train.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a4r_apps_source.json"
DEFAULT_CACHE = ROOT / "runtime" / "p0a4r_apps_validation.jsonl"
DEFAULT_TOKENIZER = ROOT / "models" / "checkpoints" / "p0a4" / "student-shared-v2-merged"
DEFAULT_HUMANEVAL = ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
DEFAULT_DEV_SELECT = ROOT / "data" / "distill" / "mbpp_v23_dev_select.jsonl"
DEFAULT_DEV_GATE = ROOT / "data" / "distill" / "mbpp_v23_dev_gate.jsonl"

APPS_URL = "https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz"
APPS_EXPECTED_SHA256 = "6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca"
APPS_EXPECTED_SIZE = 446_695_050
APPS_EXPECTED_TRAIN_TASKS = 5_000
BUILD_VERSION = "p0a4r-apps-1.0"

ALLOWED_IMPORTS = {
    "bisect", "collections", "copy", "decimal", "fractions", "functools", "heapq",
    "io", "itertools", "math", "operator", "re", "statistics", "string", "sys", "typing",
}
FORBIDDEN_CALLS = {
    "__import__", "breakpoint", "compile", "eval", "exec", "exit", "input", "open", "quit",
}
FORBIDDEN_ATTRIBUTES = {
    "connect", "exit", "fork", "kill", "popen", "remove", "rmdir", "rmtree", "system",
    "unlink", "urlopen",
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
APPS_EQUALITY_HELPER = r"""
def _p0a4r_normalize_value(value):
    if isinstance(value, tuple):
        return [_p0a4r_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_p0a4r_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _p0a4r_normalize_value(item) for key, item in value.items()}
    return value

def _p0a4r_apps_equal(actual, expected):
    actual = _p0a4r_normalize_value(actual)
    expected = _p0a4r_normalize_value(expected)
    return actual == expected or (
        isinstance(expected, list) and expected and actual == expected[0]
    )
""".strip()
EXEC_PREAMBLE = """
import resource
resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
from typing import *
import math
import re
import itertools
import functools
import collections
from collections import *
import heapq
import bisect
import string
import operator
import copy
import decimal
import fractions
import statistics
from functools import *
""".strip()


class AppsRebuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawCandidate:
    cache_key: str
    sample_id: str
    payload: dict[str, Any]


class StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.type_comment = None
        node.decorator_list = []
        return node


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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AppsRebuildError(
                    f"Expected an object at {display_path(path)}:{line_number}"
                )
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_archive(archive: Path) -> str:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file():
        if archive.stat().st_size == APPS_EXPECTED_SIZE:
            if sha256_file(archive) == APPS_EXPECTED_SHA256:
                return "existing_verified"
            raise AppsRebuildError(
                f"Existing APPS archive has the wrong hash: {display_path(archive)}"
            )
        if archive.stat().st_size > APPS_EXPECTED_SIZE:
            raise AppsRebuildError(
                f"Existing APPS archive is larger than expected: {display_path(archive)}"
            )
    command = [
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "2",
        "--continue-at", "-", "--output", str(archive), APPS_URL,
    ]
    print(f"Downloading official APPS archive to {display_path(archive)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise AppsRebuildError(f"APPS download failed with curl exit code {completed.returncode}")
    if archive.stat().st_size != APPS_EXPECTED_SIZE:
        raise AppsRebuildError(
            f"APPS archive size mismatch: {archive.stat().st_size} != {APPS_EXPECTED_SIZE}"
        )
    digest = sha256_file(archive)
    if digest != APPS_EXPECTED_SHA256:
        raise AppsRebuildError(f"APPS archive SHA256 mismatch: {digest}")
    return "downloaded_verified"


def safe_archive_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) >= 2
        and path.parts[0] == "APPS"
        and path.parts[1] == "train"
    )


def train_task_count(train_dir: Path) -> int:
    if not train_dir.is_dir():
        return 0
    return sum(task.is_dir() for task in train_dir.iterdir())


def complete_task_count(train_dir: Path) -> int:
    if not train_dir.is_dir():
        return 0
    return sum(
        task.is_dir()
        and all(
            (task / name).is_file()
            for name in ("metadata.json", "input_output.json", "solutions.json", "question.txt")
        )
        for task in train_dir.iterdir()
    )


def extract_train_only(archive: Path, dataset_dir: Path) -> str:
    train_dir = dataset_dir / "APPS" / "train"
    if train_task_count(train_dir) == APPS_EXPECTED_TRAIN_TASKS:
        return "existing_verified"
    if train_dir.exists():
        raise AppsRebuildError(
            f"Existing APPS train directory is incomplete; move it aside before retrying: "
            f"{display_path(train_dir)}"
        )
    extracted_files = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not safe_archive_member(member.name):
                continue
            if member.issym() or member.islnk():
                raise AppsRebuildError(f"APPS archive contains a link: {member.name}")
            destination = dataset_dir.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise AppsRebuildError(f"Could not read APPS archive member: {member.name}")
            with source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            extracted_files += 1
    count = train_task_count(train_dir)
    if count != APPS_EXPECTED_TRAIN_TASKS:
        raise AppsRebuildError(
            f"APPS train extraction is incomplete: {count} != {APPS_EXPECTED_TRAIN_TASKS}"
        )
    write_json(
        dataset_dir / "SOURCE.json",
        {
            "source": APPS_URL,
            "source_repository": "https://github.com/hendrycks/apps",
            "archive_sha256": APPS_EXPECTED_SHA256,
            "archive_size_bytes": APPS_EXPECTED_SIZE,
            "split": "train_only",
            "train_task_count": count,
            "extracted_file_count": extracted_files,
            "test_split_extracted": False,
            "created_ts": datetime.now(timezone.utc).isoformat(),
        },
    )
    return "extracted_verified"


def compact_description(value: str, max_chars: int) -> str:
    value = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", value.replace("\x00", " ")).strip())
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0].rstrip() + "\n[Specification truncated.]"


def safe_source(source: str) -> tuple[bool, str]:
    try:
        module = ast.parse(source)
    except (IndentationError, SyntaxError):
        return False, "syntax_error"
    for node in ast.walk(module):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef)):
            return False, f"forbidden_node_{type(node).__name__}"
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS for alias in node.names):
                return False, "forbidden_import"
        if isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".", 1)[0]
            if node.level != 0 or module_name not in ALLOWED_IMPORTS:
                return False, "forbidden_import"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return False, f"forbidden_call_{node.func.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                return False, f"forbidden_attribute_{node.attr}"
    return True, ""


def extract_apps_function(solution: str, function_name: str) -> tuple[ast.FunctionDef | None, str]:
    try:
        module = ast.parse(solution)
    except (IndentationError, SyntaxError):
        return None, "solution_syntax_error"
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    selected_node = function
    class_method = False
    if function is None:
        for node in module.body:
            if not isinstance(node, ast.ClassDef) or node.name != "Solution":
                continue
            matches = [
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == function_name
            ]
            if len(matches) == 1:
                function = matches[0]
                selected_node = function
                class_method = True
                break
    if function is None:
        return None, "entry_point_missing"
    function = copy.deepcopy(function)
    if class_method and function.args.args and function.args.args[0].arg in {"self", "cls"}:
        function.args.args = function.args.args[1:]
    if any(isinstance(node, ast.Name) and node.id in {"self", "cls"} for node in ast.walk(function)):
        return None, "instance_state_dependency"
    StripAnnotations().visit(function)
    ast.fix_missing_locations(function)
    setup_nodes: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            setup_nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node is not selected_node and node.name != function_name:
            helper = copy.deepcopy(node)
            StripAnnotations().visit(helper)
            setup_nodes.append(helper)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            setup_nodes.append(copy.deepcopy(node))
    setup = "\n".join(ast.unparse(node) for node in setup_nodes).strip()
    return function, setup


def function_prompt(function: ast.FunctionDef, description: str) -> str:
    skeleton = copy.deepcopy(function)
    skeleton.body = [ast.Pass()]
    skeleton.decorator_list = []
    skeleton.returns = None
    skeleton.type_comment = None
    StripAnnotations().visit(skeleton)
    ast.fix_missing_locations(skeleton)
    signature = ast.unparse(skeleton).splitlines()[0]
    return f"{signature}\n    {json.dumps(description, ensure_ascii=False)}\n"


def body_source(function: ast.FunctionDef) -> str:
    return textwrap.indent(
        "\n".join(ast.unparse(node) for node in function.body).strip(),
        "    ",
    )


def apps_tests(function_name: str, inputs: list[Any], outputs: list[Any]) -> list[str]:
    tests: list[str] = []
    for args, expected in zip(inputs, outputs):
        if not isinstance(args, (list, tuple)):
            raise AppsRebuildError("APPS call-based input is not a positional argument list")
        tests.append(
            f"assert _p0a4r_apps_equal({function_name}(*{repr(list(args))}), {repr(expected)})"
        )
    return tests


def apps_descriptor(task_dir: Path, max_test_chars: int) -> RawCandidate | None:
    try:
        metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        input_output = json.loads((task_dir / "input_output.json").read_text(encoding="utf-8"))
        solutions = json.loads((task_dir / "solutions.json").read_text(encoding="utf-8"))
        question = (task_dir / "question.txt").read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    difficulty = str(metadata.get("difficulty", "")).lower()
    function_name = str(input_output.get("fn_name", "")).strip()
    inputs = input_output.get("inputs", [])
    outputs = input_output.get("outputs", [])
    if difficulty not in {"introductory", "interview"} or not function_name:
        return None
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None
    if not isinstance(solutions, list) or not solutions:
        return None
    if len(stable_json({"inputs": inputs, "outputs": outputs})) > max_test_chars:
        return None
    payload = {
        "task_id": task_dir.name,
        "difficulty": difficulty,
        "url": str(metadata.get("url", "")),
        "function_name": function_name,
        "inputs": inputs,
        "outputs": outputs,
        "solutions": [str(solution) for solution in solutions],
        "question": question,
    }
    return RawCandidate(
        sha256_text(f"{BUILD_VERSION}:{stable_json(payload)}"),
        f"apps/train/{task_dir.name}",
        payload,
    )


def run_isolated_check(
    candidate_source: str,
    setup_code: str,
    tests: list[str],
    timeout_sec: float,
) -> tuple[bool, str]:
    script = "\n".join((EXEC_PREAMBLE, setup_code, candidate_source, *tests))
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if completed.returncode == 0:
        return True, "passed"
    details = (completed.stderr or completed.stdout).strip().splitlines()
    return False, details[-1][:240] if details else f"returncode={completed.returncode}"


def build_apps_row(
    candidate: RawCandidate,
    timeout_sec: float,
    max_solution_candidates: int,
    max_prompt_chars: int,
    max_answer_chars: int,
    max_training_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    payload = candidate.payload
    try:
        tests = apps_tests(payload["function_name"], payload["inputs"], payload["outputs"])
    except AppsRebuildError:
        return None, "invalid_call_tests"
    first_reason = "no_passing_canonical_solution"
    for solution_index, solution in enumerate(payload["solutions"][:max_solution_candidates]):
        function, setup_or_reason = extract_apps_function(solution, payload["function_name"])
        if function is None:
            first_reason = setup_or_reason
            continue
        setup_code = setup_or_reason
        prompt_source = function_prompt(
            function,
            compact_description(payload["question"], max_prompt_chars),
        )
        answer = body_source(function)
        if len(answer) > max_answer_chars:
            first_reason = "answer_too_long"
            continue
        if len(prompt_source) + len(setup_code) + len(answer) > max_training_chars:
            first_reason = "training_sequence_too_long"
            continue
        candidate_source = (
            f"{setup_code}\n{ast.unparse(function)}" if setup_code else ast.unparse(function)
        )
        safe, reason = safe_source(candidate_source)
        if not safe:
            first_reason = reason
            continue
        passed, detail = run_isolated_check(
            candidate_source,
            APPS_EQUALITY_HELPER,
            tests,
            timeout_sec,
        )
        if not passed:
            first_reason = f"canonical_failed:{detail}"
            continue
        user_prompt = (
            "Complete the Python function. Return only valid Python code, no markdown and no explanation. "
            "If the prompt already contains the function header, return only the indented function body.\n\n"
            f"{prompt_source}"
        )
        row = {
            "rebuild_version": BUILD_VERSION,
            "created_by": "model_compression/rebuild_p0a4r_apps_data.py",
            "source": "apps_official_train_call",
            "dataset_key": "humaneval",
            "sample_id": candidate.sample_id,
            "validation_group_id": f"apps/task/{payload['task_id']}",
            "split": "train",
            "split_role": "train",
            "messages": [{"role": "user", "content": user_prompt}],
            "answer": answer,
            "code_eval": {
                "kind": "apps_call_tests_v1",
                "entry_point": function.name,
                "prompt_source": prompt_source,
                "setup_code": f"{APPS_EQUALITY_HELPER}\n{setup_code}".strip(),
                "tests": tests,
                "test_count": len(tests),
                "tests_hash": sha256_text(stable_json(tests)),
                "execution_protocol": "isolated_python_resource_limited_complete_source_tests",
            },
            "source_metadata": {
                "apps_task_id": payload["task_id"],
                "difficulty": payload["difficulty"],
                "url": payload["url"],
                "canonical_solution_index": solution_index,
                "all_available_tests_executed": True,
            },
            "used_for_training": True,
            "used_for_validation": False,
            "used_for_final_test": False,
        }
        row["rebuild_row_hash"] = sha256_text(stable_json(row))
        return row, ""
    return None, first_reason


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        str(row["cache_key"]): row
        for row in read_jsonl(path)
        if row.get("cache_key")
    }


def validate_candidates(
    candidates: list[RawCandidate],
    cache_path: Path,
    workers: int,
    builder: Any,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    cache = load_cache(cache_path)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    pending: list[RawCandidate] = []
    for candidate in candidates:
        cached = cache.get(candidate.cache_key)
        if cached is None:
            pending.append(candidate)
        elif isinstance(cached.get("row"), dict):
            accepted.append(cached["row"])
        else:
            rejected[str(cached.get("reason", "cached_rejection"))] += 1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(builder, candidate): candidate for candidate in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                candidate = futures[future]
                try:
                    row, reason = future.result()
                except Exception as exc:
                    row, reason = None, f"worker_error:{type(exc).__name__}"
                cache_handle.write(
                    stable_json(
                        {
                            "cache_key": candidate.cache_key,
                            "sample_id": candidate.sample_id,
                            "row": row,
                            "reason": reason,
                        }
                    )
                    + "\n"
                )
                cache_handle.flush()
                if row is not None:
                    accepted.append(row)
                else:
                    rejected[reason or "unknown_rejection"] += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(
                        f"[APPS validate] {completed}/{len(pending)} "
                        f"accepted_total={len(accepted)} rejected_total={sum(rejected.values())}",
                        flush=True,
                    )
    accepted.sort(key=lambda row: str(row["sample_id"]))
    return accepted, rejected, len(candidates) - len(pending)


def normalized_tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def shingles(value: str, width: int = 5) -> set[str]:
    tokens = normalized_tokens(value)
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


class OverlapIndex:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.exact: dict[str, str] = {}
        self.entries: dict[int, tuple[str, set[str]]] = {}
        self.inverted: dict[str, set[int]] = defaultdict(set)
        self.next_id = 0

    def add(self, label: str, text: str) -> None:
        normalized = " ".join(normalized_tokens(text))
        if not normalized:
            return
        self.exact.setdefault(sha256_text(normalized), label)
        grams = shingles(text)
        entry_id = self.next_id
        self.next_id += 1
        self.entries[entry_id] = (label, grams)
        for gram in grams:
            self.inverted[gram].add(entry_id)

    def match(self, text: str) -> tuple[str, float] | None:
        normalized = " ".join(normalized_tokens(text))
        exact = self.exact.get(sha256_text(normalized)) if normalized else None
        if exact:
            return exact, 1.0
        grams = shingles(text)
        candidate_ids: set[int] = set()
        for gram in grams:
            candidate_ids.update(self.inverted.get(gram, ()))
        best: tuple[str, float] | None = None
        for entry_id in candidate_ids:
            label, reference = self.entries[entry_id]
            union = len(grams | reference)
            score = len(grams & reference) / union if union else 0.0
            if score >= self.threshold and (best is None or score > best[1]):
                best = label, score
        return best


def load_decontamination_index(
    humaneval_path: Path,
    dev_paths: tuple[Path, Path],
    threshold: float,
) -> tuple[OverlapIndex, set[str], dict[str, int]]:
    index = OverlapIndex(threshold)
    entry_points: set[str] = set()
    counts = {"humaneval_prompt_only": 0, "internal_dev_prompt_only": 0}
    with gzip.open(humaneval_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index.add(str(row.get("task_id", "HumanEval")), str(row.get("prompt", "")))
            counts["humaneval_prompt_only"] += 1
    for path in dev_paths:
        for row in read_jsonl(path):
            code_eval = row.get("code_eval", {})
            prompt = "\n".join(
                str(message.get("content", ""))
                for message in row.get("messages", [])
                if isinstance(message, dict)
            )
            index.add(str(row.get("sample_id", "internal-dev")), prompt)
            entry = str(code_eval.get("entry_point", "")).casefold()
            if entry:
                entry_points.add(entry)
            counts["internal_dev_prompt_only"] += 1
    return index, entry_points, counts


def code_fingerprint(row: dict[str, Any]) -> str:
    code_eval = row.get("code_eval", {})
    source = "\n".join(
        (
            str(code_eval.get("setup_code", "")),
            str(code_eval.get("prompt_source", "")).rstrip(),
            str(row.get("answer", "")),
        )
    )
    try:
        tree = ast.parse(source)
    except (IndentationError, SyntaxError):
        return ""
    return sha256_text(ast.dump(tree, annotate_fields=True, include_attributes=False))


def decontaminate_and_deduplicate(
    rows: list[dict[str, Any]],
    forbidden: OverlapIndex,
    forbidden_entry_points: set[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    accepted_prompts = OverlapIndex(threshold)
    fingerprints: set[str] = set()
    rejected: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        prompt = str(row["messages"][0]["content"])
        entry = str(row.get("code_eval", {}).get("entry_point", "")).casefold()
        match = forbidden.match(prompt) or accepted_prompts.match(prompt)
        reason = ""
        fingerprint = ""
        if entry and entry in forbidden_entry_points:
            reason = "internal_dev_entry_point_collision"
        elif match:
            reason = "lexical_overlap"
        else:
            fingerprint = code_fingerprint(row)
            if not fingerprint:
                reason = "invalid_code_fingerprint"
            elif fingerprint in fingerprints:
                reason = "duplicate_code_ast"
        if reason:
            rejected[reason] += 1
            if len(examples) < 20:
                examples.append({"sample_id": row["sample_id"], "reason": reason, "match": match})
            continue
        selected.append(row)
        accepted_prompts.add(str(row["sample_id"]), prompt)
        fingerprints.add(fingerprint)
    return selected, rejected, examples


def filter_token_budget(
    rows: list[dict[str, Any]],
    tokenizer_dir: Path,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    accepted: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "qwen3" in tokenizer_dir.as_posix().lower() or "student-shared" in tokenizer_dir.as_posix():
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(row["messages"], **template_kwargs)
        count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        count += len(
            tokenizer(str(row["answer"]) + tokenizer.eos_token, add_special_tokens=False)[
                "input_ids"
            ]
        )
        if count <= max_tokens:
            accepted.append(row)
            lengths.append(count)
    ordered = sorted(lengths)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0
    return accepted, {
        "input_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_sequence_too_long": len(rows) - len(accepted),
        "max_tokens": max(ordered) if ordered else 0,
        "p95_tokens": p95,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a train-only, executable APPS source for P0-A4R."
    )
    parser.add_argument("command", choices=("download", "build", "all", "status"))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--target-unique", type=int, default=1500)
    parser.add_argument("--minimum-unique", type=int, default=1000)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--max-solution-candidates", type=int, default=3)
    parser.add_argument("--max-test-chars", type=int, default=160_000)
    parser.add_argument("--max-prompt-chars", type=int, default=2_400)
    parser.add_argument("--max-answer-chars", type=int, default=3_200)
    parser.add_argument("--max-training-chars", type=int, default=6_000)
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--overlap-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def print_status(archive: Path, train_dir: Path, output: Path, audit: Path) -> None:
    print(f"archive={display_path(archive)} exists={archive.is_file()}")
    if archive.is_file():
        print(f"archive_bytes={archive.stat().st_size} archive_sha256={sha256_file(archive)}")
    print(
        f"train_dir={display_path(train_dir)} tasks={train_task_count(train_dir)} "
        f"complete_tasks={complete_task_count(train_dir)}"
    )
    print(f"output={display_path(output)} exists={output.is_file()}")
    if output.is_file():
        print(f"output_sha256={sha256_file(output)}")
    print(f"audit={display_path(audit)} exists={audit.is_file()}")


def build(args: argparse.Namespace, train_dir: Path, output: Path, audit_path: Path) -> None:
    required = (
        train_dir,
        resolve_path(args.tokenizer_dir),
        DEFAULT_HUMANEVAL,
        DEFAULT_DEV_SELECT,
        DEFAULT_DEV_GATE,
    )
    missing = [display_path(path) for path in required if not path.exists()]
    if missing:
        raise AppsRebuildError(f"Missing build inputs: {missing}")
    descriptors = [
        descriptor
        for task_dir in sorted(train_dir.iterdir())
        if task_dir.is_dir()
        for descriptor in [apps_descriptor(task_dir, args.max_test_chars)]
        if descriptor is not None
    ]
    builder = lambda candidate: build_apps_row(  # noqa: E731
        candidate,
        args.timeout_sec,
        args.max_solution_candidates,
        args.max_prompt_chars,
        args.max_answer_chars,
        args.max_training_chars,
    )
    verified, execution_rejections, cache_hits = validate_candidates(
        descriptors,
        resolve_path(args.cache),
        args.workers,
        builder,
    )
    forbidden, forbidden_entries, decontamination_counts = load_decontamination_index(
        DEFAULT_HUMANEVAL,
        (DEFAULT_DEV_SELECT, DEFAULT_DEV_GATE),
        args.overlap_threshold,
    )
    unique, overlap_rejections, overlap_examples = decontaminate_and_deduplicate(
        verified,
        forbidden,
        forbidden_entries,
        args.overlap_threshold,
    )
    within_budget, token_audit = filter_token_budget(
        unique,
        resolve_path(args.tokenizer_dir),
        args.max_sequence_tokens,
    )
    selected = sorted(
        within_budget,
        key=lambda row: sha256_text(f"{args.seed}:apps:{row['validation_group_id']}"),
    )[: args.target_unique]
    selected.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(output, selected)
    group_count = len({str(row["validation_group_id"]) for row in selected})
    errors: list[str] = []
    if group_count < args.minimum_unique:
        errors.append("insufficient_unique_verified_apps_groups")
    if group_count != len(selected):
        errors.append("duplicate_validation_group")
    forbidden_markers = ("humaneval/", "selection170", "smoke96", "official_full")
    if any(
        marker in str(row.get("sample_id", "")).lower()
        for row in selected
        for marker in forbidden_markers
    ):
        errors.append("formal_or_gate_identity_in_training")
    report = {
        "gate": "P0-A4R-APPS-TRAIN-SOURCE",
        "check_version": "1.0",
        "created_by": "model_compression/rebuild_p0a4r_apps_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "policy": {
            "source": "APPS official archive",
            "split": "train_only",
            "evaluation_split_extracted": False,
            "canonical_validation": "all stored call tests executed in a resource-limited subprocess",
            "formal_humaneval_use": "prompt-only decontamination; labels, canonical answers and tests not read",
            "internal_dev_use": "prompt/entry-point-only decontamination; labels and tests not read",
            "selection170_feedback_used": False,
            "smoke96_item_feedback_used": False,
            "formal_full_feedback_used": False,
        },
        "archive": display_path(resolve_path(args.archive)),
        "archive_sha256": sha256_file(resolve_path(args.archive)),
        "archive_expected_sha256": APPS_EXPECTED_SHA256,
        "train_dir": display_path(train_dir),
        "train_task_count": train_task_count(train_dir),
        "complete_train_task_count": complete_task_count(train_dir),
        "eligible_descriptor_count": len(descriptors),
        "execution_verified_count": len(verified),
        "execution_rejection_counts": dict(execution_rejections),
        "cache_hit_count": cache_hits,
        "decontamination_reference_counts": decontamination_counts,
        "decontamination_labels_or_tests_accessed": False,
        "overlap_rejection_counts": dict(overlap_rejections),
        "overlap_examples": overlap_examples,
        "token_budget": token_audit,
        "selected_unique_group_count": group_count,
        "minimum_unique_group_count": args.minimum_unique,
        "target_unique_group_count": args.target_unique,
        "output": display_path(output),
        "output_sha256": sha256_file(output),
        "formal_test_training_count": 0,
        "formal_test_selection_count": 0,
        "errors": errors,
    }
    report["report_hash"] = sha256_text(stable_json(report))
    write_json(audit_path, report)
    print(
        f"Wrote {display_path(output)} unique_groups={group_count} "
        f"sha256={report['output_sha256']}",
        flush=True,
    )
    print(f"Wrote {display_path(audit_path)} status={report['status']}", flush=True)
    if errors:
        raise AppsRebuildError(f"APPS source gate failed: {errors}")


def main() -> int:
    args = parse_args()
    try:
        if (
            args.workers <= 0
            or args.minimum_unique <= 0
            or args.target_unique < args.minimum_unique
            or args.timeout_sec <= 0
            or not 0 < args.overlap_threshold <= 1
        ):
            raise AppsRebuildError("Invalid worker/count/timeout/overlap setting")
        archive = resolve_path(args.archive)
        dataset_dir = resolve_path(args.dataset_dir)
        train_dir = dataset_dir / "APPS" / "train"
        output = resolve_path(args.output)
        audit = resolve_path(args.audit)
        if args.command == "status":
            print_status(archive, train_dir, output, audit)
            return 0
        if args.command in {"download", "all"}:
            acquisition = download_archive(archive)
            extraction = extract_train_only(archive, dataset_dir)
            print(f"APPS source ready: acquisition={acquisition} extraction={extraction}")
        if args.command in {"build", "all"}:
            build(args, train_dir, output, audit)
        return 0
    except (
        AppsRebuildError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as exc:
        print(f"P0-A4R APPS rebuild failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
