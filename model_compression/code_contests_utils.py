#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable


ALLOWED_IMPORTS = {
    "array",
    "bisect",
    "collections",
    "copy",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "io",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
    "typing",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "open",
    "quit",
}
FORBIDDEN_ATTRIBUTES = {
    "connect",
    "fork",
    "kill",
    "popen",
    "remove",
    "rmdir",
    "rmtree",
    "system",
    "unlink",
    "urlopen",
}
RESOURCE_PREAMBLE = """
import resource
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
""".strip()


@dataclass(frozen=True)
class ContestExecution:
    passed: bool
    reason: str
    passed_tests: int
    total_tests: int


def extract_python_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.I | re.S)
    return (fenced.group(1) if fenced else stripped).strip()


def safe_contest_source(source: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except (IndentationError, SyntaxError):
        return False, "syntax_error"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".", 1)[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "").split(".", 1)[0]]
            )
            if any(name not in ALLOWED_IMPORTS for name in names):
                return False, "forbidden_import"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, "forbidden_call"
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in FORBIDDEN_ATTRIBUTES
            ):
                return False, "forbidden_attribute"
    return True, ""


def normalize_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def normalize_tests(value: Any, limit: int = 8) -> list[dict[str, str]]:
    tests: list[dict[str, str]] = []
    if isinstance(value, dict):
        inputs = value.get("input", [])
        outputs = value.get("output", [])
        if isinstance(inputs, list) and isinstance(outputs, list):
            for test_input, expected in zip(inputs, outputs):
                tests.append(
                    {"input": str(test_input), "output": str(expected)}
                )
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "input" in item and "output" in item:
                tests.append(
                    {"input": str(item["input"]), "output": str(item["output"])}
                )
    return tests[:limit]


def run_contest_tests(
    source: str,
    tests: Iterable[dict[str, str]],
    timeout_sec: float,
) -> ContestExecution:
    source = extract_python_code(source)
    safe, reason = safe_contest_source(source)
    selected = list(tests)
    if not safe:
        return ContestExecution(False, reason, 0, len(selected))
    if not selected:
        return ContestExecution(False, "missing_tests", 0, 0)
    program = RESOURCE_PREAMBLE + "\n" + source
    passed = 0
    for test in selected:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                input=str(test["input"]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
                env={"PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ContestExecution(False, "timeout", passed, len(selected))
        if completed.returncode != 0:
            return ContestExecution(
                False,
                f"runtime_error:{completed.returncode}",
                passed,
                len(selected),
            )
        if normalize_stdout(completed.stdout) != normalize_stdout(str(test["output"])):
            return ContestExecution(False, "wrong_answer", passed, len(selected))
        passed += 1
    return ContestExecution(True, "", passed, len(selected))


def validate_contest_eval(
    code_eval: dict[str, Any],
    response: str,
    timeout_sec: float,
) -> tuple[bool, str, str]:
    if str(code_eval.get("kind")) != "code_contests_stdio_v1":
        return False, "unsupported_protocol", ""
    tests = normalize_tests(code_eval.get("tests"), limit=64)
    result = run_contest_tests(response, tests, timeout_sec)
    return result.passed, result.reason, extract_python_code(response)


def stable_test_hash(tests: Iterable[dict[str, str]]) -> str:
    import hashlib

    encoded = json.dumps(
        list(tests), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
