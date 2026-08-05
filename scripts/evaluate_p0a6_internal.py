#!/usr/bin/env python3
"""Run the isolated P0-A6 train-only, domain-aware validation.

This evaluator deliberately does not know how to load any official test split or
old P0-A5 gate.  Its only accepted inputs are the frozen P0-A6 quick/full
validation manifests built from training-only sources.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
P0A6_DATA_ROOT = ROOT / "data" / "p0a6"
ALLOWED_MANIFEST_NAMES = {"quick_validation.jsonl", "full_validation.jsonl"}
DOMAINS = ("math", "code", "nlp")
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
FORBIDDEN_CODE_TEXT = (
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
FORMAL_DATA_MARKERS = (
    "gsm8k/test/",
    "cmmlu/test/",
    "humaneval",
    "humaneval+",
    "official_test",
    "official-full",
    "official_full",
    "formal_test",
    "formal-full",
    "formal_full",
    "reports/sealed",
)
RETIRED_PROTOCOL_MARKERS = (
    "gate300",
    "smoke96",
    "selection170",
    "capability_v2",
    "p0a5_edge_student",
)


class EvaluationError(RuntimeError):
    """Raised for protocol, endpoint, or evaluation integrity failures."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_manifest_path(path: Path) -> None:
    if not is_relative_to(path, P0A6_DATA_ROOT):
        raise EvaluationError(
            f"Manifest must be inside data/p0a6, got {display_path(path)}"
        )
    if path.name not in ALLOWED_MANIFEST_NAMES:
        raise EvaluationError(
            "Only quick_validation.jsonl or full_validation.jsonl is allowed"
        )
    if not path.is_file():
        raise EvaluationError(f"Missing validation manifest: {display_path(path)}")


def validate_output_path(path: Path, suffix: str) -> None:
    if path.suffix != suffix:
        raise EvaluationError(f"Output must end in {suffix}: {display_path(path)}")
    if not is_relative_to(path, ROOT):
        raise EvaluationError(f"Output must remain inside project: {display_path(path)}")
    p0a6_audit_root = (ROOT / "reports" / "audit" / "p0a6").resolve()
    if not is_relative_to(path, p0a6_audit_root):
        raise EvaluationError(
            f"P0-A6 internal outputs must be inside reports/audit/p0a6: {display_path(path)}"
        )
    lowered = path.as_posix().casefold()
    forbidden_roots = (
        (ROOT / "reports" / "sealed").resolve(),
        (ROOT / "data" / "eval").resolve(),
        (ROOT / "data" / "capability_v2").resolve(),
    )
    if any(is_relative_to(path, root) for root in forbidden_roots):
        raise EvaluationError(f"Output targets a sealed/retired area: {display_path(path)}")
    if any(marker in lowered for marker in RETIRED_PROTOCOL_MARKERS):
        raise EvaluationError(f"Output references a retired gate: {display_path(path)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON on manifest line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"Non-object manifest row {line_number}")
            rows.append(row)
    if not rows:
        raise EvaluationError("Validation manifest is empty")
    return rows


def normalized_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} is not an object")
    try:
        counts = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} contains a non-integer count") from exc
    if set(counts) != set(DOMAINS) or any(count <= 0 for count in counts.values()):
        raise EvaluationError(
            f"{label} must contain positive math/code/nlp counts, got {counts}"
        )
    return {domain: counts[domain] for domain in DOMAINS}


def sidecar_entry(sidecar: dict[str, Any], kind: str) -> dict[str, Any] | None:
    validation = sidecar.get("validation")
    if isinstance(validation, dict) and isinstance(validation.get(kind), dict):
        return validation[kind]
    direct = sidecar.get(f"{kind}_validation")
    if isinstance(direct, dict):
        return direct
    return None


def expected_counts_from_manifest(
    manifest_path: Path,
    actual_counts: Counter[str],
) -> tuple[dict[str, int], str, Path | None, str]:
    """Read preregistered counts from the P0-A6 sidecar when it is available.

    Early data-builder runs may not yet have written the sidecar.  In that case
    the JSONL manifest itself is authoritative and its domain counts are recorded
    as derived rather than silently substituted with fixed constants.
    """

    kind = "quick" if manifest_path.name.startswith("quick_") else "full"
    sidecar_path = manifest_path.parent / "manifest.json"
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Invalid P0-A6 sidecar: {exc}") from exc
        if not isinstance(sidecar, dict):
            raise EvaluationError("P0-A6 sidecar is not an object")
        entry = sidecar_entry(sidecar, kind)
        if entry is not None and "expected_counts" in entry:
            expected = normalized_counts(
                entry["expected_counts"], f"validation.{kind}.expected_counts"
            )
            declared_path = entry.get("path")
            if declared_path:
                raw_declared = Path(str(declared_path))
                resolved_declared = (
                    raw_declared.resolve()
                    if raw_declared.is_absolute()
                    else (ROOT / raw_declared).resolve()
                )
                if (
                    resolved_declared != manifest_path.resolve()
                    and not raw_declared.is_absolute()
                ):
                    resolved_declared = (sidecar_path.parent / raw_declared).resolve()
                if resolved_declared != manifest_path.resolve():
                    raise EvaluationError(
                        f"Sidecar path mismatch: {display_path(resolved_declared)}"
                    )
            declared_hash = str(
                entry.get("hash", entry.get("sha256", entry.get("file_hash", "")))
            )
            actual_hash = sha256_file(manifest_path)
            if declared_hash and declared_hash != actual_hash:
                raise EvaluationError("Validation manifest hash does not match sidecar")
            return expected, f"{display_path(sidecar_path)}:validation.{kind}", sidecar_path, declared_hash
        validation_counts = sidecar.get("validation_counts")
        if isinstance(validation_counts, dict) and kind in validation_counts:
            expected = normalized_counts(
                validation_counts[kind], f"validation_counts.{kind}"
            )
            return expected, f"{display_path(sidecar_path)}:validation_counts.{kind}", sidecar_path, ""

    expected = normalized_counts(dict(actual_counts), "JSONL manifest domain counts")
    return expected, "jsonl_manifest_rows", sidecar_path if sidecar_path.is_file() else None, ""


def validate_rows(
    rows: list[dict[str, Any]], manifest_name: str
) -> Counter[str]:
    counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    expected_role = (
        "quick_validation" if manifest_name == "quick_validation.jsonl" else "full_validation"
    )
    for index, row in enumerate(rows, 1):
        domain = str(row.get("domain", ""))
        if domain not in DOMAINS:
            raise EvaluationError(f"Unsupported domain on row {index}: {domain!r}")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise EvaluationError(f"Missing sample_id on row {index}")
        if sample_id in sample_ids:
            raise EvaluationError(f"Duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if not str(row.get("prompt", "")).strip():
            raise EvaluationError(f"Missing prompt for {sample_id}")
        role = str(row.get("split_role", ""))
        if role and role != expected_role:
            raise EvaluationError(
                f"Unexpected split_role for {sample_id}: {role}, expected {expected_role}"
            )
        searchable = " ".join(
            str(row.get(key, ""))
            for key in ("sample_id", "dataset_key", "task_id", "source", "split_role")
        ).casefold()
        marker = next(
            (
                value
                for value in (*FORMAL_DATA_MARKERS, *RETIRED_PROTOCOL_MARKERS)
                if value in searchable
            ),
            "",
        )
        if marker:
            raise EvaluationError(
                f"Forbidden formal/retired reference in {sample_id}: {marker}"
            )
        if domain in {"math", "nlp"} and not str(row.get("reference", "")).strip():
            raise EvaluationError(f"Missing reference for {domain} row {sample_id}")
        if domain == "code":
            tests = row.get("unit_tests")
            if not isinstance(tests, list) or not tests or not all(
                isinstance(item, str) and item.strip() for item in tests
            ):
                raise EvaluationError(f"Invalid unit_tests for {sample_id}")
        counts[domain] += 1
    if set(counts) != set(DOMAINS):
        raise EvaluationError(f"Manifest must contain all three domains: {dict(counts)}")
    return counts


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Endpoint request failed: {url}: {exc}") from exc


def endpoint_root(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def discover_model(endpoint: str, requested: str, timeout: float) -> str:
    payload = request_json(endpoint_root(endpoint) + "/v1/models", None, timeout)
    if not isinstance(payload, dict):
        raise EvaluationError("Malformed /v1/models response")
    ids = [
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if requested:
        if ids and requested not in ids:
            raise EvaluationError(f"Requested model {requested!r} is not served: {ids}")
        return requested
    if len(ids) == 1:
        return ids[0]
    raise EvaluationError(f"Could not infer a unique endpoint model id: {ids}")


def discover_models_by_domain(
    endpoint: str,
    default_model_id: str,
    requested_by_domain: dict[str, str],
    timeout: float,
) -> dict[str, str]:
    model_ids: dict[str, str] = {}
    for domain in DOMAINS:
        requested = str(requested_by_domain.get(domain, "")).strip()
        if not requested:
            requested = default_model_id.strip()
        model_ids[domain] = discover_model(endpoint, requested, timeout)
    return model_ids


def generate(
    endpoint: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    enable_thinking: bool = False,
) -> tuple[str, float]:
    started = time.perf_counter()
    payload = request_json(
        endpoint_root(endpoint) + "/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        },
        timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EvaluationError(f"Malformed completion response: {payload}") from exc
    if content is None:
        raise EvaluationError("Completion content is null")
    return str(content), latency_ms


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if not number.is_finite():
        return cleaned
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def extract_number(value: str) -> tuple[str, bool]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        strict = re.fullmatch(r"####\s*(%s)\s*" % NUMBER_PATTERN.pattern, lines[-1])
        if strict:
            return normalize_number(strict.group(1)), True
    has_delimiter = "####" in value
    candidate = value.rsplit("####", 1)[-1] if has_delimiter else value
    matches = NUMBER_PATTERN.findall(candidate)
    if not matches and has_delimiter:
        matches = NUMBER_PATTERN.findall(value.rsplit("####", 1)[0])
    return (normalize_number(matches[-1]), False) if matches else ("", False)


STRICT_CHOICE = re.compile(
    r"(?:最终答案|FINAL)\s*[:：]\s*([ABCD])\s*[。.]?", re.IGNORECASE
)


def extract_choice(value: str) -> tuple[str, bool]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        strict = STRICT_CHOICE.fullmatch(lines[-1])
        if strict:
            return strict.group(1).upper(), True
    upper = value.upper()
    explicit = re.findall(
        r"(?:最终答案|正确答案|答案|FINAL|ANSWER|OPTION|选项)"
        r"\s*(?:是|为|选择)?\s*[:：]?\s*([ABCD])(?![A-Z])",
        upper,
    )
    if explicit:
        return explicit[-1], False
    selected = re.findall(r"(?:故选|应选|选择|选)\s*[:：]?\s*([ABCD])(?![A-Z])", upper)
    if selected:
        return selected[-1], False
    return "", False


def extract_code(value: str) -> tuple[str, bool]:
    complete = re.fullmatch(
        r"\s*```(?:python)?\s*\n?(.*?)```\s*",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if complete:
        return complete.group(1).strip(), True
    match = re.search(
        r"```(?:python)?\s*\n?(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (match.group(1) if match else value).strip(), False


def safe_python(source: str) -> bool:
    lowered = source.casefold()
    if any(marker in lowered for marker in FORBIDDEN_CODE_TEXT):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS
                for alias in node.names
            ):
                return False
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".", 1)[0] not in ALLOWED_IMPORTS:
                return False
    return True


def sandbox_limits(cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    os.setsid()


def score_code(
    response: str, tests: list[str], timeout: float
) -> tuple[bool, str, bool]:
    code, canonical = extract_code(response)
    source = code + "\n\n" + "\n".join(tests) + "\n"
    if not safe_python(source):
        return False, "unsafe_or_invalid_python", canonical
    cpu_seconds = max(1, min(int(math.ceil(timeout)), 30))
    with tempfile.TemporaryDirectory(prefix="p0a6-internal-code-") as temp_dir:
        program = Path(temp_dir) / "main.py"
        program.write_text(source, encoding="utf-8")
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(program)],
                cwd=temp_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=lambda: sandbox_limits(cpu_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout", canonical
    if completed.returncode == 0:
        return True, "passed", canonical
    detail = (
        completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip()
        else f"returncode={completed.returncode}"
    )
    return False, detail[:500], canonical


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    domain = str(row["domain"])
    if domain == "math":
        instruction = (
            "Solve the problem concisely. End with one line formatted as `#### 42`, "
            "where 42 is replaced by the actual numeric answer."
        )
    elif domain == "code":
        instruction = (
            "Return only a complete Python function implementation in one python code block. "
            "Do not use files, network access, third-party packages, or explanatory prose."
        )
    else:
        instruction = (
            "请简要分析这道中文选择题。最后一行必须严格使用“最终答案：A”的格式，"
            "并将A替换为实际的A、B、C或D选项。"
        )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": str(row["prompt"])},
    ]


def score_row(
    row: dict[str, Any], response: str, code_timeout: float
) -> tuple[bool, str, str, bool]:
    domain = str(row["domain"])
    if domain == "math":
        prediction, canonical = extract_number(response)
        reference, _ = extract_number(str(row["reference"]))
        return prediction == reference and bool(prediction), prediction, f"reference={reference}", canonical
    if domain == "nlp":
        prediction, canonical = extract_choice(response)
        reference = str(row["reference"]).strip().upper()
        if not re.fullmatch(r"[ABCD]", reference):
            raise EvaluationError(f"Invalid NLP reference for {row['sample_id']}: {reference}")
        return prediction == reference, prediction, f"reference={reference}", canonical
    passed, detail, canonical = score_code(
        response, [str(item) for item in row["unit_tests"]], code_timeout
    )
    return passed, "pass" if passed else "fail", detail, canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-only P0-A6 quick/full validation manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-id-math", default="")
    parser.add_argument("--model-id-code", default="")
    parser.add_argument("--model-id-nlp", default="")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--code-timeout-sec", type=float, default=5)
    parser.add_argument("--max-tokens-math", type=int, default=512)
    parser.add_argument("--max-tokens-code", type=int, default=768)
    parser.add_argument("--max-tokens-nlp", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = resolve_path(args.manifest)
        output_path = resolve_path(args.output_trace)
        audit_path = resolve_path(args.audit)
        validate_manifest_path(manifest_path)
        validate_output_path(output_path, ".jsonl")
        validate_output_path(audit_path, ".json")
        if output_path == audit_path or output_path == manifest_path or audit_path == manifest_path:
            raise EvaluationError("Manifest, trace, and audit paths must be distinct")
        rows = read_jsonl(manifest_path)
        counts = validate_rows(rows, manifest_path.name)
        expected_counts, expected_source, sidecar_path, declared_hash = (
            expected_counts_from_manifest(manifest_path, counts)
        )
        actual_counts = {domain: counts[domain] for domain in DOMAINS}
        if actual_counts != expected_counts:
            raise EvaluationError(
                f"Manifest counts differ from expected counts: actual={actual_counts} "
                f"expected={expected_counts}"
            )
        if args.timeout_sec <= 0 or args.code_timeout_sec <= 0:
            raise EvaluationError("Timeouts must be positive")
        model_ids = discover_models_by_domain(
            args.endpoint,
            args.model_id,
            {
                "math": args.model_id_math,
                "code": args.model_id_code,
                "nlp": args.model_id_nlp,
            },
            args.timeout_sec,
        )
        token_limits = {
            "math": args.max_tokens_math,
            "code": args.max_tokens_code,
            "nlp": args.max_tokens_nlp,
        }
        if any(value <= 0 for value in token_limits.values()):
            raise EvaluationError(f"Token limits must be positive: {token_limits}")

        created_ts = datetime.now(timezone.utc).isoformat()
        trace: list[dict[str, Any]] = []
        total = len(rows)
        for index, row in enumerate(rows, 1):
            domain = str(row["domain"])
            model_id = model_ids[domain]
            messages = build_messages(row)
            prompt_hash = sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            )
            response = ""
            generation_error = ""
            started = time.perf_counter()
            try:
                response, latency_ms = generate(
                    args.endpoint,
                    model_id,
                    messages,
                    token_limits[domain],
                    args.timeout_sec,
                )
                correct, prediction, detail, canonical = score_row(
                    row, response, args.code_timeout_sec
                )
            except (EvaluationError, OSError, ValueError) as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                correct = False
                prediction = ""
                detail = str(exc)
                canonical = False
                generation_error = f"{type(exc).__name__}: {exc}"
            result = {
                "capability_eval_version": "p0a6-internal-v1",
                "created_ts": created_ts,
                "candidate_name": args.candidate_name,
                "served_model_id": model_id,
                "domain": row["domain"],
                "dataset_key": row.get("dataset_key", ""),
                "sample_id": row["sample_id"],
                "prompt_hash": prompt_hash,
                "prediction": prediction,
                "correct": correct,
                "canonical_format": canonical,
                "score_detail": detail,
                "latency_ms": latency_ms,
                "generation_error": generation_error,
                "response_text": response,
            }
            result["row_hash"] = sha256_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
            trace.append(result)
            print(
                f"[{index}/{total}] {row['domain']} correct={correct} "
                f"canonical={canonical} latency_ms={latency_ms:.1f}",
                flush=True,
            )

        write_jsonl_atomic(output_path, trace)
        correct_counts: Counter[str] = Counter()
        canonical_counts: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()
        latency_by_domain: dict[str, list[float]] = defaultdict(list)
        for item in trace:
            domain = str(item["domain"])
            correct_counts[domain] += int(item["correct"] is True)
            canonical_counts[domain] += int(item["canonical_format"] is True)
            error_counts[domain] += int(bool(item["generation_error"]))
            latency_by_domain[domain].append(float(item["latency_ms"]))

        accuracy = {
            domain: correct_counts[domain] / expected_counts[domain]
            for domain in DOMAINS
        }
        canonical_rates = {
            domain: canonical_counts[domain] / expected_counts[domain]
            for domain in DOMAINS
        }
        generation_errors = sum(error_counts.values())
        manifest_hash = sha256_file(manifest_path)
        unique_model_ids = set(model_ids.values())
        served_model_id = (
            next(iter(unique_model_ids))
            if len(unique_model_ids) == 1
            else "domain_routed"
        )
        audit: dict[str, Any] = {
            "gate": "P0-A6-INTERNAL-EVAL",
            "check_version": "1.0",
            "created_by": "scripts/evaluate_p0a6_internal.py",
            "created_ts": created_ts,
            "status": "passed" if generation_errors == 0 else "failed",
            "candidate_name": args.candidate_name,
            "endpoint": args.endpoint,
            "served_model_id": served_model_id,
            "served_model_id_by_domain": model_ids,
            "manifest": display_path(manifest_path),
            "manifest_hash": manifest_hash,
            "sidecar_manifest": display_path(sidecar_path) if sidecar_path else "",
            "sidecar_manifest_hash": sha256_file(sidecar_path) if sidecar_path else "",
            "declared_manifest_hash": declared_hash,
            "expected_counts_source": expected_source,
            "expected_counts": expected_counts,
            "actual_counts": actual_counts,
            "correct_counts": {domain: correct_counts[domain] for domain in DOMAINS},
            "accuracy_by_domain": accuracy,
            "macro_accuracy": mean(accuracy.values()),
            "canonical_format_counts": {
                domain: canonical_counts[domain] for domain in DOMAINS
            },
            "canonical_format_rate_by_domain": canonical_rates,
            "macro_canonical_format_rate": mean(canonical_rates.values()),
            "generation_error_count": generation_errors,
            "generation_error_count_by_domain": {
                domain: error_counts[domain] for domain in DOMAINS
            },
            "mean_latency_ms_by_domain": {
                domain: mean(latency_by_domain[domain]) for domain in DOMAINS
            },
            "max_tokens": token_limits,
            "request_timeout_sec": args.timeout_sec,
            "code_timeout_sec": args.code_timeout_sec,
            "output_trace": display_path(output_path),
            "output_trace_hash": sha256_file(output_path),
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        write_json_atomic(audit_path, audit)
        print(f"Wrote {display_path(output_path)}")
        print(f"Wrote {display_path(audit_path)}")
        print(
            f"accuracy={accuracy} canonical_format={canonical_rates} "
            f"generation_errors={generation_errors}"
        )
        return 0 if audit["status"] == "passed" else 1
    except (EvaluationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A6 internal evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
