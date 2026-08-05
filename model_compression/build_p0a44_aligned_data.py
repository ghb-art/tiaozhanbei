#!/usr/bin/env python3
"""Build train-only P0-A44 data with the exact formal response contracts."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_chapter2_capability import run_assert_tests_check
CONFIG = ROOT / "configs/p0a44_aligned_retrain.json"
OUT = ROOT / "data/p0a44"
AUDIT = ROOT / "reports/audit/gate_p0a44_data.json"
SYSTEM = "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested."
CODE_SOURCES = (
    ROOT / "data/p0a21/code_train.jsonl",
    ROOT / "data/p0a23/code_train.jsonl",
    ROOT / "data/p0a25/code_train_pool.jsonl",
)
COIG = ROOT / "data/datasets/coig_cqia/COIG-CQIA-full.jsonl"
P0A34 = ROOT / "data/p0a34/train.jsonl"
CEVAL = ROOT / "data/datasets/ceval_exam"
CMMLU_DEV = ROOT / "data/datasets/cmmlu/data/dev"
SEED = 20260803
OPTION_RE = re.compile(r"(?:^|\n)\s*([A-E])[\.．、:：]\s*", re.I)
LABEL_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"(?:正确答案|答案)\s*(?:是|为|选择)?\s*[:：\-]?\s*([A-D])(?![A-Z])",
        r"(?:故|所以|因此)?\s*(?:本题)?\s*(?:选择|选|应选)\s*(?:错误的?)?\s*[:：]?\s*([A-D])(?![A-Z])",
        r"(?:只有|唯有)\s*([A-D])\s*(?:项)?\s*(?:是)?(?:正确|符合|对的|适合)",
        r"([A-D])\s*项?\s*(?:最)?(?:正确|符合(?:要求)?|是对的|适合)(?:。|，|,|$)",
        r"([A-D])\s*(?:才)?是\s*(?:正确|符合|对的|最佳)",
    )
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def strip_code_fence(value: str) -> str:
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", value, re.I | re.S)
    return (match.group(1) if match else value).strip()


def called_names(tests: list[str]) -> Counter[str]:
    tree = ast.parse("\n".join(tests))
    return Counter(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def code_target(source: str, tests: list[str]) -> tuple[ast.Module, ast.FunctionDef] | None:
    tree = ast.parse(source)
    counts = called_names(tests)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    ranked = sorted(((counts[node.name], node.name, node) for node in functions), reverse=True)
    if not ranked or ranked[0][0] == 0 or ranked[0][2].decorator_list:
        return None
    return tree, ranked[0][2]


def function_header(source: str, node: ast.FunctionDef) -> str:
    lines = source.splitlines()
    first_body_line = node.body[0].lineno if node.body else node.end_lineno or node.lineno
    header = textwrap.dedent("\n".join(lines[node.lineno - 1 : first_body_line - 1])).strip()
    if not header.startswith("def ") or not header.endswith(":"):
        raise BuildError(f"Unsupported function header for {node.name}")
    return header


def body_only_target(tree: ast.Module, target: ast.FunctionDef) -> str:
    statements: list[ast.stmt] = []
    for node in tree.body:
        if node is target:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            statements.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            statements.append(node)
    target_body = list(target.body)
    if target_body and isinstance(target_body[0], ast.Expr) and isinstance(target_body[0].value, ast.Constant) and isinstance(target_body[0].value.value, str):
        target_body = target_body[1:]
    statements.extend(target_body or [ast.Pass()])
    answer = "\n".join(ast.unparse(node) for node in statements).strip()
    ast.parse("def _p0a44_probe():\n" + textwrap.indent(answer, "    "))
    return answer


def formal_code_prompt(prompt_source: str) -> str:
    return (
        "Complete the Python function correctly. Return only the function body: do not repeat the def "
        "header or docstring, and do not use markdown or explanations. Start the first body statement at "
        "column 1; keep only the relative indentation required inside loops, conditions, and nested blocks. "
        "Handle the edge cases stated in the docstring.\n\n" + prompt_source
    )


def convert_code_row(row: dict[str, Any]) -> dict[str, Any] | None:
    source = strip_code_fence(str(row.get("answer", "")))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    tests = [str(value).strip() for value in metadata.get("unit_tests", []) if str(value).strip()]
    if len(tests) != 10:
        return None
    try:
        selected = code_target(source, tests)
        if selected is None:
            return None
        tree, target = selected
        header = function_header(source, target)
        users = [m for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
        if len(users) != 1:
            return None
        description = str(users[0].get("content", "")).split("Your implementation must satisfy", 1)[0].strip()
        description = description.replace('"""', "''' ")
        doc = "\n".join("    " + line for line in description.splitlines())
        prompt_source = f'{header}\n    """\n{doc}\n    """\n'
        answer = body_only_target(tree, target)
    except (SyntaxError, ValueError, BuildError):
        return None
    digest = sha256_text(str(row["sample_id"]))[:24]
    return {
        "sample_id": f"p0a44/code/{digest}",
        "dataset_key": "code",
        "domain": "code",
        "split_role": "train",
        "source": "OpenCodeInstruct-execution-verified-HumanEval-contract",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": formal_code_prompt(prompt_source)},
        ],
        "answer": answer,
        "answer_token_weight": 1.0,
        "training_weight": 1.0,
        "kl_weight": 0.12,
        "metadata": {
            "source_sample_id": row["sample_id"],
            "entry_point": target.name,
            "prompt_source": prompt_source,
            "unit_tests": tests,
            "source_execution_validation": metadata.get("independent_execution"),
            "output_contract": "humaneval_v15_body_only",
        },
    }


def build_code() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    converted: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    for path in CODE_SOURCES:
        for row in read_jsonl(path):
            source_id = str(row.get("sample_id", ""))
            if source_id in converted:
                rejected["duplicate"] += 1
                continue
            item = convert_code_row(row)
            if item is None:
                rejected["not_function_or_untransformable"] += 1
                continue
            converted[source_id] = item
    candidates = sorted(converted.values(), key=lambda row: sha256_text(f"{SEED}:{row['sample_id']}"))

    def still_passes(item: dict[str, Any]) -> bool:
        metadata = item["metadata"]
        passed, _ = run_assert_tests_check(
            str(metadata["prompt_source"]), str(metadata["entry_point"]),
            [str(value) for value in metadata["unit_tests"]], str(item["answer"]), 10,
        )
        return bool(passed)

    with ThreadPoolExecutor(max_workers=16) as executor:
        verified = list(executor.map(still_passes, candidates))
    rows = [row for row, passed in zip(candidates, verified) if passed]
    rejected["transformed_execution_failed"] += len(candidates) - len(rows)
    for row in rows:
        row["metadata"]["transformed_execution_validation"] = "passed_all_10_tests"
    if len(rows) < 12000:
        raise BuildError(f"Insufficient aligned Code rows: {len(rows)}")
    validation_raw, train = rows[:1000], rows[1000:]
    validation: list[dict[str, Any]] = []
    for row in validation_raw:
        metadata = row["metadata"]
        validation.append({
            "sample_id": row["sample_id"],
            "dataset_key": "humaneval_shape",
            "domain": "code",
            "split_role": "p0a44_internal_validation",
            "prompt": row["messages"][1]["content"],
            "prompt_source": metadata["prompt_source"],
            "entry_point": metadata["entry_point"],
            "unit_tests": metadata["unit_tests"],
            "validator": "humaneval_body_plus_hidden_asserts",
        })
    return train, validation, rejected


def cmmlu_message(question: str, options: dict[str, str]) -> str:
    return (
        "以下是单项选择题。请先判断正确选项，但最终只输出一个大写字母 A、B、C 或 D，不要解释。\n\n"
        f"题目：{question}\nA. {options['A']}\nB. {options['B']}\nC. {options['C']}\nD. {options['D']}\n最终答案："
    )


def nlp_train_row(sample_id: str, question: str, options: dict[str, str], answer: str, source: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "dataset_key": "nlp",
        "domain": "nlp",
        "split_role": "train",
        "source": source,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": cmmlu_message(question, options)},
        ],
        "answer": answer,
        "answer_token_weight": 1.0,
        "training_weight": 1.0,
        "kl_weight": 0.12,
        "metadata": {"output_contract": "cmmlu_v15_single_letter", "reference_answer": answer},
    }


def extract_options(prompt: str) -> tuple[str, dict[str, str]] | None:
    matches = list(re.finditer(r"(?:^|\n)\s*([A-D])[\.．、:：]\s*", prompt, re.I))
    if [m.group(1).upper() for m in matches] != list("ABCD"):
        return None
    question = prompt[: matches[0].start()].strip()
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        options[match.group(1).upper()] = prompt[match.end() : end].strip()
    return question, options


def source_label(output: str) -> str | None:
    labels = {label.upper() for pattern in LABEL_PATTERNS for label in pattern.findall(output)}
    return next(iter(labels)) if len(labels) == 1 else None


def build_nlp_train() -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in read_jsonl(P0A34):
        dataset = str(row.get("dataset_key", ""))
        if dataset not in {"ceval_rationale_train", "mmlu_aux_chinese"}:
            continue
        messages = [m for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
        parsed = extract_options(str(messages[0].get("content", ""))) if len(messages) == 1 else None
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        answer = str(metadata.get("reference_answer") or row.get("answer_letter") or "").strip().upper()
        if parsed is None or answer not in "ABCD":
            continue
        question, options = parsed
        identity = sha256_text(question + json.dumps(options, ensure_ascii=False, sort_keys=True))
        source = "C-Eval-labelled" if dataset == "ceval_rationale_train" else "MMLU-aux-Chinese-verified"
        rows[identity] = nlp_train_row(f"p0a44/nlp/{identity[:24]}", question.removeprefix("问题："), options, answer, source)
        counts[source] += 1
    with COIG.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = row.get("task_type") if isinstance(row.get("task_type"), dict) else {}
            if "试题" not in " ".join(str(x) for x in task.get("major", [])) or "多项" in " ".join(str(x) for x in task.get("minor", [])):
                continue
            prompt = f"{str(row.get('instruction', '')).strip()}\n{str(row.get('input', '')).strip()}".strip()
            parsed = extract_options(prompt)
            label = source_label(str(row.get("output", "")))
            if parsed is None or label is None:
                continue
            question, options = parsed
            identity = sha256_text(question + json.dumps(options, ensure_ascii=False, sort_keys=True))
            if identity in rows:
                continue
            rows[identity] = nlp_train_row(f"p0a44/nlp/{identity[:24]}", question, options, label, "COIG-CQIA-human-labelled-MCQ")
            counts["COIG-CQIA-human-labelled-MCQ"] += 1
    ordered = sorted(rows.values(), key=lambda row: sha256_text(f"{SEED}:{row['sample_id']}"))
    if len(ordered) < 3500:
        raise BuildError(f"Insufficient aligned NLP rows: {len(ordered)}")
    return ordered, counts


def build_ceval_dev() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CEVAL.glob("*/dev-*.parquet")):
        subject = path.parent.name
        for raw in pq.read_table(path).to_pylist():
            options = {letter: str(raw[letter]) for letter in "ABCD"}
            rows.append({
                "sample_id": f"ceval/dev/{subject}/{int(raw['id']):05d}", "dataset_key": "ceval_dev",
                "domain": "nlp", "split_role": "p0a44_internal_validation", "subject": subject,
                "prompt": cmmlu_message(str(raw["question"]), options), "reference": str(raw["answer"]).strip().upper(),
                "validator": "choice_exact",
            })
    return rows


def build_cmmlu_dev() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CMMLU_DEV.glob("*.csv")):
        with path.open(encoding="utf-8") as handle:
            for index, raw in enumerate(csv.DictReader(handle)):
                options = {letter: str(raw[letter]) for letter in "ABCD"}
                rows.append({
                    "sample_id": f"cmmlu/dev/{path.stem}/{index:05d}", "dataset_key": "cmmlu_dev",
                    "domain": "nlp", "split_role": "p0a44_internal_validation", "subject": path.stem,
                    "prompt": cmmlu_message(str(raw["Question"]), options), "reference": str(raw["Answer"]).strip().upper(),
                    "validator": "choice_exact",
                })
    return rows


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("protocol") != "P0-A44-ALIGNED-RETRAIN-AND-EDGE-FULL":
        raise BuildError("P0-A44 config identity changed")
    code_train, code_validation, code_rejections = build_code()
    nlp_train, nlp_counts = build_nlp_train()
    ceval_dev, cmmlu_dev = build_ceval_dev(), build_cmmlu_dev()
    if len(code_validation) != 1000 or len(ceval_dev) != 260 or len(cmmlu_dev) != 335:
        raise BuildError(f"Validation counts changed: code={len(code_validation)} ceval={len(ceval_dev)} cmmlu={len(cmmlu_dev)}")
    outputs = {
        OUT / "code_train.jsonl": code_train,
        OUT / "code_validation.jsonl": code_validation,
        OUT / "nlp_train.jsonl": nlp_train,
        OUT / "nlp_ceval_dev.jsonl": ceval_dev,
        OUT / "nlp_cmmlu_dev.jsonl": cmmlu_dev,
    }
    for path, rows in outputs.items():
        atomic_jsonl(path, rows)
    report = {
        "gate": "P0-A44-ALIGNED-TRAIN-DATA", "check_version": "1.0",
        "created_by": "model_compression/build_p0a44_aligned_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "formal_test_rows_loaded": 0,
        "code": {"train_rows": len(code_train), "validation_rows": len(code_validation), "rejections": dict(code_rejections), "contract": "humaneval_v15_body_only"},
        "nlp": {"train_rows": len(nlp_train), "source_counts": dict(nlp_counts), "ceval_dev_rows": len(ceval_dev), "cmmlu_dev_rows": len(cmmlu_dev), "contract": "cmmlu_v15_single_letter"},
        "outputs": {path.relative_to(ROOT).as_posix(): {"rows": len(rows), "sha256": sha256_file(path)} for path, rows in outputs.items()},
        "inputs": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in (CONFIG, *CODE_SOURCES, P0A34, COIG)},
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    atomic_json(AUDIT, report)
    print(f"Wrote {AUDIT.relative_to(ROOT)}")
    print(f"status=passed code_train={len(code_train)} code_validation={len(code_validation)} nlp_train={len(nlp_train)} nlp_validation={len(ceval_dev)+len(cmmlu_dev)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError, csv.Error) as exc:
        print(f"P0-A44 data build failed: {exc}")
        raise SystemExit(1)
