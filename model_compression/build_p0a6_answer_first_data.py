#!/usr/bin/env python3
"""Build the train-only P0-A6 answer-first Chinese MCQ corpus.

The corpus combines two labelled, non-test sources:

* the 1,335 C-Eval auxiliary-training rows whose 14B rationales already passed
  the human-label lock; and
* the 335 public CMMLU ``dev`` demonstrations (five per subject).

CMMLU ``test`` is never discovered or opened.  The target exposes the locked
answer before the rationale so the weighted choice token is conditioned only
on the question, while still ending in the canonical evaluation format.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEVAL = ROOT / "data/p0a6/nlp_mcq_rationale_train.jsonl"
DEFAULT_CMMLU_DEV = ROOT / "data/datasets/cmmlu/data/dev"
DEFAULT_OUTPUT = ROOT / "data/p0a6/nlp_answer_first_train.jsonl"
DEFAULT_AUDIT = ROOT / "reports/audit/gate_p0a6_answer_first_data.json"
EXPECTED_CEVAL_ROWS = 1335
EXPECTED_CMMLU_DEV_ROWS = 335
EXPECTED_CMMLU_SUBJECTS = 67
CHOICES = ("A", "B", "C", "D")
NLP_SYSTEM_PROMPT = (
    "请简要分析这道中文选择题。最后一行必须严格使用“最终答案：A”的格式，"
    "并将A替换为实际的A、B、C或D选项。"
)
FINAL_RE = re.compile(r"最终答案\s*[:：]\s*([A-D])\s*$", re.I)


class AnswerFirstDataError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnswerFirstDataError(f"Missing input: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnswerFirstDataError(
                    f"Invalid JSON at {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise AnswerFirstDataError(f"Non-object row at line {line_number}")
            rows.append(row)
    return rows


def normalized_question(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


def user_message(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def compact_rationale(answer: str, maximum_chars: int = 180) -> tuple[str, str]:
    match = FINAL_RE.search(str(answer).strip())
    if match is None:
        raise AnswerFirstDataError("Rationale target lacks a locked final answer")
    label = match.group(1).upper()
    rationale = str(answer)[: match.start()].strip()
    rationale = re.sub(r"^(?:简短理由|简短分析|理由|分析)\s*[:：]\s*", "", rationale)
    rationale = re.sub(r"\s+", " ", rationale).strip()
    if len(rationale) < 8:
        raise AnswerFirstDataError("Rationale is too short")
    parts = [
        part.strip()
        for part in re.findall(r"[^。！？!?]+[。！？!?]?", rationale)
        if part.strip()
    ]
    selected = "".join(parts[:2]).strip()
    if not selected:
        selected = rationale
    if len(selected) > maximum_chars:
        selected = selected[:maximum_chars].rstrip("，,；;：:、 ") + "。"
    return selected, label


def target(label: str, rationale: str) -> str:
    return f"答案：{label}\n简短理由：{rationale}\n最终答案：{label}"


def build_ceval_rows(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise AnswerFirstDataError(f"C-Eval row count changed: {len(rows)} != {expected_rows}")
    output: list[dict[str, Any]] = []
    for source in rows:
        if (
            source.get("dataset_key") != "ceval_rationale_train"
            or source.get("split_role") != "train"
        ):
            raise AnswerFirstDataError("C-Eval rationale source is not train-only")
        metadata = source.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("human_labelled") is not True:
            raise AnswerFirstDataError("C-Eval source is not human-labelled")
        rationale, label = compact_rationale(str(source.get("answer", "")))
        reference = str(metadata.get("reference_answer", "")).upper()
        if label != reference or label not in CHOICES:
            raise AnswerFirstDataError("C-Eval rationale/human-label disagreement")
        question = user_message(source)
        if not question:
            raise AnswerFirstDataError("C-Eval source has no question")
        source_id = str(source.get("sample_id", ""))
        output.append(
            {
                "sample_id": source_id.replace(
                    "ceval_rationale_train/", "ceval_answer_first_train/", 1
                ),
                "dataset_key": "ceval_answer_first_train",
                "domain": "nlp",
                "task_id": "nlp",
                "source": "C-Eval-labelled+14B-rationale-answer-first",
                "split_role": "train",
                "messages": [
                    {"role": "system", "content": NLP_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "answer": target(label, rationale),
                "answer_token_position": "first",
                "answer_token_weight": 2.0,
                "training_weight": 1.0,
                "quality_weight": 1.0,
                "kl_weight": 0.2,
                "metadata": {
                    "human_labelled": True,
                    "reference_answer": label,
                    "source_sample_id": source_id,
                    "source_dataset_key": "ceval_rationale_train",
                    "teacher_rationale_hash": sha256_text(rationale),
                    "answer_first": True,
                },
            }
        )
    return output


def build_cmmlu_dev_rows(
    dev_root: Path, expected_rows: int, expected_subjects: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not dev_root.is_dir() or dev_root.name.casefold() != "dev":
        raise AnswerFirstDataError(
            f"CMMLU input must be the explicit dev directory: {display_path(dev_root)}"
        )
    files = sorted(dev_root.glob("*.csv"))
    if len(files) != expected_subjects:
        raise AnswerFirstDataError(
            f"CMMLU dev subject count changed: {len(files)} != {expected_subjects}"
        )
    output: list[dict[str, Any]] = []
    file_audit: list[dict[str, Any]] = []
    for path in files:
        file_audit.append({"path": display_path(path), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Question", *CHOICES, "Answer"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise AnswerFirstDataError(f"Malformed CMMLU dev CSV: {display_path(path)}")
            for index, item in enumerate(reader):
                question = str(item.get("Question", "")).strip()
                label = str(item.get("Answer", "")).strip().upper()
                options = {choice: str(item.get(choice, "")).strip() for choice in CHOICES}
                if not question or label not in CHOICES or any(not value for value in options.values()):
                    raise AnswerFirstDataError(
                        f"Invalid CMMLU dev row: {display_path(path)}:{index + 2}"
                    )
                prompt = "问题：" + question + "\n" + "\n".join(
                    f"{choice}. {options[choice]}" for choice in CHOICES
                )
                rationale = f"根据题干条件，符合要求的选项内容是“{options[label]}”。"
                output.append(
                    {
                        "sample_id": f"cmmlu_dev_answer_first/{path.stem}/{index}",
                        "dataset_key": "cmmlu_dev_answer_first_train",
                        "domain": "nlp",
                        "task_id": "nlp",
                        "source": "CMMLU-public-dev-human-labelled",
                        "split_role": "train",
                        "messages": [
                            {"role": "system", "content": NLP_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "answer": target(label, rationale),
                        "answer_token_position": "first",
                        "answer_token_weight": 2.0,
                        "training_weight": 2.0,
                        "quality_weight": 1.0,
                        "kl_weight": 0.2,
                        "metadata": {
                            "human_labelled": True,
                            "reference_answer": label,
                            "cmmlu_split": "dev",
                            "subject": path.stem,
                            "options": options,
                            "answer_first": True,
                        },
                    }
                )
    if len(output) != expected_rows:
        raise AnswerFirstDataError(
            f"CMMLU dev row count changed: {len(output)} != {expected_rows}"
        )
    return output, file_audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    ceval_path = resolve_path(args.ceval_rationales)
    cmmlu_dev = resolve_path(args.cmmlu_dev)
    output_path = resolve_path(args.output)
    audit_path = resolve_path(args.audit)
    ceval = build_ceval_rows(ceval_path, args.expected_ceval_rows)
    cmmlu, cmmlu_files = build_cmmlu_dev_rows(
        cmmlu_dev, args.expected_cmmlu_rows, args.expected_cmmlu_subjects
    )
    rows = ceval + cmmlu
    sample_ids = [str(row["sample_id"]) for row in rows]
    questions = [normalized_question(user_message(row)) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise AnswerFirstDataError("Duplicate answer-first sample_id")
    if len(set(questions)) != len(questions):
        raise AnswerFirstDataError("Cross-source duplicate question detected")
    atomic_write_jsonl(output_path, rows)
    dataset_counts = Counter(str(row["dataset_key"]) for row in rows)
    label_counts = Counter(str(row["metadata"]["reference_answer"]) for row in rows)
    audit: dict[str, Any] = {
        "gate": "P0-A6-NLP-ANSWER-FIRST-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a6_answer_first_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "ceval_source": display_path(ceval_path),
        "ceval_source_hash": sha256_file(ceval_path),
        "cmmlu_source": display_path(cmmlu_dev),
        "cmmlu_dev_files": cmmlu_files,
        "cmmlu_test_files_opened": 0,
        "formal_test_loaded": False,
        "output": display_path(output_path),
        "output_hash": sha256_file(output_path),
        "output_rows": len(rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_questions": len(set(questions)),
        "system_prompt": NLP_SYSTEM_PROMPT,
        "system_prompt_hash": sha256_text(NLP_SYSTEM_PROMPT),
        "answer_first_token_weighted": True,
        "canonical_final_answer_repeated": True,
        "cmmlu_dev_training_weight": 2.0,
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    atomic_write_json(audit_path, audit)
    print(f"Wrote {display_path(output_path)} rows={len(rows)}")
    print(f"Wrote {display_path(audit_path)} status=passed")
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceval-rationales", default=str(DEFAULT_CEVAL))
    parser.add_argument("--cmmlu-dev", default=str(DEFAULT_CMMLU_DEV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--expected-ceval-rows", type=int, default=EXPECTED_CEVAL_ROWS)
    parser.add_argument("--expected-cmmlu-rows", type=int, default=EXPECTED_CMMLU_DEV_ROWS)
    parser.add_argument("--expected-cmmlu-subjects", type=int, default=EXPECTED_CMMLU_SUBJECTS)
    args = parser.parse_args(argv)
    if min(args.expected_ceval_rows, args.expected_cmmlu_rows, args.expected_cmmlu_subjects) <= 0:
        parser.error("Expected counts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except AnswerFirstDataError as exc:
        print(f"P0-A6 answer-first data build failed: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
