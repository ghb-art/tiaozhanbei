#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4_distillation.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")
FORMAL_MARKERS = ("gsm8k/test/", "cmmlu/test/", "humaneval/", "final_test", "official_full")


class ProtocolError(RuntimeError):
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {display_path(path)}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"Missing JSONL: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    f"Invalid JSONL {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ProtocolError(f"Non-object row {display_path(path)}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_frozen(path: Path, payload: str) -> None:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != payload:
            raise ProtocolError(
                f"Refusing to overwrite changed frozen protocol file: {display_path(path)}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")


def official_ids() -> dict[str, list[str]]:
    gsm_path = ROOT / "data/datasets/gsm8k/grade_school_math/data/test.jsonl"
    gsm_count = sum(1 for line in gsm_path.open(encoding="utf-8") if line.strip())
    gsm = [f"gsm8k/test/{index}" for index in range(gsm_count)]

    humaneval_path = ROOT / "data/datasets/humaneval/data/HumanEval.jsonl.gz"
    with gzip.open(humaneval_path, "rt", encoding="utf-8") as handle:
        humaneval = [str(json.loads(line)["task_id"]) for line in handle if line.strip()]

    cmmlu: list[str] = []
    cmmlu_dir = ROOT / "data/datasets/cmmlu/data/test"
    for csv_path in sorted(cmmlu_dir.glob("*.csv")):
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        subject = csv_path.stem
        cmmlu.extend(f"cmmlu/test/{subject}/{index}" for index in range(count))
    return {"gsm8k": gsm, "humaneval": humaneval, "cmmlu": cmmlu}


def freeze_official(config: dict[str, Any]) -> dict[str, Any]:
    split_dir = resolve_path(config["data"]["official_full_split_dir"])
    expected = {key: int(value) for key, value in config["data"]["official_counts"].items()}
    ids_by_task = official_ids()
    actual = {key: len(value) for key, value in ids_by_task.items()}
    if actual != expected:
        raise ProtocolError(f"Official dataset counts changed: expected={expected} actual={actual}")

    files: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        path = split_dir / f"{task}_test.txt"
        payload = "\n".join(ids_by_task[task]) + "\n"
        write_frozen(path, payload)
        files[task] = {
            "path": display_path(path),
            "count": len(ids_by_task[task]),
            "sha256": sha256_text(payload),
        }
    manifest = {
        "protocol_version": "p0a4-official-full-1.0",
        "created_by": "scripts/p0a4_protocol.py",
        "immutable": True,
        "dataset_counts": actual,
        "total_count": sum(actual.values()),
        "files": files,
        "combined_ids_hash": sha256_text(
            "".join("\n".join(ids_by_task[task]) + "\n" for task in TASKS)
        ),
    }
    manifest["manifest_hash"] = sha256_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    )
    write_frozen(split_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("validation_group_id") or row.get("source_sample_id") or row.get("sample_id") or "")


def stable_groups(groups: Iterable[str], seed: int, namespace: str) -> list[str]:
    return sorted(set(groups), key=lambda value: sha256_text(f"{seed}:{namespace}:{value}"))


def copy_holdout(row: dict[str, Any], split: str) -> dict[str, Any]:
    copied = {
        "p0a4_data_version": "1.0",
        "created_by": "scripts/p0a4_protocol.py",
        "source": str(row.get("source", "p0a4_train_source")),
        "source_sample_id": str(row.get("sample_id", "")),
        "dataset_key": str(row.get("dataset_key", "")),
        "sample_id": f"p0a4/{split}/{row.get('sample_id', '')}",
        "validation_group_id": group_id(row),
        "messages": row.get("messages"),
        "answer": str(row.get("answer", "")),
        "used_for_training": False,
        "used_for_validation": True,
        "used_for_final_test": False,
    }
    if isinstance(row.get("code_eval"), dict):
        copied["code_eval"] = row["code_eval"]
    copied["p0a4_row_hash"] = sha256_text(
        json.dumps(copied, ensure_ascii=False, sort_keys=True)
    )
    return copied


def load_mmlu_holdout(sample_id: str, split: str) -> dict[str, Any]:
    parts = sample_id.split("/")
    if len(parts) != 4 or parts[0] != "mmlu":
        raise ProtocolError(f"Invalid MMLU sample id: {sample_id}")
    _, source_split, subject, index_text = parts
    path = ROOT / f"data/datasets/mmlu/data/{source_split}/{subject}_{source_split}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    row = rows[int(index_text)]
    if len(row) < 6:
        raise ProtocolError(f"Invalid MMLU source row: {sample_id}")
    question, choices, answer = row[0], row[1:5], row[5].strip().upper()
    messages = [
        {
            "role": "system",
            "content": "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested.",
        },
        {
            "role": "user",
            "content": (
                "This is a single-choice question. Return only one uppercase letter A, B, C, or D.\n\n"
                f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\n"
                f"C. {choices[2]}\nD. {choices[3]}"
            ),
        },
    ]
    output = {
        "p0a4_data_version": "1.0",
        "created_by": "scripts/p0a4_protocol.py",
        "source": "mmlu_official_val_nonformal_holdout",
        "source_sample_id": sample_id,
        "dataset_key": "cmmlu",
        "sample_id": f"p0a4/{split}/{sample_id}",
        "validation_group_id": sample_id,
        "messages": messages,
        "answer": answer,
        "used_for_training": False,
        "used_for_validation": True,
        "used_for_final_test": False,
    }
    output["p0a4_row_hash"] = sha256_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True)
    )
    return output


def choice_training_row(
    *,
    sample_id: str,
    source: str,
    question: str,
    choices: list[str],
    answer: str,
    language: str,
) -> dict[str, Any]:
    if len(choices) != 4 or answer not in {"A", "B", "C", "D"}:
        raise ProtocolError(f"Invalid choice training row: {sample_id}")
    if language == "zh":
        instruction = "以下是单项选择题。只输出一个大写字母 A、B、C 或 D。"
        question_label = "题目"
    else:
        instruction = "This is a single-choice question. Return only one uppercase letter A, B, C, or D."
        question_label = "Question"
    row = {
        "p0a4_data_version": "1.1",
        "created_by": "scripts/p0a4_protocol.py",
        "source": source,
        "source_sample_id": sample_id,
        "dataset_key": "cmmlu",
        "sample_id": f"p0a4/train/{sample_id}",
        "validation_group_id": sample_id,
        "messages": [
            {
                "role": "system",
                "content": "You are DB4AI-EdgeServe edge capability evaluator. Answer exactly as requested.",
            },
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n{question_label}: {question}\n"
                    f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                ),
            },
        ],
        "answer": answer,
        "used_for_training": True,
        "used_for_validation": False,
        "used_for_final_test": False,
    }
    row["p0a4_row_hash"] = sha256_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
    )
    return row


def cmmlu_dev_training_rows(
    directory: Path,
    forbidden_groups: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, value in enumerate(csv.DictReader(handle)):
                sample_id = f"cmmlu/dev/{path.stem}/{index:05d}"
                if sample_id in forbidden_groups:
                    continue
                rows.append(
                    choice_training_row(
                        sample_id=sample_id,
                        source="cmmlu_official_dev_nonselection_train",
                        question=str(value.get("Question", "")).strip(),
                        choices=[str(value.get(letter, "")).strip() for letter in "ABCD"],
                        answer=str(value.get("Answer", "")).strip().upper(),
                        language="zh",
                    )
                )
    return rows


def stable_csv_sample(path: Path, limit: int, seed: int) -> list[tuple[int, list[str]]]:
    if limit <= 0:
        return []
    selected: list[tuple[int, int, list[str]]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) != 6:
                raise ProtocolError(f"Invalid MMLU auxiliary row: {display_path(path)}:{index + 1}")
            priority = int(sha256_text(f"{seed}:mmlu-aux:{path.stem}:{index}"), 16)
            entry = (-priority, index, row)
            if len(selected) < limit:
                heapq.heappush(selected, entry)
            elif priority < -selected[0][0]:
                heapq.heapreplace(selected, entry)
    return sorted((index, row) for _, index, row in selected)


def mmlu_auxiliary_training_rows(directory: Path, count: int, seed: int) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise ProtocolError(f"No MMLU auxiliary CSV files: {display_path(directory)}")
    base, remainder = divmod(count, len(paths))
    rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        quota = base + (1 if path_index < remainder else 0)
        for index, value in stable_csv_sample(path, quota, seed):
            rows.append(
                choice_training_row(
                    sample_id=f"mmlu/auxiliary_train/{path.stem}/{index:05d}",
                    source="mmlu_auxiliary_nonformal_train",
                    question=value[0].strip(),
                    choices=[item.strip() for item in value[1:5]],
                    answer=value[5].strip().upper(),
                    language="en",
                )
            )
    if len(rows) != count:
        raise ProtocolError(f"MMLU auxiliary selection incomplete: expected={count} actual={len(rows)}")
    return rows


def build_nlp_training_rows(
    config: dict[str, Any],
    selection_groups: set[str],
) -> list[dict[str, Any]]:
    settings = config["data"]["nlp_train"]
    target = int(settings["target_unique_prompts"])
    minimum = int(settings["min_unique_prompts"])
    if target < minimum or minimum <= 0:
        raise ProtocolError("Invalid NLP unique prompt targets")
    chinese = cmmlu_dev_training_rows(
        resolve_path(settings["cmmlu_dev_dir"]), selection_groups
    )
    if len(chinese) > target:
        chinese = sorted(
            chinese,
            key=lambda row: sha256_text(f"{config['seed']}:cmmlu-dev:{group_id(row)}"),
        )[:target]
    remaining = target - len(chinese)
    auxiliary_candidates = mmlu_auxiliary_training_rows(
        resolve_path(settings["mmlu_auxiliary_dir"]),
        remaining + min(64, remaining),
        int(config["seed"]),
    )
    seen_prompt_hashes = {
        sha256_text(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True))
        for row in chinese
    }
    auxiliary: list[dict[str, Any]] = []
    for row in sorted(
        auxiliary_candidates,
        key=lambda value: sha256_text(f"{config['seed']}:nlp-fill:{group_id(value)}"),
    ):
        prompt_hash = sha256_text(
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        )
        if prompt_hash in seen_prompt_hashes:
            continue
        seen_prompt_hashes.add(prompt_hash)
        auxiliary.append(row)
        if len(auxiliary) == remaining:
            break
    if len(auxiliary) != remaining:
        raise ProtocolError(
            f"Not enough unique MMLU auxiliary prompts: expected={remaining} actual={len(auxiliary)}"
        )
    rows = chinese + auxiliary
    prompt_hashes = {
        sha256_text(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True))
        for row in rows
    }
    if len(rows) != target or len(prompt_hashes) != len(rows):
        raise ProtocolError(
            f"NLP training prompts are not unique: rows={len(rows)} unique={len(prompt_hashes)}"
        )
    return rows


def build_corpora(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    source_path = resolve_path(data["source_train"])
    source_rows = read_jsonl(source_path)
    selection_rows = read_jsonl(resolve_path(data["selection170"]))
    seed = int(config["seed"])
    holdout_count = int(data["holdout_count_per_task"])

    selection_groups = {group_id(row) for row in selection_rows}
    source_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in source_rows:
        task = str(row.get("dataset_key", ""))
        if task not in source_by_task or row.get("used_for_training") is not True:
            raise ProtocolError(f"Invalid train source row: {row.get('sample_id', '<missing>')}")
        if group_id(row) in selection_groups:
            raise ProtocolError(f"Train/170 overlap: {group_id(row)}")
        source_by_task[task].append(row)

    teacher_groups: dict[str, set[str]] = {}
    smoke_groups: dict[str, set[str]] = {}
    all_holdouts: set[str] = set()
    for task in ("gsm8k", "humaneval"):
        ordered = stable_groups((group_id(row) for row in source_by_task[task]), seed, task)
        required = holdout_count * 2
        if len(ordered) < required:
            raise ProtocolError(f"Not enough {task} groups for two disjoint holdouts")
        teacher_groups[task] = set(ordered[:holdout_count])
        smoke_groups[task] = set(ordered[holdout_count:required])
        all_holdouts.update(ordered[:required])

    mmlu_ids_path = ROOT / "data/splits/mmlu_validation.txt"
    mmlu_ids = [line.strip() for line in mmlu_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered_mmlu = stable_groups(mmlu_ids, seed, "cmmlu-mmlu-holdout")
    required_mmlu = holdout_count * 2
    if len(ordered_mmlu) < required_mmlu:
        raise ProtocolError("Not enough MMLU validation rows for NLP holdouts")
    teacher_groups["cmmlu"] = set(ordered_mmlu[:holdout_count])
    smoke_groups["cmmlu"] = set(ordered_mmlu[holdout_count:required_mmlu])

    train_rows = [
        row
        for row in source_rows
        if row.get("dataset_key") != "cmmlu" and group_id(row) not in all_holdouts
    ]
    train_rows.extend(build_nlp_training_rows(config, selection_groups))
    teacher_validation: list[dict[str, Any]] = []
    smoke96: list[dict[str, Any]] = []
    for task in ("gsm8k", "humaneval"):
        one_per_group: dict[str, dict[str, Any]] = {}
        for row in source_by_task[task]:
            one_per_group.setdefault(group_id(row), row)
        teacher_validation.extend(
            copy_holdout(one_per_group[group], "teacher_validation")
            for group in sorted(teacher_groups[task])
        )
        smoke96.extend(
            copy_holdout(one_per_group[group], "smoke96")
            for group in sorted(smoke_groups[task])
        )
    teacher_validation.extend(
        load_mmlu_holdout(group, "teacher_validation")
        for group in sorted(teacher_groups["cmmlu"])
    )
    smoke96.extend(
        load_mmlu_holdout(group, "smoke96")
        for group in sorted(smoke_groups["cmmlu"])
    )

    train_path = resolve_path(data["train"])
    teacher_path = resolve_path(data["teacher_validation"])
    smoke_path = resolve_path(data["smoke96"])
    write_jsonl(train_path, train_rows)
    write_jsonl(teacher_path, teacher_validation)
    write_jsonl(smoke_path, smoke96)

    counts = {
        "train": dict(Counter(str(row["dataset_key"]) for row in train_rows)),
        "teacher_validation": dict(Counter(str(row["dataset_key"]) for row in teacher_validation)),
        "smoke96": dict(Counter(str(row["dataset_key"]) for row in smoke96)),
        "selection170": dict(Counter(str(row["dataset_key"]) for row in selection_rows)),
    }
    unique_prompt_counts = {
        task: len(
            {
                sha256_text(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True))
                for row in train_rows
                if row.get("dataset_key") == task
            }
        )
        for task in TASKS
    }
    minimum_unique = {
        key: int(value)
        for key, value in data["min_unique_train_prompts_by_task"].items()
    }
    unique_failures = {
        task: {"actual": unique_prompt_counts.get(task, 0), "required": required}
        for task, required in minimum_unique.items()
        if unique_prompt_counts.get(task, 0) < required
    }
    if unique_failures:
        raise ProtocolError(f"Insufficient unique training prompts: {unique_failures}")
    group_sets = {
        "train": {group_id(row) for row in train_rows},
        "teacher_validation": {group_id(row) for row in teacher_validation},
        "smoke96": {group_id(row) for row in smoke96},
        "selection170": selection_groups,
    }
    overlap = {
        f"{left}__{right}": len(group_sets[left] & group_sets[right])
        for index, left in enumerate(group_sets)
        for right in list(group_sets)[index + 1 :]
    }
    if any(overlap.values()):
        raise ProtocolError(f"P0-A4 split overlap detected: {overlap}")
    return {
        "counts": counts,
        "unique_train_prompt_counts": unique_prompt_counts,
        "min_unique_train_prompts_by_task": minimum_unique,
        "group_overlap_counts": overlap,
        "files": {
            "train": {"path": display_path(train_path), "sha256": sha256_file(train_path)},
            "teacher_validation": {"path": display_path(teacher_path), "sha256": sha256_file(teacher_path)},
            "smoke96": {"path": display_path(smoke_path), "sha256": sha256_file(smoke_path)},
            "selection170": {
                "path": display_path(resolve_path(data["selection170"])),
                "sha256": sha256_file(resolve_path(data["selection170"])),
            },
        },
    }


def row_has_formal_reference(row: dict[str, Any]) -> bool:
    identity = " ".join(
        str(row.get(key, "")).lower()
        for key in ("sample_id", "source_sample_id", "validation_group_id", "source", "split_role")
    )
    return any(marker in identity for marker in FORMAL_MARKERS)


def protocol_preflight(config: dict[str, Any], *, rebuild: bool) -> dict[str, Any]:
    official = freeze_official(config)
    corpora = build_corpora(config) if rebuild else None
    if corpora is None:
        required = ("train", "teacher_validation", "smoke96", "selection170")
        for key in required:
            if not resolve_path(config["data"][key]).is_file():
                raise ProtocolError(f"Missing P0-A4 data artifact: {config['data'][key]}")
        corpora = build_corpora(config)

    train_rows = read_jsonl(resolve_path(config["data"]["train"]))
    leaked = [str(row.get("sample_id", "")) for row in train_rows if row_has_formal_reference(row)]
    if leaked:
        raise ProtocolError(f"Formal test reference in training data: count={len(leaked)}")
    if config["data"].get("formal_test_labels_allowed_for_training") is not False:
        raise ProtocolError("P0-A4 must hard-disable formal test labels for training")

    audit = {
        "gate": "P0-A4-PROTOCOL",
        "check_version": "1.0",
        "created_by": "scripts/p0a4_protocol.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "config": display_path(DEFAULT_CONFIG),
        "config_hash": sha256_file(DEFAULT_CONFIG),
        "official_full": official,
        "corpora": corpora,
        "formal_test_reference_count_in_train": 0,
        "formal_test_item_results_allowed_for_training": False,
        "selection170_feedback": config["gates"]["selection170"]["feedback"],
        "selection170_max_student_versions": config["gates"]["selection170"]["max_student_versions"],
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True)
    )
    output = resolve_path(config["artifacts"]["protocol_audit"])
    write_json(output, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and validate the P0-A4 data protocol.")
    parser.add_argument("command", choices=("freeze-official", "build-corpora", "preflight"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = read_json(config_path)
        if args.command == "freeze-official":
            result = freeze_official(config)
        elif args.command == "build-corpora":
            result = build_corpora(config)
        else:
            result = protocol_preflight(config, rebuild=True)
    except (OSError, KeyError, ValueError, ProtocolError, json.JSONDecodeError) as exc:
        print(f"P0-A4 protocol failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
