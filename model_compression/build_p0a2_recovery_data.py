#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from evaluate_chapter2_capability import build_messages, load_cmmlu_sample  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "p0a2_recovery.json"
HUMANEVAL_PATH = ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
BUILD_TIMESTAMP = "2026-07-18T00:00:00+00:00"


class RecoveryDataError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryDataError(f"Cannot read JSON {display_path(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryDataError(f"Expected an object in {display_path(path)}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RecoveryDataError(f"Missing source: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecoveryDataError(
                    f"Invalid JSON at {display_path(path)}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RecoveryDataError(
                    f"Expected an object at {display_path(path)}:{line_number}"
                )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validation_group(row: dict[str, Any]) -> str:
    return str(row.get("validation_group_id") or row.get("sample_id") or "")


def stable_group_order(rows: list[dict[str, Any]], seed: int, namespace: str) -> list[str]:
    groups = {validation_group(row) for row in rows if validation_group(row)}
    return sorted(groups, key=lambda group: sha256_text(f"{seed}:{namespace}:{group}"))


def select_groups(
    rows: list[dict[str, Any]],
    group_ids: set[str],
    *,
    one_row_per_group: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        group = validation_group(row)
        if group not in group_ids or (one_row_per_group and group in seen):
            continue
        selected.append(row)
        seen.add(group)
    return selected


def copy_row(
    row: dict[str, Any],
    *,
    sample_id: str,
    source: str,
    used_for_training: bool,
    validation_group_id: str | None = None,
) -> dict[str, Any]:
    messages = row.get("messages")
    answer = str(row.get("answer", "")).strip()
    if not isinstance(messages, list) or not messages or not answer:
        raise RecoveryDataError(f"Incomplete source row: {row.get('sample_id', '<missing>')}")
    output: dict[str, Any] = {
        "recovery_version": "p0a2-1.0",
        "created_by": "model_compression/build_p0a2_recovery_data.py",
        "created_ts": BUILD_TIMESTAMP,
        "source": source,
        "source_sample_id": str(row.get("sample_id", "")),
        "dataset_key": str(row.get("dataset_key", "")),
        "sample_id": sample_id,
        "validation_group_id": validation_group_id or validation_group(row),
        "messages": messages,
        "answer": answer,
        "used_for_training": used_for_training,
        "used_for_validation": not used_for_training,
        "used_for_final_test": False,
    }
    if isinstance(row.get("code_eval"), dict):
        output["code_eval"] = row["code_eval"]
    output["recovery_row_hash"] = sha256_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return output


def build_cmmlu_validation(sample_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample = load_cmmlu_sample(sample_id)
        messages, _ = build_messages(sample, "v11")
        source = {
            "dataset_key": "cmmlu",
            "sample_id": sample_id,
            "validation_group_id": sample_id,
            "messages": messages,
            "answer": str(sample["reference"]).strip().upper(),
        }
        rows.append(
            copy_row(
                source,
                sample_id=f"p0a2/validation/{sample_id}",
                source="cmmlu_official_dev_p0a2_validation",
                used_for_training=False,
            )
        )
    return rows


def humaneval_prompt_hashes() -> set[str]:
    if not HUMANEVAL_PATH.is_file():
        raise RecoveryDataError(f"Missing HumanEval source: {display_path(HUMANEVAL_PATH)}")
    hashes: set[str] = set()
    with gzip.open(HUMANEVAL_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                hashes.add(sha256_text(str(row.get("prompt", "")).strip()))
    return hashes


def final_reference(row: dict[str, Any], final_ids: set[str]) -> bool:
    fields = " ".join(
        str(row.get(key, "")).lower()
        for key in (
            "sample_id",
            "source_sample_id",
            "validation_group_id",
            "source",
            "split",
            "split_role",
        )
    )
    if any(marker in fields for marker in ("gsm8k/test", "cmmlu/test", "humaneval/", "final_test")):
        return True
    return str(row.get("sample_id", "")) in final_ids or str(row.get("source_sample_id", "")) in final_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the train-only corpus and model-agnostic 170-row selection Dev reused by P0-A3."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = read_json(config_path)
        sources = dict(config["sources"])
        selection = dict(config["selection"])
        outputs = dict(config["outputs"])

        source_paths = {name: resolve_path(str(value)) for name, value in sources.items()}
        for path in source_paths.values():
            if not path.is_file():
                raise RecoveryDataError(f"Missing source: {display_path(path)}")

        math_rows = read_jsonl(source_paths["math_train"])
        code_train_rows = read_jsonl(source_paths["code_train"])
        code_validation_rows = read_jsonl(source_paths["code_validation"])
        nlp_rows = read_jsonl(source_paths["nlp_train"])
        for name, rows in (
            ("math_train", math_rows),
            ("code_train", code_train_rows),
            ("nlp_train", nlp_rows),
        ):
            invalid = [row for row in rows if row.get("used_for_training") is not True]
            if invalid:
                raise RecoveryDataError(f"{name} contains {len(invalid)} non-training rows")
        invalid_validation = [
            row for row in code_validation_rows if row.get("used_for_training") is not False
        ]
        if invalid_validation:
            raise RecoveryDataError(
                f"code_validation contains {len(invalid_validation)} rows not marked selection-only"
            )

        seed = int(config["seed"])
        math_group_order = stable_group_order(math_rows, seed, "math")
        math_validation_count = int(selection["math_validation_groups"])
        math_train_count = int(selection["math_train_groups"])
        if len(math_group_order) < math_validation_count + math_train_count:
            raise RecoveryDataError(
                "Not enough Math groups for the configured disjoint train/validation selection"
            )
        math_validation_groups = set(math_group_order[:math_validation_count])
        math_train_groups = set(
            math_group_order[math_validation_count : math_validation_count + math_train_count]
        )
        selected_math_train = select_groups(
            math_rows, math_train_groups, one_row_per_group=True
        )
        selected_math_validation = select_groups(
            math_rows, math_validation_groups, one_row_per_group=True
        )

        train_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        for row in selected_math_train:
            train_rows.append(
                copy_row(
                    row,
                    sample_id=f"p0a2/train/{row['sample_id']}",
                    source="gsm8k_train_teacher_verified_p0a2",
                    used_for_training=True,
                )
            )
        for row in selected_math_validation:
            validation_rows.append(
                copy_row(
                    row,
                    sample_id=f"p0a2/validation/{row['sample_id']}",
                    source="gsm8k_train_holdout_p0a2_validation",
                    used_for_training=False,
                )
            )

        code_repeats = int(selection["code_train_repeats"])
        if code_repeats <= 0:
            raise RecoveryDataError("code_train_repeats must be positive")
        for exposure in range(code_repeats):
            for row in code_train_rows:
                train_rows.append(
                    copy_row(
                        row,
                        sample_id=f"p0a2/train/{row['sample_id']}/exposure/{exposure:02d}",
                        source="mbpp_official_train_p0a2",
                        used_for_training=True,
                    )
                )
        for row in code_validation_rows:
            validation_rows.append(
                copy_row(
                    row,
                    sample_id=f"p0a2/validation/{row['sample_id']}",
                    source="mbpp_official_dev_select_p0a2_validation",
                    used_for_training=False,
                )
            )

        for row in nlp_rows:
            train_rows.append(
                copy_row(
                    row,
                    sample_id=f"p0a2/train/{row['sample_id']}",
                    source="synthetic_cmmlu_format_recovery_p0a2",
                    used_for_training=True,
                )
            )

        cmmlu_ids = [
            line.strip()
            for line in source_paths["cmmlu_validation_ids"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cmmlu_limit = int(selection["cmmlu_validation_samples"])
        cmmlu_ids = sorted(
            cmmlu_ids, key=lambda sample_id: sha256_text(f"{seed}:cmmlu-dev:{sample_id}")
        )[:cmmlu_limit]
        validation_rows.extend(build_cmmlu_validation(cmmlu_ids))

        train_rows.sort(key=lambda row: str(row["sample_id"]))
        validation_rows.sort(key=lambda row: str(row["sample_id"]))
        train_groups = {validation_group(row) for row in train_rows}
        validation_groups = {validation_group(row) for row in validation_rows}
        group_overlap = train_groups & validation_groups
        duplicate_train = len(train_rows) - len({str(row["sample_id"]) for row in train_rows})
        duplicate_validation = len(validation_rows) - len(
            {str(row["sample_id"]) for row in validation_rows}
        )

        final_ids: set[str] = set()
        final_split_paths: dict[str, Path] = {}
        for dataset in ("gsm8k", "humaneval", "cmmlu"):
            path = ROOT / "data" / "splits" / f"{dataset}_test.txt"
            final_split_paths[dataset] = path
            final_ids.update(
                line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        formal_reference_count = sum(
            final_reference(row, final_ids) for row in train_rows
        )
        formal_prompt_hashes = humaneval_prompt_hashes()
        code_prompt_overlap = 0
        for row in train_rows:
            code_eval = row.get("code_eval")
            if not isinstance(code_eval, dict):
                continue
            prompt_source = str(code_eval.get("prompt_source", "")).strip()
            if prompt_source and sha256_text(prompt_source) in formal_prompt_hashes:
                code_prompt_overlap += 1

        errors: list[str] = []
        if group_overlap:
            errors.append(f"train/validation group overlap: {len(group_overlap)}")
        if duplicate_train or duplicate_validation:
            errors.append(
                f"duplicate sample ids: train={duplicate_train} validation={duplicate_validation}"
            )
        if formal_reference_count:
            errors.append(f"formal test references in training: {formal_reference_count}")
        if code_prompt_overlap:
            errors.append(f"HumanEval prompt overlap in training: {code_prompt_overlap}")
        if any(row.get("used_for_training") is not True for row in train_rows):
            errors.append("training output contains a non-training row")
        if any(row.get("used_for_training") is not False for row in validation_rows):
            errors.append("validation output contains a training row")
        if errors:
            raise RecoveryDataError("; ".join(errors))

        train_path = resolve_path(str(outputs["train"]))
        validation_path = resolve_path(str(outputs["validation"]))
        audit_path = resolve_path(str(outputs["audit"]))
        write_jsonl(train_path, train_rows)
        write_jsonl(validation_path, validation_rows)

        report: dict[str, Any] = {
            "gate": "P0-A2-recovery-data",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0a2_recovery_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "config": display_path(config_path),
            "config_sha256": sha256_file(config_path),
            "seed": seed,
            "sources": {
                name: {"path": display_path(path), "sha256": sha256_file(path)}
                for name, path in sorted(source_paths.items())
            },
            "outputs": {
                "train": {
                    "path": display_path(train_path),
                    "sha256": sha256_file(train_path),
                    "rows": len(train_rows),
                },
                "validation": {
                    "path": display_path(validation_path),
                    "sha256": sha256_file(validation_path),
                    "rows": len(validation_rows),
                },
            },
            "train_dataset_counts": dict(
                sorted(Counter(str(row["dataset_key"]) for row in train_rows).items())
            ),
            "validation_dataset_counts": dict(
                sorted(Counter(str(row["dataset_key"]) for row in validation_rows).items())
            ),
            "train_group_count": len(train_groups),
            "validation_group_count": len(validation_groups),
            "train_validation_group_overlap_count": len(group_overlap),
            "duplicate_train_sample_count": duplicate_train,
            "duplicate_validation_sample_count": duplicate_validation,
            "formal_test_reference_count": formal_reference_count,
            "formal_humaneval_prompt_overlap_count": code_prompt_overlap,
            "formal_split_hashes": {
                dataset: sha256_file(path) for dataset, path in sorted(final_split_paths.items())
            },
            "validation_used_for_training": False,
            "formal_test_labels_used_for_training": False,
            "errors": [],
        }
        report["report_hash"] = sha256_text(
            json.dumps(
                {key: value for key, value in report.items() if key != "report_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        write_json(audit_path, report)
        print(f"Wrote {display_path(train_path)} ({len(train_rows)} rows)")
        print(f"Wrote {display_path(validation_path)} ({len(validation_rows)} rows)")
        print(f"Wrote {display_path(audit_path)}")
        print("Frozen edge-candidate data gate passed.")
        return 0
    except (KeyError, TypeError, ValueError, RecoveryDataError) as exc:
        print(f"Frozen edge-candidate data gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
