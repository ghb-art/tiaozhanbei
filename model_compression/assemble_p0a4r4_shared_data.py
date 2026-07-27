#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4r4_long_code_distillation.json"
DEFAULT_OUTPUT = ROOT / "data" / "distill" / "p0a4r4_shared_train.jsonl"
DEFAULT_VALIDATION = (
    ROOT / "data" / "distill" / "p0a4r4_train_only_validation.jsonl"
)
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a4r4_shared_data.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")
FORMAL_MARKERS = (
    "gsm8k/test",
    "cmmlu/test",
    "humaneval/",
    "official_full",
    "selection170",
    "smoke96",
    "final_test",
)


class AssemblyError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AssemblyError(f"Missing input: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssemblyError(
                    f"Non-object row at {display_path(path)}:{line_number}"
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def group_id(row: dict[str, Any]) -> str:
    value = str(row.get("validation_group_id", "")).strip()
    if not value:
        raise AssemblyError(
            f"Missing validation_group_id: {row.get('sample_id', '<missing>')}"
        )
    return value


def formal_reference(row: dict[str, Any]) -> bool:
    identity = " ".join(
        str(row.get(key, "")).casefold()
        for key in (
            "sample_id",
            "source_sample_id",
            "validation_group_id",
            "source",
            "split",
            "split_role",
        )
    )
    return any(marker in identity for marker in FORMAL_MARKERS)


def copy_row(
    row: dict[str, Any],
    *,
    train: bool,
    origin: str | None = None,
) -> dict[str, Any]:
    copied = dict(row)
    copied.pop("training_weight", None)
    copied["used_for_training"] = train
    copied["used_for_validation"] = not train
    copied["used_for_final_test"] = False
    if origin is not None:
        copied["origin"] = origin
    return copied


def select_fresh_math(
    source: list[dict[str, Any]],
    count: int,
    seed: int,
    excluded_groups: set[str],
) -> list[dict[str, Any]]:
    unique = {
        group_id(row): row
        for row in source
        if row.get("dataset_key") == "gsm8k"
        and row.get("used_for_final_test") is not True
        and row.get("used_for_validation") is not True
        and not formal_reference(row)
        and group_id(row) not in excluded_groups
    }
    ordered = sorted(
        unique.values(),
        key=lambda row: sha256_text(f"{seed}:math-validation:{group_id(row)}"),
    )
    if len(ordered) < count:
        raise AssemblyError(
            f"Fresh Math validation has {len(ordered)} rows; requires {count}"
        )
    selected: list[dict[str, Any]] = []
    for row in ordered[:count]:
        copied = copy_row(row, train=False)
        copied["sample_id"] = (
            f"p0a4r4/math-validation/{sha256_text(group_id(row))[:20]}"
        )
        copied["source"] = "p0a4r4_fresh_train_only_math_validation"
        selected.append(copied)
    return selected


def assign_training_weights(
    rows: list[dict[str, Any]],
    weight_key: str,
) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    code_rows = [row for row in rows if row.get("dataset_key") == "humaneval"]
    counts = Counter(str(row.get("origin", "")).strip() for row in code_rows)
    if not counts or "" in counts or set(counts) != {
        "mbpp",
        "apps",
        "code_contests",
    }:
        raise AssemblyError(f"Unexpected Code source counts: {dict(counts)}")
    code_total = len(code_rows)
    weights = {
        origin: code_total / (len(counts) * count)
        for origin, count in sorted(counts.items())
    }
    for row in rows:
        if row.get("dataset_key") == "humaneval":
            row[weight_key] = weights[str(row["origin"])]
        else:
            row[weight_key] = 1.0
    mass = Counter()
    for row in code_rows:
        mass[str(row["origin"])] += float(row[weight_key])
    return dict(counts), weights, dict(mass)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble P0-A4R4 compact executable Code distillation with fresh "
            "train-only validation and equal Code-source loss mass."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-output", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    args = parser.parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        data = config["data"]
        code_config = data["code"]
        nlp_config = data["nlp"]
        inputs = {
            "predecessor_train": resolve_path(data["predecessor_shared_train"]),
            "compact_contests": resolve_path(
                code_config["compact_code_contests_train"]
            ),
            "apps_validation": resolve_path(
                code_config["fresh_apps_validation"]
            ),
            "contests_validation": resolve_path(
                code_config["fresh_code_contests_validation"]
            ),
            "nlp_train": resolve_path(nlp_config["verified_train"]),
            "nlp_validation": resolve_path(nlp_config["fresh_validation"]),
            "math_source": resolve_path(data["math_replay_source"]),
        }
        protected_paths = [
            resolve_path(value) for value in data["protected_validation_sets"]
        ]
        protected_groups = {
            group_id(row)
            for path in protected_paths
            if path.is_file()
            for row in read_jsonl(path)
        }
        predecessor = read_jsonl(inputs["predecessor_train"])
        math_train = [
            copy_row(row, train=True)
            for row in predecessor
            if row.get("dataset_key") == "gsm8k"
        ]
        predecessor_code = [
            row for row in predecessor if row.get("dataset_key") == "humaneval"
        ]
        mbpp_train = [
            copy_row(row, train=True, origin="mbpp")
            for row in predecessor_code
            if str(row.get("origin", "")) == "mbpp"
        ]
        apps_train = [
            copy_row(row, train=True, origin="apps")
            for row in predecessor_code
            if str(row.get("origin", "")) == "apps"
        ]
        contests_train = [
            copy_row(row, train=True, origin="code_contests")
            for row in read_jsonl(inputs["compact_contests"])
        ]
        nlp_train = [
            copy_row(row, train=True) for row in read_jsonl(inputs["nlp_train"])
        ]
        apps_validation = [
            copy_row(row, train=False, origin="apps")
            for row in read_jsonl(inputs["apps_validation"])
        ]
        contests_validation = [
            copy_row(row, train=False, origin="code_contests")
            for row in read_jsonl(inputs["contests_validation"])
        ]
        nlp_validation = [
            copy_row(row, train=False)
            for row in read_jsonl(inputs["nlp_validation"])
        ]
        provisional_train = [
            *math_train,
            *mbpp_train,
            *apps_train,
            *contests_train,
            *nlp_train,
        ]
        provisional_validation = [
            *apps_validation,
            *contests_validation,
            *nlp_validation,
        ]
        excluded_math_groups = (
            protected_groups
            | {group_id(row) for row in provisional_train}
            | {group_id(row) for row in provisional_validation}
        )
        math_validation = select_fresh_math(
            read_jsonl(inputs["math_source"]),
            int(data["math_fresh_validation_groups"]),
            int(config["seed"]),
            excluded_math_groups,
        )
        train = provisional_train
        validation = [*math_validation, *provisional_validation]
        weight_key = str(data["sample_weight_key"])
        code_counts, code_weights, code_mass = assign_training_weights(
            train, weight_key
        )
        errors: list[str] = []

        expected_code = {
            key: int(value)
            for key, value in code_config["expected_train_by_origin"].items()
        }
        if code_counts != expected_code:
            errors.append(
                f"unexpected_code_train_counts:{code_counts}!={expected_code}"
            )
        expected_validation = {
            key: int(value)
            for key, value in code_config[
                "fresh_validation_by_origin"
            ].items()
        }
        actual_validation = Counter(
            str(row.get("origin", ""))
            for row in validation
            if row.get("dataset_key") == "humaneval"
        )
        if dict(actual_validation) != expected_validation:
            errors.append(
                "unexpected_code_validation_counts:"
                f"{dict(actual_validation)}!={expected_validation}"
            )
        if len(nlp_train) != int(nlp_config["expected_train_groups"]):
            errors.append("unexpected_nlp_train_count")
        if len(nlp_validation) != int(nlp_config["fresh_validation_groups"]):
            errors.append("unexpected_nlp_validation_count")
        if len(math_train) != int(data["math_replay_train_groups"]):
            errors.append("unexpected_math_train_count")
        if len(math_validation) != int(data["math_fresh_validation_groups"]):
            errors.append("unexpected_math_validation_count")

        for row in contests_train:
            quality = row.get("distillation_quality")
            if not isinstance(quality, dict):
                errors.append("code_contests_missing_distillation_quality")
                break
            if quality.get("solution_selection") != code_config[
                "solution_selection"
            ]:
                errors.append("code_contests_not_shortest_solution")
                break
            if int(quality.get("answer_token_count", 10**9)) > int(
                code_config["max_answer_tokens"]
            ):
                errors.append("code_contests_answer_token_limit_exceeded")
                break
            if quality.get("all_selected_tests_passed") is not True:
                errors.append("code_contests_execution_not_verified")
                break
        if any(
            not isinstance(row.get("code_eval"), dict)
            for row in train + validation
            if row.get("dataset_key") == "humaneval"
        ):
            errors.append("unverified_code_row")
        if any(
            not isinstance(row.get("teacher_verification"), dict)
            for row in train + validation
            if row.get("dataset_key") == "cmmlu"
        ):
            errors.append("unverified_nlp_row")

        train_groups = [group_id(row) for row in train]
        validation_groups = [group_id(row) for row in validation]
        train_set = set(train_groups)
        validation_set = set(validation_groups)
        if len(train_set) != len(train_groups):
            errors.append("duplicate_train_group")
        if len(validation_set) != len(validation_groups):
            errors.append("duplicate_validation_group")
        if train_set & validation_set:
            errors.append("train_validation_group_overlap")
        if validation_set & protected_groups:
            errors.append("fresh_validation_reuses_protected_group")
        if any(formal_reference(row) for row in train + validation):
            errors.append("formal_or_protected_identity_detected")
        if any(
            not math.isfinite(float(row.get(weight_key, 0)))
            or float(row.get(weight_key, 0)) <= 0
            for row in train
        ):
            errors.append("invalid_training_weight")
        if code_mass and max(code_mass.values()) - min(code_mass.values()) > 1e-6:
            errors.append("unequal_code_source_loss_mass")

        train_counts = Counter(str(row.get("dataset_key", "")) for row in train)
        validation_counts = Counter(
            str(row.get("dataset_key", "")) for row in validation
        )
        expected_train_counts = {
            key: int(value)
            for key, value in config["training"]["student_shared"][
                "target_rows_by_task"
            ].items()
        }
        if dict(train_counts) != expected_train_counts:
            errors.append(
                f"unexpected_train_task_counts:{dict(train_counts)}"
            )
        if set(validation_counts) != set(TASKS):
            errors.append("missing_validation_task")

        random.Random(int(config["seed"])).shuffle(train)
        validation.sort(
            key=lambda row: (
                str(row.get("dataset_key", "")),
                str(row.get("sample_id", "")),
            )
        )
        output = resolve_path(args.output)
        validation_output = resolve_path(args.validation_output)
        write_jsonl(output, train)
        write_jsonl(validation_output, validation)
        audit = {
            "gate": "P0-A4R4-SHARED-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/assemble_p0a4r4_shared_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not errors else "failed",
            "policy": config["policy"],
            "inputs": {
                name: {
                    "path": display_path(path),
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            },
            "protected_validation_inputs": {
                display_path(path): sha256_file(path)
                for path in protected_paths
                if path.is_file()
            },
            "train": display_path(output),
            "train_hash": sha256_file(output),
            "train_rows": len(train),
            "train_counts": dict(train_counts),
            "train_unique_groups": len(train_set),
            "validation": display_path(validation_output),
            "validation_hash": sha256_file(validation_output),
            "validation_rows": len(validation),
            "validation_counts": dict(validation_counts),
            "validation_unique_groups": len(validation_set),
            "train_validation_overlap": len(train_set & validation_set),
            "protected_validation_overlap": len(
                validation_set & protected_groups
            ),
            "code_source_counts": code_counts,
            "code_source_weights": code_weights,
            "code_weighted_mass": code_mass,
            "sample_weight_key": weight_key,
            "code_solution_policy": {
                "selection": code_config["solution_selection"],
                "max_answer_tokens": code_config["max_answer_tokens"],
                "max_sequence_tokens": code_config["max_sequence_tokens"],
            },
            "formal_test_reference_count": sum(
                formal_reference(row) for row in train + validation
            ),
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(stable_json(audit))
        write_json(resolve_path(args.audit), audit)
        print(
            f"P0-A4R4 shared data status={audit['status']} "
            f"train={dict(train_counts)} validation={dict(validation_counts)}",
            flush=True,
        )
        if errors:
            raise AssemblyError(str(errors))
        return 0
    except (
        AssemblyError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"P0-A4R4 assembly failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
