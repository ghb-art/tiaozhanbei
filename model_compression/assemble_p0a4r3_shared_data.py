#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4r3_shared_distillation.json"
DEFAULT_OUTPUT = ROOT / "data" / "distill" / "p0a4r3_shared_train.jsonl"
DEFAULT_VALIDATION = (
    ROOT / "data" / "distill" / "p0a4r3_train_only_validation.jsonl"
)
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a4r3_shared_data.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AssemblyError(f"Missing input: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssemblyError(
                    f"Invalid JSON at {display_path(path)}:{line_number}: {exc}"
                ) from exc
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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def group_id(row: dict[str, Any]) -> str:
    value = str(row.get("validation_group_id", "")).strip()
    if not value:
        raise AssemblyError(f"Missing validation_group_id: {row.get('sample_id')}")
    return value


def copy_row(
    row: dict[str, Any],
    capability_domain: str,
    train: bool,
) -> dict[str, Any]:
    copied = dict(row)
    copied["capability_domain"] = capability_domain
    copied["used_for_training"] = train
    copied["used_for_validation"] = not train
    copied["used_for_final_test"] = False
    return copied


def copy_code_row(
    row: dict[str, Any],
    train: bool,
    origin: str,
) -> dict[str, Any]:
    copied = copy_row(row, "code", train)
    copied["origin"] = origin
    return copied


def old_validation_groups(paths: list[Path]) -> set[str]:
    groups: set[str] = set()
    for path in paths:
        if path.is_file():
            groups.update(group_id(row) for row in load_jsonl(path))
    return groups


def select_math_replay(
    source: list[dict[str, Any]],
    train_count: int,
    validation_count: int,
    seed: int,
    excluded_groups: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        row
        for row in source
        if row.get("dataset_key") == "gsm8k"
        and row.get("used_for_final_test") is not True
        and row.get("used_for_validation") is not True
        and "gsm8k/test" not in str(row.get("sample_id", "")).casefold()
        and group_id(row) not in excluded_groups
    ]
    unique = {group_id(row): row for row in eligible}
    ordered = sorted(
        unique.values(),
        key=lambda row: sha256_text(f"{seed}:{group_id(row)}"),
    )
    required = train_count + validation_count
    if len(ordered) < required:
        raise AssemblyError(
            f"Math replay has {len(ordered)} unique groups; requires {required}"
        )
    validation_source = ordered[:validation_count]
    train_source = ordered[validation_count:required]

    def convert(row: dict[str, Any], train: bool) -> dict[str, Any]:
        copied = copy_row(row, "math_replay", train)
        copied["sample_id"] = (
            f"p0a4r3/math-{'train' if train else 'validation'}/"
            f"{sha256_text(group_id(row))[:20]}"
        )
        copied["source"] = "p0a4r3_frozen_math_teacher_verified_replay"
        copied["supervision_type"] = "frozen_math_replay_no_math_specific_training"
        return copied

    return (
        [convert(row, True) for row in train_source],
        [convert(row, False) for row in validation_source],
    )


def formal_reference(row: dict[str, Any]) -> bool:
    searchable = " ".join(
        str(row.get(key, "")).casefold()
        for key in (
            "sample_id",
            "validation_group_id",
            "source_sample_id",
            "split",
            "source_split",
        )
    )
    return any(
        marker in searchable
        for marker in (
            "gsm8k/test",
            "cmmlu/test",
            "humaneval/",
            "official_full",
            "selection170",
            "smoke96",
            "final_test",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble P0-A4R3 shared Code+NLP distillation with Math replay."
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
        mbpp_source = resolve_path(code_config["mbpp_verified"])
        apps_source = resolve_path(code_config["apps_verified"])
        contests_train_source = resolve_path(code_config["code_contests_train"])
        contests_validation_source = resolve_path(
            code_config["code_contests_validation"]
        )
        nlp_train_source = resolve_path(nlp_config["verified_train"])
        nlp_validation_source = resolve_path(nlp_config["verified_validation"])
        math_source = resolve_path(data["math_replay_source"])
        old_validation_paths = [
            resolve_path(value) for value in data["old_validation_sets"]
        ]
        old_groups = old_validation_groups(old_validation_paths)

        mbpp_rows = [
            row
            for row in load_jsonl(mbpp_source)
            if "mbpp" in str(row.get("source", "")).casefold()
        ]
        apps_rows = load_jsonl(apps_source)
        contests_train_rows = load_jsonl(contests_train_source)
        contests_validation_rows = load_jsonl(contests_validation_source)
        nlp_train_rows = load_jsonl(nlp_train_source)
        nlp_validation_rows = load_jsonl(nlp_validation_source)
        math_train, math_validation = select_math_replay(
            load_jsonl(math_source),
            int(data["math_replay_train_groups"]),
            int(data["math_replay_validation_groups"]),
            int(config["seed"]),
            old_groups,
        )

        code_train = [
            *[copy_code_row(row, True, "mbpp") for row in mbpp_rows],
            *[copy_code_row(row, True, "apps") for row in apps_rows],
            *[
                copy_code_row(row, True, "code_contests")
                for row in contests_train_rows
            ],
        ]
        code_validation = [
            copy_code_row(row, False, "code_contests")
            for row in contests_validation_rows
        ]
        nlp_train = [copy_row(row, "nlp", True) for row in nlp_train_rows]
        nlp_validation = [
            copy_row(row, "nlp", False) for row in nlp_validation_rows
        ]
        train = [*math_train, *code_train, *nlp_train]
        validation = [*math_validation, *code_validation, *nlp_validation]
        errors: list[str] = []

        train_groups = [group_id(row) for row in train]
        validation_groups = [group_id(row) for row in validation]
        train_group_set = set(train_groups)
        validation_group_set = set(validation_groups)
        if len(train_group_set) != len(train_groups):
            errors.append("duplicate_train_group")
        if len(validation_group_set) != len(validation_groups):
            errors.append("duplicate_validation_group")
        if train_group_set & validation_group_set:
            errors.append("train_validation_group_overlap")
        if (train_group_set | validation_group_set) & old_groups:
            errors.append("old_reused_validation_group")
        if any(formal_reference(row) for row in train + validation):
            errors.append("protected_or_formal_identity_detected")

        code_origins = Counter(str(row.get("origin", "")) for row in code_train)
        minimum_code = int(code_config["minimum_unique_train_groups"])
        if len({group_id(row) for row in code_train}) < minimum_code:
            errors.append("insufficient_unique_code_train")
        for origin, minimum in code_config["minimum_unique_by_origin"].items():
            if int(code_origins.get(origin, 0)) < int(minimum):
                errors.append(f"insufficient_code_origin:{origin}")
        if len({group_id(row) for row in code_validation}) < int(
            code_config["new_validation_groups"]
        ):
            errors.append("insufficient_new_code_validation")

        nlp_domains = Counter(str(row.get("domain", "")) for row in nlp_train)
        if len({group_id(row) for row in nlp_train}) < int(
            nlp_config["minimum_unique_train_groups"]
        ):
            errors.append("insufficient_unique_nlp_train")
        if len([domain for domain in nlp_domains if domain]) < int(
            nlp_config["required_domains"]
        ):
            errors.append("insufficient_nlp_domains")
        if len({group_id(row) for row in nlp_validation}) < int(
            nlp_config["new_validation_groups"]
        ):
            errors.append("insufficient_new_nlp_validation")

        dataset_counts = Counter(str(row.get("dataset_key", "")) for row in train)
        validation_counts = Counter(
            str(row.get("dataset_key", "")) for row in validation
        )
        if set(dataset_counts) != set(TASKS) or set(validation_counts) != set(TASKS):
            errors.append("missing_capability_domain")
        for row in code_train:
            if not isinstance(row.get("code_eval"), dict):
                errors.append("unverified_code_row")
                break
        for row in nlp_train:
            if not isinstance(row.get("teacher_verification"), dict):
                errors.append("unverified_nlp_row")
                break

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
            "gate": "P0-A4R3-SHARED-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/assemble_p0a4r3_shared_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not errors else "failed",
            "policy": config["policy"],
            "inputs": {
                display_path(path): sha256_file(path)
                for path in (
                    mbpp_source,
                    apps_source,
                    contests_train_source,
                    contests_validation_source,
                    nlp_train_source,
                    nlp_validation_source,
                    math_source,
                )
            },
            "train": display_path(output),
            "train_hash": sha256_file(output),
            "train_rows": len(train),
            "train_counts": dict(dataset_counts),
            "train_unique_groups": len(train_group_set),
            "code_origin_counts": dict(code_origins),
            "nlp_domain_counts": dict(nlp_domains),
            "validation": display_path(validation_output),
            "validation_hash": sha256_file(validation_output),
            "validation_rows": len(validation),
            "validation_counts": dict(validation_counts),
            "validation_unique_groups": len(validation_group_set),
            "train_validation_overlap": len(train_group_set & validation_group_set),
            "old_validation_overlap": len(
                (train_group_set | validation_group_set) & old_groups
            ),
            "formal_test_reference_count": sum(
                formal_reference(row) for row in train + validation
            ),
            "math_policy": "frozen_replay_only",
            "candidate_limit": int(config["policy"]["max_preregistered_candidates"]),
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(stable_json(audit))
        write_json(resolve_path(args.audit), audit)
        print(
            f"P0-A4R3 shared data status={audit['status']} "
            f"train={dict(dataset_counts)} validation={dict(validation_counts)}",
            flush=True,
        )
        if errors:
            raise AssemblyError(str(errors))
        return 0
    except (
        AssemblyError,
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"P0-A4R3 assembly failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
