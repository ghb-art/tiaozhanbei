from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILES = {
    "dataset": "dataset_manifest.json",
    "run": "manifest.json",
    "conflict": "conflict_gt_manifest.json",
}

PLACEHOLDER_VALUES = {
    "",
    "...",
    "actual",
    "实际值",
    "TBD",
    "TODO",
    "placeholder",
}

PLACEHOLDER_PREFIXES = (
    "REPLACE_WITH_",
    "REPLACE_",
)

REQUIRED_DATASETS = {
    "GSM8K",
    "HumanEval",
    "MMLU",
    "CMMLU",
    "MVTec AD",
    "NEU-DET",
    "CityFlow",
    "UA-DETRAC",
}

DATASET_REQUIRED_KEYS = [
    "dataset_name",
    "dataset_version",
    "official_scale",
    "used_train_count",
    "used_validation_count",
    "used_test_count",
    "final_gate_sample_ids_hash",
    "split_hash",
    "leakage_check_pass",
]

GLOBAL_LEAKAGE_KEYS = [
    "test_in_distill",
    "test_in_planner_train",
    "test_in_policy_train",
    "test_in_graph_train",
    "test_in_calibration",
    "test_in_validation",
]

RUN_MANIFEST_REQUIRED_KEYS = [
    "git_commit",
    "final_config_hash",
    "risk_matrix_hash",
    "fallback_policy_hash",
    "preflight_report_hash",
    "baseline_fairness_hash",
    "teacher_model_id",
    "student_init_model_id",
    "edge_model_name",
    "edge_model_sha256",
    "distill_dataset_hash",
    "teacher_trace_hash",
    "student_probe_trace_hash",
    "counterfactual_repair_trace_hash",
    "quant_behavior_trace_hash",
    "quant_config_hash",
    "planner_model_hash",
    "calibration_snapshot_hash",
    "runtime_state_model_hash",
    "policy_snapshot_hash",
    "graph_model_hash",
    "trust_posterior_snapshot_hash",
    "conflict_gt_audit_hash",
    "conflict_gt_sample_audit_hash",
    "latency_breakdown_hash",
    "online_perception_smoke_hash",
    "data_split_hash",
    "network_profile_hash",
    "prompt_template_hash",
    "policy_config_hash",
    "decision_parser_hash",
    "capability_metric_script_hash",
    "ttft_metric_script_hash",
    "e2e_metric_script_hash",
    "conflict_metric_script_hash",
    "resolution_metric_script_hash",
    "perception_metric_script_hash",
    "communication_metric_script_hash",
    "resource_metric_script_hash",
    "frozen_scu_hash",
    "synthetic_chinese_nlp_hash",
    "fallback_events",
    "timestamp",
]

CONFLICT_REQUIRED_KEYS = [
    "manifest_version",
    "created_by",
    "created_ts",
    "datasets",
    "split",
    "conflict_groups",
    "conflict_group_count",
    "conflict_type_distribution",
    "manifest_hash",
]

CONFLICT_GROUP_REQUIRED_KEYS = [
    "conflict_group_id",
    "event_id",
    "node_ids",
    "conflict_type_gt",
    "global_decision_gt",
    "label_source",
    "source_dataset",
    "time_window_id",
]

CONFLICT_TYPES = {
    "class_conflict",
    "risk_conflict",
    "action_conflict",
    "duplicate_alert",
    "none",
}

LABEL_SOURCES = {
    "cityflow_annotation",
    "derived_rule",
    "manual_review",
    "industrial_label",
}

RISK_ATTRS = {"low", "medium", "high"}
ACTIONS = {"pass", "reject", "inspect", "alert", "ignore", "upload"}
SPLITS = {"train", "validation", "test", "all"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def is_hash_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_hash") or lowered.endswith("_sha256") or lowered == "manifest_hash"


def path_join(base: str, part: str) -> str:
    if base == "$":
        return f"$.{part}"
    return f"{base}.{part}"


def index_join(base: str, index: int) -> str:
    return f"{base}[{index}]"


def is_placeholder_string(value: str) -> bool:
    stripped = value.strip()
    if stripped in PLACEHOLDER_VALUES:
        return True
    return any(stripped.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def check_no_placeholders(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []

    if value is None:
        errors.append(f"{path} must not be null")
        return errors

    if isinstance(value, str):
        if is_placeholder_string(value):
            errors.append(f"{path} contains placeholder value: {value!r}")
        return errors

    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(check_no_placeholders(item, index_join(path, index)))
        return errors

    if isinstance(value, dict):
        for key, item in value.items():
            errors.extend(check_no_placeholders(item, path_join(path, str(key))))
        return errors

    return errors


def check_hash_fields(value: Any, path: str = "$", key: str | None = None) -> list[str]:
    errors: list[str] = []

    if key and is_hash_key(key):
        if not isinstance(value, str):
            errors.append(f"{path} must be a SHA256 string")
        elif not SHA256_RE.match(value):
            errors.append(f"{path} must be a 64-character SHA256 hex value")
        return errors

    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(check_hash_fields(item, index_join(path, index)))
    elif isinstance(value, dict):
        for child_key, item in value.items():
            errors.extend(check_hash_fields(item, path_join(path, str(child_key)), str(child_key)))

    return errors


def require_keys(data: dict[str, Any], required: list[str], path: str) -> list[str]:
    errors: list[str] = []
    for key in required:
        if key not in data:
            errors.append(f"{path} missing required key: {key}")
    return errors


def require_non_negative_int(value: Any, path: str) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return [f"{path} must be a non-negative integer"]
    if value < 0:
        return [f"{path} must be non-negative"]
    return []


def require_boolean(value: Any, path: str) -> list[str]:
    if not isinstance(value, bool):
        return [f"{path} must be boolean"]
    return []


def load_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{path.name} is not valid JSON: {exc}"]

    if not isinstance(data, dict):
        return None, [f"{path.name} must contain a JSON object"]

    return data, []


def validate_dataset_manifest(data: dict[str, Any], strict_gdata: bool) -> list[str]:
    errors: list[str] = []
    errors.extend(require_keys(data, [
        "manifest_version",
        "created_by",
        "created_ts",
        "sampling_seed",
        "datasets",
        "scu_support",
        "global_leakage_check",
    ], "$"))

    if data.get("template_only") is True:
        errors.append("$.template_only must not be true for formal dataset_manifest.json")

    if data.get("manifest_version") != "1.0":
        errors.append("$.manifest_version must be '1.0'")

    if data.get("sampling_seed") != 42:
        errors.append("$.sampling_seed must be 42")

    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append("$.datasets must be a non-empty list")
        return errors

    seen_names: set[str] = set()
    for index, dataset in enumerate(datasets):
        item_path = f"$.datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{item_path} must be an object")
            continue

        errors.extend(require_keys(dataset, DATASET_REQUIRED_KEYS, item_path))
        name = dataset.get("dataset_name")
        if isinstance(name, str):
            seen_names.add(name)

        for count_key in [
            "used_train_count",
            "used_validation_count",
            "used_test_count",
            "teacher_generated_train_count",
        ]:
            if count_key in dataset:
                errors.extend(require_non_negative_int(dataset[count_key], f"{item_path}.{count_key}"))

        if "leakage_check_pass" in dataset:
            errors.extend(require_boolean(dataset["leakage_check_pass"], f"{item_path}.leakage_check_pass"))
            if dataset["leakage_check_pass"] is not True:
                errors.append(f"{item_path}.leakage_check_pass must be true")

        if name == "CityFlow":
            for key in [
                "relation_node_count",
                "relation_edge_count",
                "relation_group_count",
                "conflict_group_count",
                "vehicle_id_count",
            ]:
                if key not in dataset:
                    errors.append(f"{item_path} missing CityFlow key: {key}")
                else:
                    errors.extend(require_non_negative_int(dataset[key], f"{item_path}.{key}"))

            if strict_gdata:
                thresholds = {
                    "conflict_group_count": 50,
                    "relation_edge_count": 200,
                    "relation_group_count": 200,
                }
                for key, minimum in thresholds.items():
                    value = dataset.get(key)
                    if isinstance(value, int) and value < minimum:
                        errors.append(f"{item_path}.{key} must be >= {minimum} for G-DATA")

        if name == "MVTec AD" and "official_test_full_final_gate" in dataset:
            errors.extend(
                require_boolean(
                    dataset["official_test_full_final_gate"],
                    f"{item_path}.official_test_full_final_gate",
                )
            )

        if name == "NEU-DET" and "class_split_counts" not in dataset:
            errors.append(f"{item_path} missing NEU-DET key: class_split_counts")

    missing_datasets = sorted(REQUIRED_DATASETS - seen_names)
    if missing_datasets:
        errors.append(f"$.datasets missing required datasets: {', '.join(missing_datasets)}")

    scu_support = data.get("scu_support")
    if isinstance(scu_support, dict):
        errors.extend(require_keys(scu_support, [
            "scu_sample_count",
            "scu_sample_hash",
            "scu_source",
            "used_for_training",
            "used_for_validation",
            "used_for_final_support",
        ], "$.scu_support"))
        if "scu_sample_count" in scu_support:
            errors.extend(require_non_negative_int(scu_support["scu_sample_count"], "$.scu_support.scu_sample_count"))
        for key, expected in [
            ("used_for_training", False),
            ("used_for_validation", False),
            ("used_for_final_support", True),
        ]:
            if key in scu_support:
                errors.extend(require_boolean(scu_support[key], f"$.scu_support.{key}"))
                if scu_support[key] is not expected:
                    errors.append(f"$.scu_support.{key} must be {expected}")
    else:
        errors.append("$.scu_support must be an object")

    leakage = data.get("global_leakage_check")
    if isinstance(leakage, dict):
        errors.extend(require_keys(leakage, GLOBAL_LEAKAGE_KEYS, "$.global_leakage_check"))
        for key in GLOBAL_LEAKAGE_KEYS:
            if key in leakage:
                errors.extend(require_boolean(leakage[key], f"$.global_leakage_check.{key}"))
                if leakage[key] is not False:
                    errors.append(f"$.global_leakage_check.{key} must be false")
    else:
        errors.append("$.global_leakage_check must be an object")

    return errors


def validate_run_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(require_keys(data, RUN_MANIFEST_REQUIRED_KEYS, "$"))

    if data.get("template_only") is True:
        errors.append("$.template_only must not be true for formal manifest.json")

    git_commit = data.get("git_commit")
    if isinstance(git_commit, str) and not GIT_COMMIT_RE.match(git_commit):
        errors.append("$.git_commit must be a 7-40 character hex commit id")

    if data.get("teacher_model_id") != "Qwen/Qwen2.5-14B-Instruct-AWQ":
        errors.append("$.teacher_model_id must match the implementation plan")
    if data.get("student_init_model_id") != "Qwen/Qwen3-1.7B":
        errors.append("$.student_init_model_id must match the implementation plan")
    if data.get("edge_model_name") != "DB4AI-Edge-P0A3-Qwen3-1.7B-Q3_K_M-CANDIDATE":
        errors.append("$.edge_model_name must match the implementation plan")

    fallback_events = data.get("fallback_events")
    if not isinstance(fallback_events, list):
        errors.append("$.fallback_events must be a list")

    return errors


def validate_conflict_manifest(data: dict[str, Any], strict_gdata: bool) -> list[str]:
    errors: list[str] = []
    errors.extend(require_keys(data, CONFLICT_REQUIRED_KEYS, "$"))

    if data.get("template_only") is True:
        errors.append("$.template_only must not be true for formal conflict_gt_manifest.json")

    if data.get("manifest_version") != "1.0":
        errors.append("$.manifest_version must be '1.0'")

    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append("$.datasets must be a non-empty list")
    else:
        for dataset in ["CityFlow", "MVTec AD", "NEU-DET"]:
            if dataset not in datasets:
                errors.append(f"$.datasets must include {dataset}")

    split = data.get("split")
    if split not in SPLITS:
        errors.append("$.split must be one of train, validation, test, all")

    groups = data.get("conflict_groups")
    if not isinstance(groups, list):
        errors.append("$.conflict_groups must be a list")
        groups = []

    if strict_gdata and len(groups) < 50:
        errors.append("$.conflict_groups must contain at least 50 groups for G-DATA")

    count = data.get("conflict_group_count")
    if "conflict_group_count" in data:
        errors.extend(require_non_negative_int(count, "$.conflict_group_count"))
        if isinstance(count, int) and count != len(groups):
            errors.append("$.conflict_group_count must match len($.conflict_groups)")

    for index, group in enumerate(groups):
        item_path = f"$.conflict_groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{item_path} must be an object")
            continue

        errors.extend(require_keys(group, CONFLICT_GROUP_REQUIRED_KEYS, item_path))

        for key in ["conflict_group_id", "event_id", "time_window_id", "source_dataset"]:
            value = group.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{item_path}.{key} must be a non-empty string")

        node_ids = group.get("node_ids")
        if not isinstance(node_ids, list) or not node_ids:
            errors.append(f"{item_path}.node_ids must be a non-empty list")
        elif any(not isinstance(node_id, str) or not node_id.strip() for node_id in node_ids):
            errors.append(f"{item_path}.node_ids must contain non-empty strings")

        if group.get("conflict_type_gt") not in CONFLICT_TYPES:
            errors.append(f"{item_path}.conflict_type_gt is not allowed")

        if group.get("label_source") not in LABEL_SOURCES:
            errors.append(f"{item_path}.label_source is not allowed")

        decision = group.get("global_decision_gt")
        if isinstance(decision, dict):
            errors.extend(require_keys(decision, ["event_type", "risk_attr", "action"], f"{item_path}.global_decision_gt"))
            event_type = decision.get("event_type")
            if not isinstance(event_type, str) or not event_type.strip():
                errors.append(f"{item_path}.global_decision_gt.event_type must be a non-empty string")
            if decision.get("risk_attr") not in RISK_ATTRS:
                errors.append(f"{item_path}.global_decision_gt.risk_attr is not allowed")
            if decision.get("action") not in ACTIONS:
                errors.append(f"{item_path}.global_decision_gt.action is not allowed")
        else:
            errors.append(f"{item_path}.global_decision_gt must be an object")

    distribution = data.get("conflict_type_distribution")
    if isinstance(distribution, dict):
        total = 0
        for key, value in distribution.items():
            if key not in CONFLICT_TYPES:
                errors.append(f"$.conflict_type_distribution contains unknown type: {key}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"$.conflict_type_distribution.{key} must be a non-negative integer")
            else:
                total += value
        if isinstance(count, int) and total != count:
            errors.append("$.conflict_type_distribution values must sum to conflict_group_count")
    else:
        errors.append("$.conflict_type_distribution must be an object")

    return errors


def validate_file(name: str, path: Path, strict_gdata: bool) -> list[str]:
    data, errors = load_json(path)
    if errors or data is None:
        return errors

    errors.extend(check_no_placeholders(data))
    errors.extend(check_hash_fields(data))

    if name == "dataset":
        errors.extend(validate_dataset_manifest(data, strict_gdata))
    elif name == "run":
        errors.extend(validate_run_manifest(data))
    elif name == "conflict":
        errors.extend(validate_conflict_manifest(data, strict_gdata))
    else:
        errors.append(f"Unknown manifest type: {name}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate formal DB4AI-EdgeServe manifest JSON files."
    )
    parser.add_argument("--dataset-manifest", default=DEFAULT_FILES["dataset"])
    parser.add_argument("--manifest", default=DEFAULT_FILES["run"])
    parser.add_argument("--conflict-gt-manifest", default=DEFAULT_FILES["conflict"])
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success when formal manifest files have not been generated yet.",
    )
    parser.add_argument(
        "--strict-gdata",
        action="store_true",
        help="Apply extra G-DATA count thresholds where the manifest schema exposes them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "dataset": ROOT / args.dataset_manifest,
        "run": ROOT / args.manifest,
        "conflict": ROOT / args.conflict_gt_manifest,
    }

    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        if args.allow_missing:
            print("Formal manifest files are not generated yet:")
            for filename in missing:
                print(f"[SKIP] Missing {filename}")
            return 0

        print("Formal manifest validation failed:")
        for filename in missing:
            fail(f"Missing {filename}")
        return 1

    all_errors: list[str] = []
    for name, path in paths.items():
        errors = validate_file(name, path, args.strict_gdata)
        if errors:
            for error in errors:
                all_errors.append(f"{path.name}: {error}")
        else:
            ok(f"{path.name} passed")

    if all_errors:
        print()
        print("Formal manifest validation failed:")
        for error in all_errors:
            fail(error)
        return 1

    print()
    print("Formal manifest validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
