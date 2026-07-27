from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_NOTICE = (
    "Template only. Final G-DATA/Final Gate artifacts must replace all "
    "REPLACE_WITH_* values with measured values and hashes."
)


def dataset_manifest_template() -> dict[str, Any]:
    return {
        "template_only": True,
        "template_notice": TEMPLATE_NOTICE,
        "manifest_version": "1.0",
        "created_by": "scripts/setup_datasets.sh",
        "created_ts": "REPLACE_WITH_UNIX_TIMESTAMP_FLOAT",
        "sampling_seed": 42,
        "datasets": [
            {
                "dataset_name": "GSM8K",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "train 7473 / test 1319",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": 500,
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "HumanEval",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "164 full",
                "used_train_count": 0,
                "used_validation_count": 0,
                "used_test_count": 164,
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": 0,
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "MMLU",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "57 subjects",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": 1000,
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "CMMLU",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "67 subjects",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": 1000,
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "synthetic_chinese_nlp_hash": "REPLACE_WITH_SHA256_OR_NOT_USED",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "MVTec AD",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "train 3629 normal / test 1725",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": 1725,
                "official_test_full_final_gate": "REPLACE_WITH_BOOLEAN",
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "NEU-DET",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "6 classes x 300 = 1800",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": 360,
                "class_split_counts": "REPLACE_WITH_CLASS_SPLIT_COUNTS",
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "CityFlow",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "REPLACE_WITH_DETECTED_SCALE",
                "used_train_count": "REPLACE_WITH_INT",
                "used_validation_count": "REPLACE_WITH_INT",
                "used_test_count": "REPLACE_WITH_INT",
                "relation_node_count": "REPLACE_WITH_INT",
                "relation_edge_count": "REPLACE_WITH_INT",
                "relation_group_count": "REPLACE_WITH_INT",
                "conflict_group_count": "REPLACE_WITH_INT",
                "vehicle_id_count": "REPLACE_WITH_INT",
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": "REPLACE_WITH_INT",
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
            {
                "dataset_name": "UA-DETRAC",
                "dataset_version": "REPLACE_WITH_DATASET_VERSION",
                "official_scale": "100 videos, >140k frames",
                "used_train_count": 0,
                "used_validation_count": 0,
                "used_test_count": "REPLACE_WITH_INT",
                "final_gate_sample_ids_hash": "REPLACE_WITH_SHA256",
                "split_hash": "REPLACE_WITH_SHA256",
                "teacher_generated_train_count": 0,
                "leakage_check_pass": "REPLACE_WITH_BOOLEAN",
            },
        ],
        "scu_support": {
            "scu_sample_count": 300,
            "scu_sample_hash": "REPLACE_WITH_SHA256",
            "scu_source": "independent_support_sources",
            "used_for_training": False,
            "used_for_validation": False,
            "used_for_final_support": True,
        },
        "global_leakage_check": {
            "test_in_distill": False,
            "test_in_planner_train": False,
            "test_in_policy_train": False,
            "test_in_graph_train": False,
            "test_in_calibration": False,
            "test_in_validation": False,
        },
    }


def conflict_gt_manifest_template() -> dict[str, Any]:
    return {
        "template_only": True,
        "template_notice": TEMPLATE_NOTICE,
        "manifest_version": "1.0",
        "created_by": "scripts/build_conflict_gt.py",
        "created_ts": "REPLACE_WITH_UNIX_TIMESTAMP_FLOAT",
        "datasets": ["CityFlow", "MVTec AD", "NEU-DET"],
        "split": "REPLACE_WITH_train_validation_test_OR_all",
        "conflict_groups": [
            {
                "conflict_group_id": "REPLACE_WITH_CONFLICT_GROUP_ID",
                "event_id": "REPLACE_WITH_EVENT_ID",
                "node_ids": ["REPLACE_WITH_NODE_ID"],
                "conflict_type_gt": (
                    "class_conflict|risk_conflict|action_conflict|"
                    "duplicate_alert|none"
                ),
                "global_decision_gt": {
                    "event_type": "REPLACE_WITH_EVENT_TYPE",
                    "risk_attr": "low|medium|high",
                    "action": "pass|reject|inspect|alert|ignore|upload",
                },
                "label_source": (
                    "cityflow_annotation|derived_rule|manual_review|"
                    "industrial_label"
                ),
                "source_dataset": "REPLACE_WITH_SOURCE_DATASET",
                "time_window_id": "REPLACE_WITH_TIME_WINDOW_ID",
            }
        ],
        "conflict_group_count": "REPLACE_WITH_INT",
        "conflict_type_distribution": {
            "class_conflict": "REPLACE_WITH_INT",
            "risk_conflict": "REPLACE_WITH_INT",
            "action_conflict": "REPLACE_WITH_INT",
            "duplicate_alert": "REPLACE_WITH_INT",
            "none": "REPLACE_WITH_INT",
        },
        "manifest_hash": "REPLACE_WITH_SHA256",
    }


def manifest_template() -> dict[str, Any]:
    return {
        "template_only": True,
        "template_notice": TEMPLATE_NOTICE,
        "git_commit": "REPLACE_WITH_GIT_COMMIT",
        "final_config_hash": "REPLACE_WITH_SHA256",
        "risk_matrix_hash": "REPLACE_WITH_SHA256",
        "fallback_policy_hash": "REPLACE_WITH_SHA256",
        "preflight_report_hash": "REPLACE_WITH_SHA256",
        "baseline_fairness_hash": "REPLACE_WITH_SHA256",
        "teacher_model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "student_init_model_id": "Qwen/Qwen3-1.7B",
        "edge_model_name": "DB4AI-Edge-P0A3-Qwen3-1.7B-Q3_K_M-CANDIDATE",
        "edge_model_sha256": "REPLACE_WITH_SHA256",
        "distill_dataset_hash": "REPLACE_WITH_SHA256",
        "teacher_trace_hash": "REPLACE_WITH_SHA256",
        "student_probe_trace_hash": "REPLACE_WITH_SHA256",
        "counterfactual_repair_trace_hash": "REPLACE_WITH_SHA256",
        "quant_behavior_trace_hash": "REPLACE_WITH_SHA256",
        "quant_config_hash": "REPLACE_WITH_SHA256",
        "planner_model_hash": "REPLACE_WITH_SHA256",
        "calibration_snapshot_hash": "REPLACE_WITH_SHA256",
        "runtime_state_model_hash": "REPLACE_WITH_SHA256",
        "policy_snapshot_hash": "REPLACE_WITH_SHA256",
        "graph_model_hash": "REPLACE_WITH_SHA256",
        "trust_posterior_snapshot_hash": "REPLACE_WITH_SHA256",
        "conflict_gt_audit_hash": "REPLACE_WITH_SHA256",
        "conflict_gt_sample_audit_hash": "REPLACE_WITH_SHA256",
        "latency_breakdown_hash": "REPLACE_WITH_SHA256",
        "online_perception_smoke_hash": "REPLACE_WITH_SHA256",
        "data_split_hash": "REPLACE_WITH_SHA256",
        "network_profile_hash": "REPLACE_WITH_SHA256",
        "prompt_template_hash": "REPLACE_WITH_SHA256",
        "policy_config_hash": "REPLACE_WITH_SHA256",
        "decision_parser_hash": "REPLACE_WITH_SHA256",
        "capability_metric_script_hash": "REPLACE_WITH_SHA256",
        "ttft_metric_script_hash": "REPLACE_WITH_SHA256",
        "e2e_metric_script_hash": "REPLACE_WITH_SHA256",
        "conflict_metric_script_hash": "REPLACE_WITH_SHA256",
        "resolution_metric_script_hash": "REPLACE_WITH_SHA256",
        "perception_metric_script_hash": "REPLACE_WITH_SHA256",
        "communication_metric_script_hash": "REPLACE_WITH_SHA256",
        "resource_metric_script_hash": "REPLACE_WITH_SHA256",
        "frozen_scu_hash": "REPLACE_WITH_SHA256",
        "synthetic_chinese_nlp_hash": "REPLACE_WITH_SHA256_OR_NOT_USED",
        "fallback_events": [],
        "timestamp": "REPLACE_WITH_ISO8601_TIMESTAMP",
    }


TEMPLATES = {
    "dataset_manifest.template.json": dataset_manifest_template,
    "manifest.template.json": manifest_template,
    "conflict_gt_manifest.template.json": conflict_gt_manifest_template,
}


def write_template(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DB4AI-EdgeServe manifest template JSON files."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory relative to project root where templates will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing template files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, factory in TEMPLATES.items():
        output_path = output_dir / filename
        write_template(output_path, factory(), args.overwrite)
        print(f"Wrote {output_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
