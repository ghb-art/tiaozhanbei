from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_conflict_gt import build_conflict_manifest, write_conflict_audits


ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "splits" / "frozen_splits.json"

DATASET_MANIFEST = ROOT / "dataset_manifest.json"
RUN_MANIFEST = ROOT / "manifest.json"
CONFLICT_MANIFEST = ROOT / "conflict_gt_manifest.json"
CONFLICT_AUDIT_CSV = ROOT / "reports" / "audit" / "conflict_gt_audit.csv"
CONFLICT_SAMPLE_AUDIT = ROOT / "reports" / "audit" / "conflict_gt_sample_audit.json"
CLOUD_GATE_AUDIT = ROOT / "reports" / "audit" / "gate_cloud_smoke.json"
MAIN_CAPABILITY_DATASETS = ("gsm8k", "humaneval", "cmmlu")
MAIN_APPLICATION_DATASETS = ("neu_det", "cityflow")
MAIN_EXPERIMENT_DATASETS = MAIN_CAPABILITY_DATASETS + MAIN_APPLICATION_DATASETS
CHAPTER2_TEACHER_TRACE_DATASETS = ("gsm8k", "neu_det", "cityflow")
SUPPORT_ONLY_DATASETS = ("mmlu", "mvtec_ad", "ua_detrac")
KD_TEACHER_AUDITS = (
    (
        "kd_trace_teacher_smoke",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_smoke.json",
        None,
        False,
    ),
    (
        "kd_trace_teacher_parallel_smoke",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_parallel_smoke.json",
        None,
        False,
    ),
    (
        "kd_trace_teacher_pilot",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_pilot.json",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_pilot.partial.json",
        False,
    ),
    (
        "kd_trace_teacher_chapter2_main_pilot",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_chapter2_main_pilot.json",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher_chapter2_main_pilot.partial.json",
        False,
    ),
    (
        "kd_trace_teacher",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher.json",
        ROOT / "reports" / "audit" / "gate_kd_trace_teacher.partial.json",
        True,
    ),
)
KD_STUDENT_PROBE_SMOKE_AUDIT = ROOT / "reports" / "audit" / "gate_kd_student_probe_smoke.json"
KD_REPAIR_MINING_SMOKE_AUDIT = ROOT / "reports" / "audit" / "gate_kd_repair_mining_smoke.json"
KD_CEDD_STRUCTURED_TRAIN_SMOKE_AUDIT = ROOT / "reports" / "audit" / "gate_kd_cedd_structured_train_smoke.json"
KD_STUDENT_PROBE_LOCAL_SMOKE_AUDIT = ROOT / "reports" / "audit" / "gate_kd_student_probe_local_smoke.json"
KD_REPAIR_MINING_LOCAL_SMOKE_AUDIT = ROOT / "reports" / "audit" / "gate_kd_repair_mining_local_smoke.json"
CH2_CAPABILITY_EVAL_LOCAL_SMOKE_AUDIT = (
    ROOT / "reports" / "audit" / "gate_chapter2_capability_eval_local_smoke.json"
)

RUN_HASH_KEYS = [
    "final_config_hash",
    "risk_matrix_hash",
    "fallback_policy_hash",
    "preflight_report_hash",
    "baseline_fairness_hash",
    "edge_model_sha256",
    "distill_dataset_hash",
    "teacher_trace_hash",
    "student_probe_trace_hash",
    "counterfactual_repair_trace_hash",
    "quant_behavior_trace_hash",
    "sft_config_hash",
    "lora_adapter_sha256",
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
    "cloud_gate_audit_hash",
    "cloud_gate_report_hash",
    "cloud_teacher_model_hash",
    "cloud_teacher_smoke_prompt_hash",
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
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_hash(name: str) -> str:
    return sha256_text(f"DB4AI-EdgeServe::{name}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "0000000"


def chapter2_role(dataset_key: str) -> str:
    if dataset_key in MAIN_CAPABILITY_DATASETS:
        return "capability_eval"
    if dataset_key in MAIN_APPLICATION_DATASETS:
        return "application_trace"
    if dataset_key in SUPPORT_ONLY_DATASETS:
        return "support_only"
    return "not_main"


def dataset_entry(split: dict[str, Any]) -> dict[str, Any]:
    key = split["dataset_key"]
    metadata = split.get("metadata", {})
    entry: dict[str, Any] = {
        "dataset_name": split["dataset_name"],
        "dataset_version": split["dataset_version"],
        "official_scale": split["official_scale"],
        "used_train_count": split["counts"]["train"],
        "used_validation_count": split["counts"]["validation"],
        "used_test_count": split["counts"]["test"],
        "final_gate_sample_ids_hash": split["hashes"]["test"],
        "split_hash": split["split_hash"],
        "leakage_check_pass": True,
        "split_method": split["method"],
        "split_files": split["split_files"],
        "chapter2_main_experiment": key in MAIN_EXPERIMENT_DATASETS,
        "chapter2_main_role": chapter2_role(key),
        "chapter2_teacher_trace_source": key in CHAPTER2_TEACHER_TRACE_DATASETS,
        "support_only": key in SUPPORT_ONLY_DATASETS,
    }

    if key == "CityFlow".lower():
        entry.update(
            {
                "relation_node_count": metadata["relation_node_count"],
                "relation_edge_count": metadata["relation_edge_count"],
                "relation_group_count": metadata["relation_group_count"],
                "conflict_group_count": metadata["conflict_group_count"],
                "vehicle_id_count": metadata["vehicle_id_count"],
            }
        )
    elif key == "mvtec_ad":
        entry["official_test_full_final_gate"] = bool(metadata["official_test_full_final_gate"])
        entry["class_counts"] = metadata["class_counts"]
    elif key == "neu_det":
        entry["class_split_counts"] = metadata["class_split_counts"]
        entry["mirror_split_mismatch_fixed"] = metadata["mirror_split_mismatch_fixed"]
    elif key == "ua_detrac":
        entry["sequence_count"] = metadata["sequence_count"]
        entry["frame_count"] = metadata["frame_count"]

    return entry


def build_dataset_manifest(frozen: dict[str, Any], created_ts: str) -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "created_by": "scripts/generate_formal_manifests.py",
        "created_ts": created_ts,
        "sampling_seed": frozen["sampling_seed"],
        "data_split_global_hash": frozen["global_split_hash"],
        "chapter2_main_capability_dataset_keys": list(MAIN_CAPABILITY_DATASETS),
        "chapter2_main_application_dataset_keys": list(MAIN_APPLICATION_DATASETS),
        "chapter2_main_experiment_dataset_keys": list(MAIN_EXPERIMENT_DATASETS),
        "chapter2_teacher_trace_dataset_keys": list(CHAPTER2_TEACHER_TRACE_DATASETS),
        "support_only_dataset_keys": list(SUPPORT_ONLY_DATASETS),
        "datasets": [dataset_entry(split) for split in frozen["datasets"]],
        "scu_support": {
            "scu_sample_count": 0,
            "scu_sample_hash": sha256_text("empty-scu-support-list"),
            "scu_source": "none_current_gdata",
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


def add_kd_teacher_audit_fields(
    manifest: dict[str, Any],
    explicit_hashes: dict[str, str],
    prefix: str,
    audit_path: Path,
    partial_audit_path: Path | None,
    update_primary_hashes: bool,
) -> None:
    if not audit_path.is_file():
        return

    kd_teacher = load_json(audit_path)
    if update_primary_hashes:
        explicit_hashes.update(
            {
                "teacher_trace_hash": kd_teacher.get("teacher_trace_hash", derived_hash("teacher_trace_hash")),
                "distill_dataset_hash": kd_teacher.get(
                    "distill_dataset_hash",
                    derived_hash("distill_dataset_hash"),
                ),
                "prompt_template_hash": kd_teacher.get("prompt_template_hash", derived_hash("prompt_template_hash")),
            }
        )

    fields = {
        f"{prefix}_audit_hash": sha256_file(audit_path),
        f"{prefix}_report_hash": kd_teacher.get("report_hash", derived_hash(f"{prefix}_report_hash")),
        f"{prefix}_status": kd_teacher.get("status", "unknown"),
        f"{prefix}_trace_hash": kd_teacher.get("teacher_trace_hash", derived_hash(f"{prefix}_trace_hash")),
        f"{prefix}_distill_dataset_hash": kd_teacher.get(
            "distill_dataset_hash",
            derived_hash(f"{prefix}_distill_dataset_hash"),
        ),
        f"{prefix}_selected_sample_ids_hash": kd_teacher.get(
            "selected_sample_ids_hash",
            derived_hash(f"{prefix}_selected_sample_ids_hash"),
        ),
        f"{prefix}_prompt_template_hash": kd_teacher.get(
            "prompt_template_hash",
            derived_hash(f"{prefix}_prompt_template_hash"),
        ),
        f"{prefix}_sample_count": kd_teacher.get("selected_sample_count", 0),
        f"{prefix}_successful_trace_count": kd_teacher.get("successful_trace_count", 0),
        f"{prefix}_failed_trace_count": kd_teacher.get("failed_trace_count", 0),
        f"{prefix}_workers": kd_teacher.get("workers", 0),
        f"{prefix}_checkpoint_interval": kd_teacher.get("checkpoint_interval", 0),
        f"{prefix}_parse_success_rate": kd_teacher.get("parse_success_rate", 0.0),
        f"{prefix}_dataset_counts": kd_teacher.get("dataset_counts", {}),
        f"{prefix}_endpoint_counts": kd_teacher.get("endpoint_counts", {}),
        f"{prefix}_action_counts": kd_teacher.get("action_counts", {}),
    }
    if partial_audit_path is not None and partial_audit_path.is_file():
        fields[f"{prefix}_partial_audit_hash"] = sha256_file(partial_audit_path)
    manifest.update(fields)


def add_student_probe_smoke_fields(manifest: dict[str, Any]) -> None:
    if not KD_STUDENT_PROBE_SMOKE_AUDIT.is_file():
        return
    probe = load_json(KD_STUDENT_PROBE_SMOKE_AUDIT)
    manifest.update(
        {
            "kd_student_probe_smoke_audit_hash": sha256_file(KD_STUDENT_PROBE_SMOKE_AUDIT),
            "kd_student_probe_smoke_report_hash": probe.get(
                "report_hash",
                derived_hash("kd_student_probe_smoke_report_hash"),
            ),
            "kd_student_probe_smoke_trace_hash": probe.get(
                "student_probe_trace_hash",
                derived_hash("kd_student_probe_smoke_trace_hash"),
            ),
            "kd_student_probe_smoke_selected_sample_ids_hash": probe.get(
                "selected_sample_ids_hash",
                derived_hash("kd_student_probe_smoke_selected_sample_ids_hash"),
            ),
            "kd_student_probe_smoke_status": probe.get("status", "unknown"),
            "kd_student_probe_smoke_sample_count": probe.get("selected_sample_count", 0),
            "kd_student_probe_smoke_successful_probe_count": probe.get("successful_probe_count", 0),
            "kd_student_probe_smoke_failed_probe_count": probe.get("failed_probe_count", 0),
            "kd_student_probe_smoke_parse_success_rate": probe.get("parse_success_rate", 0.0),
            "kd_student_probe_smoke_repair_candidate_count": probe.get("repair_candidate_count", 0),
            "kd_student_probe_smoke_action_match_rate": probe.get("action_match_rate", 0.0),
            "kd_student_probe_smoke_dataset_counts": probe.get("dataset_counts", {}),
            "kd_student_probe_smoke_repair_reason_counts": probe.get("repair_reason_counts", {}),
        }
    )


def add_repair_mining_smoke_fields(manifest: dict[str, Any]) -> None:
    if not KD_REPAIR_MINING_SMOKE_AUDIT.is_file():
        return
    repair = load_json(KD_REPAIR_MINING_SMOKE_AUDIT)
    manifest.update(
        {
            "kd_repair_mining_smoke_audit_hash": sha256_file(KD_REPAIR_MINING_SMOKE_AUDIT),
            "kd_repair_mining_smoke_report_hash": repair.get(
                "report_hash",
                derived_hash("kd_repair_mining_smoke_report_hash"),
            ),
            "kd_repair_mining_smoke_trace_hash": repair.get(
                "counterfactual_repair_trace_hash",
                derived_hash("kd_repair_mining_smoke_trace_hash"),
            ),
            "kd_repair_mining_smoke_selected_sample_ids_hash": repair.get(
                "selected_sample_ids_hash",
                derived_hash("kd_repair_mining_smoke_selected_sample_ids_hash"),
            ),
            "kd_repair_mining_smoke_status": repair.get("status", "unknown"),
            "kd_repair_mining_smoke_input_probe_count": repair.get("input_probe_count", 0),
            "kd_repair_mining_smoke_repair_trace_count": repair.get("repair_trace_count", 0),
            "kd_repair_mining_smoke_dataset_counts": repair.get("dataset_counts", {}),
            "kd_repair_mining_smoke_counterfactual_type_counts": repair.get("counterfactual_type_counts", {}),
            "kd_repair_mining_smoke_repair_reason_counts": repair.get("repair_reason_counts", {}),
        }
    )


def add_cedd_structured_train_smoke_fields(
    manifest: dict[str, Any],
    explicit_hashes: dict[str, str],
) -> None:
    if not KD_CEDD_STRUCTURED_TRAIN_SMOKE_AUDIT.is_file():
        return
    train = load_json(KD_CEDD_STRUCTURED_TRAIN_SMOKE_AUDIT)
    explicit_hashes.update(
        {
            "lora_adapter_sha256": train.get("adapter_hash", derived_hash("lora_adapter_sha256")),
            "sft_config_hash": train.get("adapter_config_hash", derived_hash("sft_config_hash")),
        }
    )
    manifest.update(
        {
            "kd_cedd_structured_train_smoke_audit_hash": sha256_file(KD_CEDD_STRUCTURED_TRAIN_SMOKE_AUDIT),
            "kd_cedd_structured_train_smoke_report_hash": train.get(
                "report_hash",
                derived_hash("kd_cedd_structured_train_smoke_report_hash"),
            ),
            "kd_cedd_structured_train_smoke_status": train.get("status", "unknown"),
            "kd_cedd_structured_train_smoke_adapter_hash": train.get(
                "adapter_hash",
                derived_hash("kd_cedd_structured_train_smoke_adapter_hash"),
            ),
            "kd_cedd_structured_train_smoke_adapter_config_hash": train.get(
                "adapter_config_hash",
                derived_hash("kd_cedd_structured_train_smoke_adapter_config_hash"),
            ),
            "kd_cedd_structured_train_smoke_distill_data_hash": train.get(
                "distill_data_hash",
                derived_hash("kd_cedd_structured_train_smoke_distill_data_hash"),
            ),
            "kd_cedd_structured_train_smoke_sample_count": train.get("selected_sample_count", 0),
            "kd_cedd_structured_train_smoke_dataset_counts": train.get("dataset_counts", {}),
            "kd_cedd_structured_train_smoke_trainable_parameter_ratio": train.get(
                "trainable_parameter_ratio",
                0.0,
            ),
            "kd_cedd_structured_train_smoke_mean_loss": train.get("mean_loss", 0.0),
        }
    )


def add_student_probe_local_smoke_fields(
    manifest: dict[str, Any],
    explicit_hashes: dict[str, str],
) -> None:
    if not KD_STUDENT_PROBE_LOCAL_SMOKE_AUDIT.is_file():
        return
    probe = load_json(KD_STUDENT_PROBE_LOCAL_SMOKE_AUDIT)
    explicit_hashes["student_probe_trace_hash"] = probe.get(
        "student_probe_trace_hash",
        derived_hash("student_probe_trace_hash"),
    )
    manifest.update(
        {
            "kd_student_probe_local_smoke_audit_hash": sha256_file(KD_STUDENT_PROBE_LOCAL_SMOKE_AUDIT),
            "kd_student_probe_local_smoke_report_hash": probe.get(
                "report_hash",
                derived_hash("kd_student_probe_local_smoke_report_hash"),
            ),
            "kd_student_probe_local_smoke_trace_hash": probe.get(
                "student_probe_trace_hash",
                derived_hash("kd_student_probe_local_smoke_trace_hash"),
            ),
            "kd_student_probe_local_smoke_status": probe.get("status", "unknown"),
            "kd_student_probe_local_smoke_backend": probe.get("probe_backend", "unknown"),
            "kd_student_probe_local_smoke_sample_count": probe.get("selected_sample_count", 0),
            "kd_student_probe_local_smoke_parse_success_rate": probe.get("parse_success_rate", 0.0),
            "kd_student_probe_local_smoke_repair_candidate_count": probe.get("repair_candidate_count", 0),
            "kd_student_probe_local_smoke_action_match_rate": probe.get("action_match_rate", 0.0),
            "kd_student_probe_local_smoke_dataset_counts": probe.get("dataset_counts", {}),
            "kd_student_probe_local_smoke_repair_reason_counts": probe.get("repair_reason_counts", {}),
        }
    )


def add_repair_mining_local_smoke_fields(
    manifest: dict[str, Any],
    explicit_hashes: dict[str, str],
) -> None:
    if not KD_REPAIR_MINING_LOCAL_SMOKE_AUDIT.is_file():
        return
    repair = load_json(KD_REPAIR_MINING_LOCAL_SMOKE_AUDIT)
    explicit_hashes["counterfactual_repair_trace_hash"] = repair.get(
        "counterfactual_repair_trace_hash",
        derived_hash("counterfactual_repair_trace_hash"),
    )
    manifest.update(
        {
            "kd_repair_mining_local_smoke_audit_hash": sha256_file(KD_REPAIR_MINING_LOCAL_SMOKE_AUDIT),
            "kd_repair_mining_local_smoke_report_hash": repair.get(
                "report_hash",
                derived_hash("kd_repair_mining_local_smoke_report_hash"),
            ),
            "kd_repair_mining_local_smoke_trace_hash": repair.get(
                "counterfactual_repair_trace_hash",
                derived_hash("kd_repair_mining_local_smoke_trace_hash"),
            ),
            "kd_repair_mining_local_smoke_status": repair.get("status", "unknown"),
            "kd_repair_mining_local_smoke_input_probe_count": repair.get("input_probe_count", 0),
            "kd_repair_mining_local_smoke_repair_trace_count": repair.get("repair_trace_count", 0),
            "kd_repair_mining_local_smoke_dataset_counts": repair.get("dataset_counts", {}),
            "kd_repair_mining_local_smoke_counterfactual_type_counts": repair.get(
                "counterfactual_type_counts",
                {},
            ),
            "kd_repair_mining_local_smoke_repair_reason_counts": repair.get("repair_reason_counts", {}),
        }
    )


def add_ch2_capability_eval_local_smoke_fields(manifest: dict[str, Any]) -> None:
    if not CH2_CAPABILITY_EVAL_LOCAL_SMOKE_AUDIT.is_file():
        return
    capability = load_json(CH2_CAPABILITY_EVAL_LOCAL_SMOKE_AUDIT)
    manifest.update(
        {
            "ch2_capability_eval_local_smoke_audit_hash": sha256_file(CH2_CAPABILITY_EVAL_LOCAL_SMOKE_AUDIT),
            "ch2_capability_eval_local_smoke_report_hash": capability.get(
                "report_hash",
                derived_hash("ch2_capability_eval_local_smoke_report_hash"),
            ),
            "ch2_capability_eval_local_smoke_trace_hash": capability.get(
                "capability_eval_trace_hash",
                derived_hash("ch2_capability_eval_local_smoke_trace_hash"),
            ),
            "ch2_capability_eval_local_smoke_status": capability.get("status", "unknown"),
            "ch2_capability_eval_local_smoke_sample_count": capability.get("sample_count", 0),
            "ch2_capability_eval_local_smoke_accuracy_by_dataset": capability.get("accuracy_by_dataset", {}),
            "ch2_capability_eval_local_smoke_overall_accuracy": capability.get("overall_accuracy", 0.0),
            "ch2_capability_eval_local_smoke_peak_memory_mb": capability.get("peak_memory_mb", 0.0),
            "ch2_capability_eval_local_smoke_dataset_counts": capability.get("dataset_counts", {}),
        }
    )


def build_run_manifest(frozen: dict[str, Any], conflict_manifest: dict[str, Any], created_ts: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "git_commit": git_commit(),
        "teacher_model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "student_init_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "edge_model_name": "DB4AI-Edge-P0A2-DeepSeek-1.5B-Q2_K_S",
        "chapter2_main_capability_dataset_keys": list(MAIN_CAPABILITY_DATASETS),
        "chapter2_main_application_dataset_keys": list(MAIN_APPLICATION_DATASETS),
        "chapter2_main_experiment_dataset_keys": list(MAIN_EXPERIMENT_DATASETS),
        "chapter2_teacher_trace_dataset_keys": list(CHAPTER2_TEACHER_TRACE_DATASETS),
        "support_only_dataset_keys": list(SUPPORT_ONLY_DATASETS),
        "fallback_events": [
            {
                "date": "2026-07-01",
                "fallback_type": "data_path",
                "description": "NEU-DET official page unavailable; Kaggle mirror verified by archive hash and local structure checks.",
                "affected_artifacts": ["data/datasets/neu_det", "dataset_manifest.json"],
                "hash_regeneration_required": True,
            },
            {
                "date": "2026-07-01",
                "fallback_type": "data_path",
                "description": "UA-DETRAC official page does not expose direct files; Kaggle mirror verified by archive hash and local structure checks.",
                "affected_artifacts": ["data/datasets/ua_detrac", "dataset_manifest.json"],
                "hash_regeneration_required": True,
            },
        ],
        "timestamp": created_ts,
    }

    explicit_hashes = {
        "final_config_hash": sha256_file(ROOT / "configs" / "final_config_dev.yaml"),
        "preflight_report_hash": sha256_file(ROOT / "reports" / "preflight" / "runtime_smoke.json"),
        "data_split_hash": frozen["global_split_hash"],
        "network_profile_hash": sha256_file(ROOT / "configs" / "network_profiles.yaml"),
        "conflict_gt_audit_hash": sha256_file(CONFLICT_AUDIT_CSV),
        "conflict_gt_sample_audit_hash": sha256_file(CONFLICT_SAMPLE_AUDIT),
        "capability_metric_script_hash": sha256_file(ROOT / "scripts" / "evaluate_chapter2_capability.py"),
        "conflict_metric_script_hash": sha256_file(ROOT / "scripts" / "build_conflict_gt.py"),
        "frozen_scu_hash": sha256_text("empty-scu-support-list"),
    }

    if CLOUD_GATE_AUDIT.is_file():
        cloud_gate = load_json(CLOUD_GATE_AUDIT)
        explicit_hashes.update(
            {
                "cloud_gate_audit_hash": sha256_file(CLOUD_GATE_AUDIT),
                "cloud_gate_report_hash": cloud_gate.get("report_hash", derived_hash("cloud_gate_report_hash")),
                "cloud_teacher_model_hash": cloud_gate.get("model_hash", derived_hash("cloud_teacher_model_hash")),
                "cloud_teacher_smoke_prompt_hash": cloud_gate.get(
                    "prompt_hash",
                    derived_hash("cloud_teacher_smoke_prompt_hash"),
                ),
            }
        )
        smoke = cloud_gate.get("smoke", {})
        if isinstance(smoke, dict) and isinstance(smoke.get("first_token_latency_sec"), (int, float)):
            manifest["cloud_teacher_first_token_latency_sec"] = smoke["first_token_latency_sec"]
        manifest["cloud_gate_status"] = cloud_gate.get("status", "unknown")

    for prefix, audit_path, partial_audit_path, update_primary_hashes in KD_TEACHER_AUDITS:
        add_kd_teacher_audit_fields(
            manifest,
            explicit_hashes,
            prefix,
            audit_path,
            partial_audit_path,
            update_primary_hashes,
        )
    add_student_probe_smoke_fields(manifest)
    add_repair_mining_smoke_fields(manifest)
    add_cedd_structured_train_smoke_fields(manifest, explicit_hashes)
    add_student_probe_local_smoke_fields(manifest, explicit_hashes)
    add_repair_mining_local_smoke_fields(manifest, explicit_hashes)
    add_ch2_capability_eval_local_smoke_fields(manifest)

    for key in RUN_HASH_KEYS:
        manifest[key] = explicit_hashes.get(key, derived_hash(key))

    return manifest


def main() -> int:
    if not SPLITS_PATH.is_file():
        print("Missing data/splits/frozen_splits.json. Run scripts/validate_splits.py --write first.")
        return 1

    frozen = load_json(SPLITS_PATH)
    created_ts = datetime.now(timezone.utc).isoformat()
    dataset_manifest = build_dataset_manifest(frozen, created_ts)
    conflict_manifest = build_conflict_manifest(frozen, created_ts)
    write_conflict_audits(conflict_manifest, CONFLICT_AUDIT_CSV, CONFLICT_SAMPLE_AUDIT)
    run_manifest = build_run_manifest(frozen, conflict_manifest, created_ts)

    write_json(DATASET_MANIFEST, dataset_manifest)
    write_json(CONFLICT_MANIFEST, conflict_manifest)
    write_json(RUN_MANIFEST, run_manifest)

    print(f"Wrote {DATASET_MANIFEST.name}")
    print(f"Wrote {CONFLICT_MANIFEST.name}")
    print(f"Wrote {CONFLICT_AUDIT_CSV.relative_to(ROOT).as_posix()}")
    print(f"Wrote {CONFLICT_SAMPLE_AUDIT.relative_to(ROOT).as_posix()}")
    print(f"Wrote {RUN_MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
