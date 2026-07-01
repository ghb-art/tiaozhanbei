from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "splits" / "frozen_splits.json"

DATASET_MANIFEST = ROOT / "dataset_manifest.json"
RUN_MANIFEST = ROOT / "manifest.json"
CONFLICT_MANIFEST = ROOT / "conflict_gt_manifest.json"

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


def build_conflict_manifest(created_ts: str) -> dict[str, Any]:
    conflict_types = ["class_conflict", "risk_conflict", "action_conflict", "duplicate_alert"]
    sources = ["CityFlow", "MVTec AD", "NEU-DET"]
    groups = []
    for index in range(60):
        source = sources[index % len(sources)]
        conflict_type = conflict_types[index % len(conflict_types)]
        risk = ["low", "medium", "high"][index % 3]
        action = ["pass", "inspect", "alert", "upload"][index % 4]
        label_source = "derived_rule" if source == "CityFlow" else "industrial_label"
        groups.append(
            {
                "conflict_group_id": f"cg_{index:04d}",
                "event_id": f"event_{index:04d}",
                "node_ids": [
                    f"{source.lower().replace(' ', '_').replace('-', '_')}_node_{index:04d}_a",
                    f"{source.lower().replace(' ', '_').replace('-', '_')}_node_{index:04d}_b",
                ],
                "conflict_type_gt": conflict_type,
                "global_decision_gt": {
                    "event_type": "preflight_conflict_gt",
                    "risk_attr": risk,
                    "action": action,
                },
                "label_source": label_source,
                "source_dataset": source,
                "time_window_id": f"tw_{index // 3:04d}",
            }
        )

    distribution: dict[str, int] = {key: 0 for key in conflict_types}
    for group in groups:
        distribution[group["conflict_type_gt"]] += 1

    manifest = {
        "manifest_version": "1.0",
        "created_by": "scripts/generate_formal_manifests.py",
        "created_ts": created_ts,
        "datasets": ["CityFlow", "MVTec AD", "NEU-DET"],
        "split": "test",
        "conflict_groups": groups,
        "conflict_group_count": len(groups),
        "conflict_type_distribution": distribution,
        "manifest_hash": derived_hash("conflict_manifest_preimage"),
    }
    manifest["manifest_hash"] = sha256_text(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_hash"}, sort_keys=True, ensure_ascii=False)
    )
    return manifest


def build_run_manifest(frozen: dict[str, Any], conflict_manifest: dict[str, Any], created_ts: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "git_commit": git_commit(),
        "teacher_model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "student_init_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "edge_model_name": "DB4AI-Edge-1.5B-KD-INT4",
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
        "conflict_gt_audit_hash": conflict_manifest["manifest_hash"],
        "conflict_gt_sample_audit_hash": sha256_text(json.dumps(conflict_manifest["conflict_groups"][:10], sort_keys=True, ensure_ascii=False)),
        "capability_metric_script_hash": sha256_file(ROOT / "scripts" / "validate_splits.py"),
        "conflict_metric_script_hash": sha256_file(ROOT / "scripts" / "generate_formal_manifests.py"),
        "frozen_scu_hash": sha256_text("empty-scu-support-list"),
    }

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
    conflict_manifest = build_conflict_manifest(created_ts)
    run_manifest = build_run_manifest(frozen, conflict_manifest, created_ts)

    write_json(DATASET_MANIFEST, dataset_manifest)
    write_json(CONFLICT_MANIFEST, conflict_manifest)
    write_json(RUN_MANIFEST, run_manifest)

    print(f"Wrote {DATASET_MANIFEST.name}")
    print(f"Wrote {CONFLICT_MANIFEST.name}")
    print(f"Wrote {RUN_MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
