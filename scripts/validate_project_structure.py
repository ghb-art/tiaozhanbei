from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "configs",
    "scripts",
    "experiments",
    "model_compression",
    "data",
    "data/raw",
    "data/processed",
    "data/splits",
    "data/kwdb",
    "data/distill",
    "data/datasets",
    "models",
    "results",
    "reports",
    "runtime",
    "docs",
    "sql",
    "docker",
    "logs",
]

REQUIRED_FILES = [
    ".gitignore",
    "README.md",
    "IMPLEMENTATION_PLAN.md",
    "docs/REVISION_LOG.md",
    "docs/PROJECT_STATUS.md",
    "docs/TODO.md",
    "docs/DATASET_SOURCES.md",
    "dataset_manifest.template.json",
    "manifest.template.json",
    "conflict_gt_manifest.template.json",
    "dataset_manifest.json",
    "manifest.json",
    "conflict_gt_manifest.json",
    "reports/audit/conflict_gt_audit.csv",
    "reports/audit/conflict_gt_sample_audit.json",
    "reports/audit/gate_db_schema_check.json",
    "reports/audit/gate_db_smoke.csv",
    "reports/audit/gate_cloud_smoke.json",
    "reports/audit/gate_kd_trace_teacher_smoke.json",
    "reports/audit/model_downloads.json",
    "configs/network_profiles.yaml",
    "configs/workload_profiles.yaml",
    "configs/models.yaml",
    "configs/final_config_dev.yaml",
    "scripts/generate_manifest_template.py",
    "scripts/validate_manifest_files.py",
    "scripts/inspect_datasets.py",
    "scripts/validate_dataset_presence.py",
    "scripts/preflight_runtime_smoke.py",
    "scripts/validate_splits.py",
    "scripts/setup_datasets.sh",
    "scripts/build_conflict_gt.py",
    "scripts/download_models.py",
    "scripts/verify_gate_cloud.py",
    "model_compression/generate_teacher_traces.py",
    "scripts/generate_formal_manifests.py",
    "scripts/verify_gate_db.py",
    "sql/cloud_schema.sql",
    "docker/docker-compose.kwdb.yml",
    "docs/DATASET_SPLIT_STRATEGY.md",
    "data/splits/frozen_splits.json",
]

GITIGNORE_PATTERNS = [
    "logs/",
    "runtime/",
    "data/raw/",
    "data/processed/",
    "data/kwdb/",
    "data/distill/",
    "data/datasets/**",
    "!data/datasets/**/.gitkeep",
    "results/",
    "models/pretrained/",
    "models/checkpoints/",
    "models/quantized/",
    "*.gguf",
]

NETWORK_PROFILES = [
    "normal",
    "high_delay",
    "low_bandwidth",
    "high_loss",
    "short_disconnect",
]

WORKLOAD_PROFILES = [
    "stable",
    "burst",
    "replay",
    "scale",
]

MODEL_KEYS = [
    "cloud_teacher",
    "high_edge",
    "edge_student",
]

TEMPLATE_FILES = [
    "dataset_manifest.template.json",
    "manifest.template.json",
    "conflict_gt_manifest.template.json",
]

TEMPLATE_REQUIRED_KEYS = {
    "dataset_manifest.template.json": [
        "manifest_version",
        "created_by",
        "created_ts",
        "sampling_seed",
        "datasets",
        "scu_support",
        "global_leakage_check",
    ],
    "manifest.template.json": [
        "git_commit",
        "final_config_hash",
        "teacher_model_id",
        "student_init_model_id",
        "edge_model_name",
        "fallback_events",
        "timestamp",
    ],
    "conflict_gt_manifest.template.json": [
        "manifest_version",
        "created_by",
        "created_ts",
        "datasets",
        "split",
        "conflict_groups",
        "conflict_group_count",
        "conflict_type_distribution",
        "manifest_hash",
    ],
}


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_paths() -> list[str]:
    errors: list[str] = []

    if (ROOT / ".git").is_dir():
        ok("Git repository exists")
    else:
        errors.append("Missing .git directory. Run git init first.")

    for relative_path in REQUIRED_DIRS:
        path = ROOT / relative_path
        if path.is_dir():
            ok(f"Directory exists: {relative_path}")
        else:
            errors.append(f"Missing directory: {relative_path}")

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if path.is_file():
            ok(f"File exists: {relative_path}")
        else:
            errors.append(f"Missing file: {relative_path}")

    return errors


def check_gitignore() -> list[str]:
    errors: list[str] = []
    gitignore_path = ROOT / ".gitignore"
    if not gitignore_path.is_file():
        return ["Missing .gitignore"]

    text = read_text(".gitignore")
    for pattern in GITIGNORE_PATTERNS:
        if pattern in text:
            ok(f".gitignore contains: {pattern}")
        else:
            errors.append(f".gitignore missing pattern: {pattern}")
    return errors


def check_config_keywords() -> list[str]:
    errors: list[str] = []

    network_text = read_text("configs/network_profiles.yaml")
    for profile in NETWORK_PROFILES:
        token = f"  {profile}:"
        if token in network_text:
            ok(f"Network profile declared: {profile}")
        else:
            errors.append(f"Missing network profile: {profile}")

    workload_text = read_text("configs/workload_profiles.yaml")
    for profile in WORKLOAD_PROFILES:
        token = f"  {profile}:"
        if token in workload_text:
            ok(f"Workload profile declared: {profile}")
        else:
            errors.append(f"Missing workload profile: {profile}")

    models_text = read_text("configs/models.yaml")
    for model_key in MODEL_KEYS:
        token = f"  {model_key}:"
        if token in models_text:
            ok(f"Model role declared: {model_key}")
        else:
            errors.append(f"Missing model role: {model_key}")

    return errors


def check_manifest_templates() -> list[str]:
    errors: list[str] = []

    for relative_path in TEMPLATE_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing manifest template: {relative_path}")
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in {relative_path}: {exc}")
            continue

        ok(f"Manifest template is valid JSON: {relative_path}")

        if data.get("template_only") is True:
            ok(f"Manifest template marked template_only: {relative_path}")
        else:
            errors.append(f"Manifest template missing template_only=true: {relative_path}")

        for key in TEMPLATE_REQUIRED_KEYS[relative_path]:
            if key in data:
                ok(f"Manifest template key present: {relative_path}::{key}")
            else:
                errors.append(f"Manifest template missing key: {relative_path}::{key}")

    return errors


def main() -> int:
    print(f"Project root: {ROOT}")
    errors: list[str] = []
    errors.extend(check_paths())
    errors.extend(check_gitignore())
    errors.extend(check_config_keywords())
    errors.extend(check_manifest_templates())

    if errors:
        print()
        print("Validation failed:")
        for error in errors:
            fail(error)
        return 1

    print()
    print("Project structure validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
