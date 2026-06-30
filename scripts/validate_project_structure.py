from __future__ import annotations

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
    "configs/network_profiles.yaml",
    "configs/workload_profiles.yaml",
    "configs/models.yaml",
    "configs/final_config_dev.yaml",
]

GITIGNORE_PATTERNS = [
    "logs/",
    "runtime/",
    "data/raw/",
    "data/processed/",
    "data/kwdb/",
    "data/distill/",
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


def main() -> int:
    print(f"Project root: {ROOT}")
    errors: list[str] = []
    errors.extend(check_paths())
    errors.extend(check_gitignore())
    errors.extend(check_config_keywords())

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
