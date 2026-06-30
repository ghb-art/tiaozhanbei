from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = ROOT / "data" / "datasets"
DEFAULT_OUTPUT = ROOT / "reports" / "preflight" / "data_inventory.json"

EXPECTED_DATASETS = [
    {
        "dataset_key": "gsm8k",
        "dataset_name": "GSM8K",
        "relative_path": "data/datasets/gsm8k",
        "purpose": "Math capability evaluation and distillation train source.",
    },
    {
        "dataset_key": "humaneval",
        "dataset_name": "HumanEval",
        "relative_path": "data/datasets/humaneval",
        "purpose": "Code capability final evaluation.",
    },
    {
        "dataset_key": "mmlu",
        "dataset_name": "MMLU",
        "relative_path": "data/datasets/mmlu",
        "purpose": "English NLP capability evaluation.",
    },
    {
        "dataset_key": "cmmlu",
        "dataset_name": "CMMLU",
        "relative_path": "data/datasets/cmmlu",
        "purpose": "Chinese NLP capability evaluation.",
    },
    {
        "dataset_key": "mvtec_ad",
        "dataset_name": "MVTec AD",
        "relative_path": "data/datasets/mvtec_ad",
        "purpose": "Industrial defect detection support and final evaluation.",
    },
    {
        "dataset_key": "neu_det",
        "dataset_name": "NEU-DET",
        "relative_path": "data/datasets/neu_det",
        "purpose": "Industrial defect detection auxiliary data.",
    },
    {
        "dataset_key": "cityflow",
        "dataset_name": "CityFlow",
        "relative_path": "data/datasets/cityflow",
        "purpose": "Traffic multi-camera relation and conflict evaluation.",
    },
    {
        "dataset_key": "ua_detrac",
        "dataset_name": "UA-DETRAC",
        "relative_path": "data/datasets/ua_detrac",
        "purpose": "Traffic detection and tracking auxiliary evaluation.",
    },
]

IGNORED_PAYLOAD_FILES = {".gitkeep"}
MAX_LIST_ITEMS = 20


def to_relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iso_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": to_relative(path),
        "size_bytes": stat.st_size,
        "modified_ts": iso_from_timestamp(stat.st_mtime),
    }


def inspect_dataset(entry: dict[str, str]) -> dict[str, Any]:
    path = ROOT / entry["relative_path"]
    result: dict[str, Any] = {
        "dataset_key": entry["dataset_key"],
        "dataset_name": entry["dataset_name"],
        "path": entry["relative_path"],
        "purpose": entry["purpose"],
        "exists": path.exists(),
        "status": "missing",
        "file_count": 0,
        "payload_file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "payload_bytes": 0,
        "extensions": {},
        "top_level_entries": [],
        "sample_payload_files": [],
        "last_modified_ts": None,
    }

    if not path.exists():
        return result

    top_level_entries = sorted(
        child.name + ("/" if child.is_dir() else "")
        for child in path.iterdir()
        if child.name not in IGNORED_PAYLOAD_FILES
    )
    result["top_level_entries"] = top_level_entries[:MAX_LIST_ITEMS]

    file_count = 0
    payload_file_count = 0
    directory_count = 0
    total_bytes = 0
    payload_bytes = 0
    latest_mtime: float | None = None
    extensions: Counter[str] = Counter()
    sample_payload_files: list[dict[str, Any]] = []

    for child in path.rglob("*"):
        try:
            stat = child.stat()
        except OSError:
            continue

        latest_mtime = stat.st_mtime if latest_mtime is None else max(latest_mtime, stat.st_mtime)

        if child.is_dir():
            directory_count += 1
            continue

        if not child.is_file():
            continue

        file_count += 1
        total_bytes += stat.st_size

        if child.name in IGNORED_PAYLOAD_FILES:
            continue

        payload_file_count += 1
        payload_bytes += stat.st_size
        extension = child.suffix.lower() if child.suffix else "<no_ext>"
        extensions[extension] += 1
        if len(sample_payload_files) < MAX_LIST_ITEMS:
            sample_payload_files.append(file_info(child))

    result.update(
        {
            "status": "has_payload" if payload_file_count else "empty",
            "file_count": file_count,
            "payload_file_count": payload_file_count,
            "directory_count": directory_count,
            "total_bytes": total_bytes,
            "payload_bytes": payload_bytes,
            "extensions": dict(sorted(extensions.items())),
            "sample_payload_files": sample_payload_files,
            "last_modified_ts": iso_from_timestamp(latest_mtime),
        }
    )
    return result


def build_inventory(datasets_root: Path) -> dict[str, Any]:
    datasets = [inspect_dataset(entry) for entry in EXPECTED_DATASETS]
    missing = [item["dataset_key"] for item in datasets if item["status"] == "missing"]
    empty = [item["dataset_key"] for item in datasets if item["status"] == "empty"]
    with_payload = [item["dataset_key"] for item in datasets if item["status"] == "has_payload"]

    unknown_dirs: list[str] = []
    if datasets_root.exists():
        expected_paths = {ROOT / entry["relative_path"] for entry in EXPECTED_DATASETS}
        for child in sorted(datasets_root.iterdir(), key=lambda item: item.name):
            if child.is_dir() and child not in expected_paths:
                unknown_dirs.append(to_relative(child))

    return {
        "inventory_version": "0.1",
        "created_by": "scripts/inspect_datasets.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "datasets_root": to_relative(datasets_root),
        "notes": [
            "This inventory is a preflight snapshot, not dataset_manifest.json.",
            "Files named .gitkeep are ignored when deciding whether a dataset has payload data.",
            "Run scripts/validate_manifest_files.py only after formal manifests are generated.",
        ],
        "summary": {
            "expected_dataset_count": len(EXPECTED_DATASETS),
            "present_dataset_dir_count": sum(1 for item in datasets if item["exists"]),
            "payload_dataset_count": len(with_payload),
            "empty_dataset_count": len(empty),
            "missing_dataset_count": len(missing),
            "total_payload_files": sum(item["payload_file_count"] for item in datasets),
            "total_payload_bytes": sum(item["payload_bytes"] for item in datasets),
            "missing_dataset_keys": missing,
            "empty_dataset_keys": empty,
            "payload_dataset_keys": with_payload,
            "unknown_dataset_dirs": unknown_dirs,
        },
        "datasets": datasets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local dataset directories before G-DATA manifest generation."
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Dataset root directory to inspect.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON report path to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    output = Path(args.output)

    if not datasets_root.is_absolute():
        datasets_root = ROOT / datasets_root
    if not output.is_absolute():
        output = ROOT / output

    inventory = build_inventory(datasets_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = inventory["summary"]
    print(f"Wrote {to_relative(output)}")
    print(
        "Datasets: "
        f"{summary['payload_dataset_count']} with payload, "
        f"{summary['empty_dataset_count']} empty, "
        f"{summary['missing_dataset_count']} missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
