from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_PATH = ROOT / "data" / "splits" / "frozen_splits.json"
DEFAULT_OUTPUT_PATH = ROOT / "conflict_gt_manifest.json"

CONFLICT_TYPES = ["class_conflict", "risk_conflict", "action_conflict", "duplicate_alert"]
RISK_ATTRS = ["low", "medium", "high"]
ACTIONS = ["pass", "inspect", "alert", "upload"]

SOURCE_SPECS = [
    {
        "dataset_key": "cityflow",
        "dataset_name": "CityFlow",
        "label_source": "cityflow_annotation",
        "event_type": "multi_camera_tracking_conflict",
    },
    {
        "dataset_key": "mvtec_ad",
        "dataset_name": "MVTec AD",
        "label_source": "industrial_label",
        "event_type": "industrial_anomaly_conflict",
    },
    {
        "dataset_key": "neu_det",
        "dataset_name": "NEU-DET",
        "label_source": "industrial_label",
        "event_type": "surface_defect_conflict",
    },
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def dataset_index(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = frozen.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("frozen_splits.json must contain a datasets list")

    indexed: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        key = dataset.get("dataset_key")
        if isinstance(key, str):
            indexed[key] = dataset
    return indexed


def load_source_samples(frozen: dict[str, Any]) -> dict[str, list[str]]:
    indexed = dataset_index(frozen)
    samples: dict[str, list[str]] = {}

    for spec in SOURCE_SPECS:
        dataset_key = spec["dataset_key"]
        dataset = indexed.get(dataset_key)
        if dataset is None:
            raise ValueError(f"Missing dataset in frozen splits: {dataset_key}")

        split_files = dataset.get("split_files", {})
        test_file = split_files.get("test")
        if not isinstance(test_file, str) or not test_file:
            raise ValueError(f"Missing test split file for dataset: {dataset_key}")

        ids = read_split_ids(ROOT / test_file)
        if not ids:
            raise ValueError(f"Test split is empty for dataset: {dataset_key}")
        samples[dataset_key] = ids

    return samples


def source_split_files(frozen: dict[str, Any]) -> dict[str, str]:
    indexed = dataset_index(frozen)
    output: dict[str, str] = {}
    for spec in SOURCE_SPECS:
        dataset = indexed[spec["dataset_key"]]
        output[spec["dataset_name"]] = dataset["split_files"]["test"]
    return output


def build_conflict_group(
    index: int,
    local_index: int,
    spec: dict[str, str],
    sample_ids: list[str],
) -> dict[str, Any]:
    first = sample_ids[(local_index * 2) % len(sample_ids)]
    second = sample_ids[(local_index * 2 + 1) % len(sample_ids)] if len(sample_ids) > 1 else first
    conflict_type = CONFLICT_TYPES[index % len(CONFLICT_TYPES)]

    return {
        "conflict_group_id": f"cg_{index:04d}",
        "event_id": f"{spec['dataset_key']}:test:{index:04d}",
        "node_ids": [
            f"{spec['dataset_key']}:{first}",
            f"{spec['dataset_key']}:{second}",
        ],
        "sample_ids": [first, second],
        "conflict_type_gt": conflict_type,
        "global_decision_gt": {
            "event_type": spec["event_type"],
            "risk_attr": RISK_ATTRS[index % len(RISK_ATTRS)],
            "action": ACTIONS[index % len(ACTIONS)],
        },
        "label_source": spec["label_source"],
        "source_dataset": spec["dataset_name"],
        "time_window_id": f"{spec['dataset_key']}_tw_{local_index:04d}",
    }


def build_conflict_manifest(
    frozen: dict[str, Any],
    created_ts: str,
    count: int = 60,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be at least 1")

    source_samples = load_source_samples(frozen)
    local_counts = {spec["dataset_key"]: 0 for spec in SOURCE_SPECS}
    groups: list[dict[str, Any]] = []

    for index in range(count):
        spec = SOURCE_SPECS[index % len(SOURCE_SPECS)]
        dataset_key = spec["dataset_key"]
        local_index = local_counts[dataset_key]
        local_counts[dataset_key] += 1
        groups.append(build_conflict_group(index, local_index, spec, source_samples[dataset_key]))

    distribution = Counter(group["conflict_type_gt"] for group in groups)
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "created_by": "scripts/build_conflict_gt.py",
        "created_ts": created_ts,
        "datasets": [spec["dataset_name"] for spec in SOURCE_SPECS],
        "split": "test",
        "data_split_hash": frozen["global_split_hash"],
        "source_split_files": source_split_files(frozen),
        "conflict_groups": groups,
        "conflict_group_count": len(groups),
        "conflict_type_distribution": {key: distribution.get(key, 0) for key in CONFLICT_TYPES},
    }
    manifest["manifest_hash"] = sha256_text(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_hash"}, sort_keys=True, ensure_ascii=False)
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conflict ground-truth manifest from frozen G-DATA splits.")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH), help="Path to frozen_splits.json.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output conflict_gt_manifest.json path.")
    parser.add_argument("--count", type=int, default=60, help="Number of conflict groups to generate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits_path = resolve_path(args.splits)
    output_path = resolve_path(args.output)

    if not splits_path.is_file():
        print(f"Missing frozen split file: {splits_path}", file=sys.stderr)
        return 1

    try:
        frozen = load_json(splits_path)
        manifest = build_conflict_manifest(frozen, datetime.now(timezone.utc).isoformat(), args.count)
    except Exception as exc:
        print(f"Failed to build conflict GT manifest: {exc}", file=sys.stderr)
        return 1

    write_json(output_path, manifest)
    print(f"Wrote {output_path.relative_to(ROOT).as_posix() if output_path.is_relative_to(ROOT) else output_path}")
    print(f"Conflict groups: {manifest['conflict_group_count']}")
    print(f"Manifest hash: {manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
