from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "data" / "datasets"
IGNORED_FILES = {".gitkeep"}

EXPECTED_DATASETS = [
    ("gsm8k", "GSM8K"),
    ("humaneval", "HumanEval"),
    ("mmlu", "MMLU"),
    ("cmmlu", "CMMLU"),
    ("mvtec_ad", "MVTec AD"),
    ("neu_det", "NEU-DET"),
    ("cityflow", "CityFlow"),
    ("ua_detrac", "UA-DETRAC"),
]


def has_payload(path: Path) -> bool:
    if not path.is_dir():
        return False

    for child in path.rglob("*"):
        if child.is_file() and child.name not in IGNORED_FILES:
            return True
    return False


def dataset_status(key: str, name: str) -> dict[str, Any]:
    path = DATASETS_ROOT / key
    payload_files = []
    total_payload_bytes = 0

    if path.is_dir():
        for child in path.rglob("*"):
            if not child.is_file() or child.name in IGNORED_FILES:
                continue
            payload_files.append(child)
            total_payload_bytes += child.stat().st_size

    return {
        "key": key,
        "name": name,
        "path": path,
        "exists": path.is_dir(),
        "payload_file_count": len(payload_files),
        "payload_bytes": total_payload_bytes,
        "has_payload": bool(payload_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether expected dataset directories contain real payload files."
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success even when dataset directories exist but have no payload yet.",
    )
    parser.add_argument(
        "--dataset",
        choices=[key for key, _ in EXPECTED_DATASETS],
        action="append",
        help="Validate only selected dataset key. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.dataset or [key for key, _ in EXPECTED_DATASETS])
    statuses = [
        dataset_status(key, name)
        for key, name in EXPECTED_DATASETS
        if key in selected
    ]

    errors: list[str] = []
    for item in statuses:
        relative_path = item["path"].relative_to(ROOT).as_posix()
        if not item["exists"]:
            print(f"[MISSING] {item['name']} -> {relative_path}")
            errors.append(f"{item['key']} directory is missing")
            continue

        if item["has_payload"]:
            print(
                f"[OK] {item['name']} -> {relative_path} "
                f"({item['payload_file_count']} payload files, {item['payload_bytes']} bytes)"
            )
            continue

        label = "[EMPTY-ALLOWED]" if args.allow_empty else "[EMPTY]"
        print(f"{label} {item['name']} -> {relative_path} has no payload files")
        if not args.allow_empty:
            errors.append(f"{item['key']} has no payload files")

    if errors:
        print()
        print("Dataset presence validation failed:")
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print()
    if args.allow_empty:
        print("Dataset presence check completed with empty directories allowed.")
    else:
        print("Dataset presence validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
