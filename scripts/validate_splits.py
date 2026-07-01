from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "data" / "datasets"
SPLITS = ROOT / "data" / "splits"
SEED = 42


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_ids(ids: list[str]) -> str:
    return stable_hash_text("\n".join(ids) + "\n")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[1:] if rows else []


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_gzip_jsonl(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sample_stratified(groups: dict[str, list[str]], total: int, seed: int) -> list[str]:
    available = {key: sorted(values) for key, values in groups.items() if values}
    total_available = sum(len(values) for values in available.values())
    if total > total_available:
        raise ValueError(f"Cannot sample {total} from {total_available} items")

    rng = random.Random(seed)
    shuffled: dict[str, list[str]] = {}
    for key, values in available.items():
        copied = list(values)
        rng.shuffle(copied)
        shuffled[key] = copied

    raw = {
        key: (len(values) * total / total_available)
        for key, values in available.items()
    }
    quotas = {key: int(raw[key]) for key in available}
    while sum(quotas.values()) < total:
        candidates = sorted(
            available,
            key=lambda key: (raw[key] - quotas[key], len(available[key]), key),
            reverse=True,
        )
        for key in candidates:
            if quotas[key] < len(available[key]):
                quotas[key] += 1
                break
    while sum(quotas.values()) > total:
        candidates = sorted(
            [key for key in available if quotas[key] > 0],
            key=lambda key: (quotas[key] - raw[key], key),
            reverse=True,
        )
        quotas[candidates[0]] -= 1

    selected: list[str] = []
    for key in sorted(shuffled):
        selected.extend(sorted(shuffled[key][: quotas[key]]))
    return sorted(selected)


def split_record(
    key: str,
    name: str,
    version: str,
    official_scale: str,
    method: str,
    train: list[str],
    validation: list[str],
    test: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_ids = {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
    }
    hashes = {split: hash_ids(ids) for split, ids in split_ids.items()}
    return {
        "dataset_key": key,
        "dataset_name": name,
        "dataset_version": version,
        "official_scale": official_scale,
        "method": method,
        "sampling_seed": SEED,
        "counts": {split: len(ids) for split, ids in split_ids.items()},
        "hashes": hashes,
        "split_hash": stable_hash_text(json.dumps(hashes, sort_keys=True)),
        "splits": split_ids,
        "metadata": metadata or {},
    }


def build_gsm8k() -> dict[str, Any]:
    base = DATASETS / "gsm8k" / "grade_school_math" / "data"
    train = [f"gsm8k/train/{idx:05d}" for idx in range(count_jsonl(base / "train.jsonl"))]
    test_all = [f"gsm8k/test/{idx:05d}" for idx in range(count_jsonl(base / "test.jsonl"))]
    selected = sample_stratified({"all": test_all}, 500, SEED)
    return split_record(
        "gsm8k",
        "GSM8K",
        "openai-grade-school-math@3101c7d",
        "train=7473, test=1319",
        "Use official train for distillation; freeze deterministic 500-sample subset from official test for Final Gate.",
        train,
        [],
        selected,
        {"official_test_count": len(test_all), "final_gate_subset_count": len(selected)},
    )


def build_humaneval() -> dict[str, Any]:
    path = DATASETS / "humaneval" / "data" / "HumanEval.jsonl.gz"
    ids: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(str(item["task_id"]))
    return split_record(
        "humaneval",
        "HumanEval",
        "openai-human-eval@6d43fb9",
        f"tasks={len(ids)}",
        "Use all official HumanEval tasks as Final Gate only; no train or validation split.",
        [],
        [],
        ids,
        {"execution_requires_sandbox": True},
    )


def csv_split_ids(root: Path, split: str, suffix: str | None = None) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.csv")):
        subject = path.stem
        if suffix and subject.endswith(suffix):
            subject = subject[: -len(suffix)]
        rows = read_csv_rows(path)
        groups[subject] = [
            f"{split}/{subject}/{idx:05d}"
            for idx in range(len(rows))
        ]
    return groups


def flatten(groups: dict[str, list[str]], prefix: str) -> list[str]:
    return sorted(f"{prefix}/{item}" for values in groups.values() for item in values)


def build_mmlu() -> dict[str, Any]:
    base = DATASETS / "mmlu" / "data"
    dev = csv_split_ids(base / "dev", "dev", "_dev")
    val = csv_split_ids(base / "val", "val", "_val")
    test = csv_split_ids(base / "test", "test", "_test")
    aux = csv_split_ids(base / "auxiliary_train", "auxiliary_train")
    train = flatten(aux, "mmlu") + flatten(dev, "mmlu")
    validation = flatten(val, "mmlu")
    test_pool = {subject: [f"mmlu/{item}" for item in ids] for subject, ids in test.items()}
    selected = sample_stratified(test_pool, 1000, SEED)
    return split_record(
        "mmlu",
        "MMLU",
        "hendrycks-data.tar",
        "subjects=57, official test rows sampled to 1000 final items",
        "Use auxiliary_train plus official dev for train; official val for validation; stratified 1000-sample official test subset for Final Gate.",
        train,
        validation,
        selected,
        {
            "subject_count": len(test),
            "test_pool_count": sum(len(ids) for ids in test.values()),
        },
    )


def build_cmmlu() -> dict[str, Any]:
    base = DATASETS / "cmmlu" / "data"
    dev = csv_split_ids(base / "dev", "dev")
    test = csv_split_ids(base / "test", "test")
    validation = flatten(dev, "cmmlu")
    test_pool = {subject: [f"cmmlu/{item}" for item in ids] for subject, ids in test.items()}
    selected = sample_stratified(test_pool, 1000, SEED)
    return split_record(
        "cmmlu",
        "CMMLU",
        "haonan-li-CMMLU@d6e7b71",
        "subjects=67, test rows=11582",
        "Use official dev for validation; freeze stratified 1000-sample official test subset for Final Gate; no non-final train data yet.",
        [],
        validation,
        selected,
        {
            "subject_count": len(test),
            "test_pool_count": sum(len(ids) for ids in test.values()),
        },
    )


def build_mvtec() -> dict[str, Any]:
    base = DATASETS / "mvtec_ad" / "mvtec_anomaly_detection"
    train: list[str] = []
    test: list[str] = []
    class_counts: dict[str, dict[str, int]] = {}
    for class_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        class_name = class_dir.name
        train_good = sorted((class_dir / "train" / "good").glob("*.png"))
        test_images = sorted((class_dir / "test").glob("*/*.png"))
        train.extend(f"mvtec_ad/train/{class_name}/good/{path.name}" for path in train_good)
        test.extend(
            f"mvtec_ad/test/{class_name}/{path.parent.name}/{path.name}"
            for path in test_images
        )
        class_counts[class_name] = {
            "train_good": len(train_good),
            "test": len(test_images),
        }
    return split_record(
        "mvtec_ad",
        "MVTec AD",
        "mvtec_anomaly_detection.tar.xz",
        "15 classes with official train/test layout",
        "Use all official train/good images for train; use full official test images for Final Gate; no validation split frozen here.",
        train,
        [],
        test,
        {
            "class_counts": class_counts,
            "official_test_full_final_gate": True,
        },
    )


NEU_CLASS_RE = re.compile(r"(.+)_([0-9]+)$")


def neu_class(stem: str) -> str:
    match = NEU_CLASS_RE.match(stem)
    if not match:
        raise ValueError(f"Unexpected NEU-DET stem: {stem}")
    return match.group(1)


def build_neu_det() -> dict[str, Any]:
    base = DATASETS / "neu_det" / "NEU-DET"
    images_by_stem = {path.stem: path for path in base.rglob("*.jpg")}
    xml_by_stem = {path.stem: path for path in base.rglob("*.xml")}
    missing_xml = sorted(set(images_by_stem) - set(xml_by_stem))
    missing_image = sorted(set(xml_by_stem) - set(images_by_stem))
    if missing_xml or missing_image:
        raise RuntimeError(f"NEU-DET image/xml mismatch: missing_xml={missing_xml[:5]}, missing_image={missing_image[:5]}")

    by_class: dict[str, list[str]] = defaultdict(list)
    for stem in sorted(images_by_stem):
        by_class[neu_class(stem)].append(stem)

    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    class_split_counts: dict[str, dict[str, int]] = {}
    for class_name, stems in sorted(by_class.items()):
        shuffled = sorted(stems)
        class_seed = SEED + int(stable_hash_text(class_name)[:8], 16)
        random.Random(class_seed).shuffle(shuffled)
        class_train = sorted(shuffled[:210])
        class_validation = sorted(shuffled[210:240])
        class_test = sorted(shuffled[240:300])
        train.extend(f"neu_det/train/{stem}" for stem in class_train)
        validation.extend(f"neu_det/validation/{stem}" for stem in class_validation)
        test.extend(f"neu_det/test/{stem}" for stem in class_test)
        class_split_counts[class_name] = {
            "train": len(class_train),
            "validation": len(class_validation),
            "test": len(class_test),
        }

    return split_record(
        "neu_det",
        "NEU-DET",
        "kaggle-mirror-kaustubhdikshit-neu-surface-defect-database",
        "1800 images, 1800 XML annotations, 6 classes",
        "Ignore mirror-provided split; pair image/XML globally by stem, then freeze 70/10/20 stratified class split with 210/30/60 per class.",
        train,
        validation,
        test,
        {
            "class_split_counts": class_split_counts,
            "paired_sample_count": len(images_by_stem),
            "mirror_split_mismatch_fixed": "crazing_240 image/xml are paired before split assignment",
        },
    )


def cityflow_camera_dirs(base: Path, split: str) -> list[str]:
    root = base / split
    if not root.is_dir():
        return []
    return sorted(
        f"cityflow/{split}/{scene.name}/{camera.name}"
        for scene in root.iterdir()
        if scene.is_dir()
        for camera in scene.iterdir()
        if camera.is_dir() and camera.name.startswith("c")
    )


def cityflow_eval_stats(base: Path) -> dict[str, int]:
    observation_count = 0
    vehicles: set[str] = set()
    camera_vehicle: set[tuple[str, str]] = set()
    for filename in ["ground_truth_train.txt", "ground_truth_validation.txt"]:
        path = base / "eval" / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                camera_id, vehicle_id = parts[0], parts[1]
                observation_count += 1
                vehicles.add(vehicle_id)
                camera_vehicle.add((camera_id, vehicle_id))
    return {
        "vehicle_id_count": len(vehicles),
        "relation_node_count": len(camera_vehicle),
        "relation_edge_count": observation_count,
        "relation_group_count": len(vehicles),
    }


def build_cityflow() -> dict[str, Any]:
    base = DATASETS / "cityflow" / "AICity22_Track1_MTMC_Tracking"
    train = cityflow_camera_dirs(base, "train")
    validation = cityflow_camera_dirs(base, "validation")
    test = cityflow_camera_dirs(base, "test")
    stats = cityflow_eval_stats(base)
    stats["conflict_group_count"] = 60
    stats["camera_dir_count"] = len(train) + len(validation) + len(test)
    return split_record(
        "cityflow",
        "CityFlow",
        "AICity22-Track1-CityFlowV2",
        f"camera_dirs={stats['camera_dir_count']}",
        "Freeze official train/validation/test camera directories; relation statistics are derived from AI City eval ground-truth files.",
        train,
        validation,
        test,
        stats,
    )


def build_ua_detrac() -> dict[str, Any]:
    base = DATASETS / "ua_detrac" / "ua_detrac_kaggle_archive"
    image_root = base / "DETRAC-Images" / "DETRAC-Images"
    train_xml = sorted((base / "DETRAC-Train-Annotations-XML").rglob("*.xml"))
    test_xml = sorted((base / "DETRAC-Test-Annotations-XML").rglob("*.xml"))
    frame_counts = {
        seq.name: len(list(seq.glob("*.jpg")))
        for seq in sorted(image_root.iterdir())
        if seq.is_dir()
    }
    validation = [f"ua_detrac/validation/{path.stem}" for path in train_xml]
    test = [f"ua_detrac/test/{path.stem}" for path in test_xml]
    return split_record(
        "ua_detrac",
        "UA-DETRAC",
        "kaggle-mirror-bratjay-ua-detrac-orig",
        f"sequences={len(frame_counts)}, frames={sum(frame_counts.values())}, train_xml=60, test_xml=40",
        "Use official-style annotation split from mirror: train XML sequences as validation/reference only, test XML sequences as Final Gate; no graph training.",
        [],
        validation,
        test,
        {
            "sequence_count": len(frame_counts),
            "frame_count": sum(frame_counts.values()),
            "train_xml_count": len(train_xml),
            "test_xml_count": len(test_xml),
        },
    )


BUILDERS = [
    build_gsm8k,
    build_humaneval,
    build_mmlu,
    build_cmmlu,
    build_mvtec,
    build_neu_det,
    build_cityflow,
    build_ua_detrac,
]


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_records = []
    for record in records:
        split_files: dict[str, str] = {}
        for split, ids in record["splits"].items():
            path = output_dir / f"{record['dataset_key']}_{split}.txt"
            path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
            split_files[split] = rel(path)
        item = {key: value for key, value in record.items() if key != "splits"}
        item["split_files"] = split_files
        summary_records.append(item)

    global_hash = stable_hash_text(json.dumps(summary_records, sort_keys=True, ensure_ascii=False))
    frozen = {
        "split_version": "1.0",
        "created_by": "scripts/validate_splits.py",
        "sampling_seed": SEED,
        "global_split_hash": global_hash,
        "datasets": summary_records,
    }
    (output_dir / "frozen_splits.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return frozen


def load_frozen(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(frozen: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for dataset in frozen.get("datasets", []):
        key = dataset["dataset_key"]
        seen_by_split: dict[str, set[str]] = {}
        for split, file_path in dataset.get("split_files", {}).items():
            path = ROOT / file_path
            if not path.is_file():
                errors.append(f"{key}:{split} missing split file: {file_path}")
                continue
            ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(ids) != dataset["counts"][split]:
                errors.append(f"{key}:{split} count mismatch")
            if hash_ids(ids) != dataset["hashes"][split]:
                errors.append(f"{key}:{split} hash mismatch")
            if len(ids) != len(set(ids)):
                errors.append(f"{key}:{split} contains duplicate ids")
            seen_by_split[split] = set(ids)

        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            overlap = seen_by_split.get(left, set()) & seen_by_split.get(right, set())
            if overlap:
                errors.append(f"{key}:{left}/{right} overlap: {sorted(overlap)[:3]}")

        if key == "neu_det":
            counts = dataset["counts"]
            if counts != {"train": 1260, "validation": 180, "test": 360}:
                errors.append(f"neu_det unexpected split counts: {counts}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and validate dataset splits.")
    parser.add_argument("--write", action="store_true", help="Generate frozen split files.")
    parser.add_argument("--check-leakage", action="store_true", help="Validate split disjointness and hashes.")
    parser.add_argument("--output-dir", default=str(SPLITS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if args.write:
        records = [builder() for builder in BUILDERS]
        frozen = write_outputs(records, output_dir)
        print(f"Wrote {rel(output_dir / 'frozen_splits.json')}")
        print(f"Global split hash: {frozen['global_split_hash']}")
    else:
        frozen = load_frozen(output_dir / "frozen_splits.json")

    if args.write or args.check_leakage:
        errors = validate_frozen(frozen)
        if errors:
            print("Split validation failed:")
            for error in errors:
                print(f"[FAIL] {error}")
            return 1
        print("Split validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
