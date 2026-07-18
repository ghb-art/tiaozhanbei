from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOUD = ROOT / "reports" / "audit" / "gate_chapter2_capability_cloud.json"
DEFAULT_EDGE = ROOT / "reports" / "audit" / "gate_chapter2_capability_edge.json"
DEFAULT_OUTPUT = ROOT / "reports" / "audit" / "gate_g1_capability_retention.json"

DATASET_TO_METRIC = {
    "gsm8k": "math_ratio",
    "humaneval": "code_ratio",
    "cmmlu": "nlp_ratio",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Chapter 2 Cloud/Edge capability retention ratios.")
    parser.add_argument("--cloud-audit", "--cloud_audit", default=str(DEFAULT_CLOUD))
    parser.add_argument("--edge-audit", "--edge_audit", default=str(DEFAULT_EDGE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-ratio", "--min_ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_ratio <= 1:
        print("--min-ratio must be in [0, 1]", file=sys.stderr)
        return 2

    cloud_path = resolve_path(args.cloud_audit)
    edge_path = resolve_path(args.edge_audit)
    output_path = resolve_path(args.output)
    cloud = load_json(cloud_path)
    edge = load_json(edge_path)
    cloud_acc = cloud.get("accuracy_by_dataset", {})
    edge_acc = edge.get("accuracy_by_dataset", {})

    ratios: dict[str, float] = {}
    capped: dict[str, float] = {}
    errors: list[str] = []
    for dataset_key, metric_name in DATASET_TO_METRIC.items():
        if dataset_key not in cloud_acc:
            errors.append(f"Missing cloud accuracy for {dataset_key}")
            continue
        if dataset_key not in edge_acc:
            errors.append(f"Missing edge accuracy for {dataset_key}")
            continue
        denominator = float(cloud_acc[dataset_key])
        numerator = float(edge_acc[dataset_key])
        ratio = numerator / denominator if denominator > 0 else 0.0
        ratios[metric_name] = ratio
        capped[f"{metric_name}_cap"] = min(ratio, 1.0)

    overall = sum(capped.values()) / len(capped) if capped else 0.0
    hard_gate_pass = (
        not errors
        and all(value >= args.min_ratio for value in ratios.values())
        and len(ratios) == len(DATASET_TO_METRIC)
        and overall >= args.min_ratio
    )
    report = {
        "gate": "G1-capability-retention",
        "check_version": "1.1",
        "created_by": "scripts/summarize_chapter2_capability.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if hard_gate_pass else "failed",
        "min_ratio": args.min_ratio,
        "gate_policy": "implementation_plan_strict_math_code_nlp_overall",
        "hard_gate_metrics": ["math_ratio", "code_ratio", "nlp_ratio", "overall_r_cap"],
        "auxiliary_metrics": [],
        "cloud_audit_path": display_path(cloud_path),
        "cloud_audit_hash": sha256_file(cloud_path),
        "edge_audit_path": display_path(edge_path),
        "edge_audit_hash": sha256_file(edge_path),
        "cloud_accuracy_by_dataset": cloud_acc,
        "edge_accuracy_by_dataset": edge_acc,
        "ratios": ratios,
        "capped_ratios": capped,
        "overall_r_cap": overall,
        "overall_r_cap_note": "Hard gate metric together with the three independent ratios, per IMPLEMENTATION_PLAN.md.",
        "dataset_mapping": DATASET_TO_METRIC,
        "errors": errors,
    }
    report["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in report.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(output_path, report)
    print(f"Wrote {display_path(output_path)}")
    print(f"status={report['status']}")
    print(f"overall_r_cap={overall:.6f}")
    for key, value in ratios.items():
        print(f"{key}={value:.6f}")
    return 0 if hard_gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
