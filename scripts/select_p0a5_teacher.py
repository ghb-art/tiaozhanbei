#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class SelectionError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_evaluation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"Missing checkpoint evaluation: {display_path(path)}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("role") != "teacher"
        or report.get("mode") != "checkpoint_evaluation"
        or int(report.get("formal_test_reference_count", -1)) != 0
    ):
        raise SelectionError(f"Rejected checkpoint evaluation: {display_path(path)}")
    loss = report.get("evaluation_metrics", {}).get("eval_loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise SelectionError(f"Invalid eval_loss: {display_path(path)}")
    checkpoint = resolve_path(str(report.get("checkpoint", "")))
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not state_path.is_file() or not adapter_path.is_file():
        raise SelectionError(f"Evaluation checkpoint is incomplete: {display_path(checkpoint)}")
    if sha256_file(state_path) != report.get("checkpoint_state_hash"):
        raise SelectionError(f"Checkpoint state hash changed: {display_path(checkpoint)}")
    if sha256_file(adapter_path) != report.get("adapter_hash"):
        raise SelectionError(f"Adapter hash changed: {display_path(checkpoint)}")
    return report


def select_best(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evaluations) < 2:
        raise SelectionError("At least two checkpoint evaluations are required")
    config_hashes = {str(item.get("config_hash", "")) for item in evaluations}
    validation_hashes = {
        str(item.get("validation_data_hash", "")) for item in evaluations
    }
    evaluated_rows = {int(item.get("evaluated_rows", -1)) for item in evaluations}
    steps = [int(item.get("checkpoint_step", -1)) for item in evaluations]
    if len(config_hashes) != 1 or "" in config_hashes:
        raise SelectionError("Checkpoint evaluations used different configs")
    if len(validation_hashes) != 1 or "" in validation_hashes:
        raise SelectionError("Checkpoint evaluations used different validation data")
    if evaluated_rows != {2200}:
        raise SelectionError(f"Expected 2,200 validation rows, got {evaluated_rows}")
    if len(set(steps)) != len(steps) or any(step < 1 for step in steps):
        raise SelectionError(f"Invalid checkpoint steps: {steps}")
    return min(
        evaluations,
        key=lambda item: (
            float(item["evaluation_metrics"]["eval_loss"]),
            int(item["checkpoint_step"]),
        ),
    )


def publish_adapter(checkpoint: Path, output: Path) -> list[str]:
    sources = [
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "README.md",
    ]
    if not all(path.is_file() for path in sources[:2]):
        raise SelectionError(f"Selected checkpoint has no adapter: {display_path(checkpoint)}")
    output.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for source in sources:
        if source.is_file():
            shutil.copy2(source, output / source.name)
            published.append(source.name)
    return published


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and publish the best P0-A5 Teacher checkpoint."
    )
    parser.add_argument("--config", default="configs/p0a5_capability.json")
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        output_dir = resolve_path(args.output_dir)
        audit_path = resolve_path(args.audit)
        evaluations = [
            read_evaluation(resolve_path(path)) for path in args.evaluation
        ]
        current_config_hash = sha256_file(config_path)
        if any(item["config_hash"] != current_config_hash for item in evaluations):
            raise SelectionError("Evaluation config hash does not match current config")
        selected = select_best(evaluations)
        checkpoint = resolve_path(selected["checkpoint"])
        published = publish_adapter(checkpoint, output_dir)
        selection = {
            "gate": "P0-A5-TEACHER-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a5_teacher.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "role": "teacher",
            "selection_policy": "lowest_internal_validation_eval_loss",
            "config": display_path(config_path),
            "config_hash": current_config_hash,
            "evaluations": [
                {
                    "checkpoint": item["checkpoint"],
                    "checkpoint_step": int(item["checkpoint_step"]),
                    "eval_loss": float(item["evaluation_metrics"]["eval_loss"]),
                    "report_hash": item["report_hash"],
                }
                for item in sorted(
                    evaluations, key=lambda item: int(item["checkpoint_step"])
                )
            ],
            "best_checkpoint": display_path(checkpoint),
            "global_step": int(selected["checkpoint_step"]),
            "best_metric": float(selected["evaluation_metrics"]["eval_loss"]),
            "validation_data": selected["validation_data"],
            "validation_data_hash": selected["validation_data_hash"],
            "evaluated_rows": int(selected["evaluated_rows"]),
            "formal_test_reference_count": 0,
            "published_files": published,
            "published_adapter_hash": sha256_file(
                output_dir / "adapter_model.safetensors"
            ),
            "errors": [],
        }
        selection["report_hash"] = sha256_text(
            json.dumps(selection, ensure_ascii=False, sort_keys=True)
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Selected Teacher: {display_path(checkpoint)}")
        print(f"eval_loss={selection['best_metric']}")
        print(f"Audit: {display_path(audit_path)}")
        return 0
    except (SelectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 Teacher selection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
