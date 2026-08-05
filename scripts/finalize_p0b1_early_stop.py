#!/usr/bin/env python3
"""Publish a verified P0-B1 early-stop checkpoint after terminal DDP failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
)


class FinalizeError(RuntimeError):
    pass


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizeError(f"Missing JSON evidence: {display(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizeError(f"Invalid JSON evidence: {display(path)}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="models/checkpoints/p0b1/shared/checkpoint-4000"
    )
    parser.add_argument(
        "--validation-audit",
        default="reports/audit/gate_p0b1_checkpoint_4000_validation.json",
    )
    parser.add_argument(
        "--preflight-audit", default="reports/audit/gate_p0b1_train_preflight.json"
    )
    parser.add_argument(
        "--failure-audit",
        default=(
            "reports/runtime/"
            "memory_watchdog_20260804_200317_627213_1857486.json"
        ),
    )
    parser.add_argument("--output", default="models/checkpoints/p0b1/shared")
    parser.add_argument(
        "--audit", default="reports/audit/gate_p0b1_train_shared.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve(args.checkpoint)
    validation_path = resolve(args.validation_audit)
    preflight_path = resolve(args.preflight_audit)
    failure_path = resolve(args.failure_audit)
    output = resolve(args.output)
    audit_path = resolve(args.audit)
    if not checkpoint.is_dir():
        raise FinalizeError(f"Missing checkpoint: {display(checkpoint)}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FinalizeError(f"Incomplete checkpoint files: {missing}")

    state = read_json(checkpoint / "trainer_state.json")
    validation = read_json(validation_path)
    preflight = read_json(preflight_path)
    failure = read_json(failure_path)
    step = int(state.get("global_step", -1))
    if checkpoint.name != f"checkpoint-{step}":
        raise FinalizeError("Checkpoint directory and trainer global_step disagree")
    best_path = Path(str(state.get("best_model_checkpoint", ""))).resolve()
    if best_path != checkpoint.resolve():
        raise FinalizeError(
            f"Checkpoint is not Trainer's best checkpoint: {best_path}"
        )
    callback = state.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {})
    callback_args = callback.get("args", {})
    callback_attributes = callback.get("attributes", {})
    patience = int(callback_args.get("early_stopping_patience", -1))
    counter = int(callback_attributes.get("early_stopping_patience_counter", -1))
    control = state.get("stateful_callbacks", {}).get("TrainerControl", {}).get("args", {})
    if patience < 1 or counter < patience or not bool(control.get("should_training_stop")):
        raise FinalizeError(
            f"Checkpoint has no terminal early-stop evidence: counter={counter} patience={patience}"
        )
    if validation.get("status") != "passed" or validation.get("mode") != "checkpoint_evaluation":
        raise FinalizeError("Independent checkpoint validation did not pass")
    if Path(str(validation.get("checkpoint", ""))).name != checkpoint.name:
        raise FinalizeError("Independent validation belongs to another checkpoint")
    if int(validation.get("checkpoint_step", -1)) != step:
        raise FinalizeError("Independent validation step mismatch")
    adapter_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    if validation.get("adapter_hash") != adapter_hash:
        raise FinalizeError("Validated adapter hash changed")
    eval_loss = float(validation.get("evaluation_metrics", {}).get("eval_loss", "nan"))
    best_metric = float(state.get("best_metric", "nan"))
    if abs(eval_loss - best_metric) > 1e-8:
        raise FinalizeError(
            f"Independent eval loss did not reproduce best metric: {eval_loss} != {best_metric}"
        )
    if preflight.get("status") != "dry_run_passed":
        raise FinalizeError("Training preflight audit is not passed")
    for key in ("config_hash", "train_data_hash", "validation_data_hash"):
        if validation.get(key) != preflight.get(key):
            raise FinalizeError(f"Preflight/validation {key} mismatch")
    if failure.get("status") != "command_failed" or int(failure.get("child_return_code", 0)) == 0:
        raise FinalizeError("Expected failed terminal command evidence is missing")
    if failure.get("active_violations"):
        raise FinalizeError("Recovery is forbidden after a memory-threshold violation")

    output.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        source = checkpoint / name
        if not source.is_file():
            continue
        temporary = output / f".{name}.tmp-{os.getpid()}"
        shutil.copy2(source, temporary)
        temporary.replace(output / name)
        published.append(name)
    from transformers import AutoTokenizer

    base = resolve(str(validation["model_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=True
    )
    tokenizer.save_pretrained(output)
    published = sorted(
        set(published)
        | {
            path.name
            for path in output.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
    )

    report = dict(validation)
    report.update(
        {
            "gate": "P0-B1-RECOVERED-EARLY-STOP-TRAIN",
            "check_version": "1.0",
            "created_by": "scripts/finalize_p0b1_early_stop.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "mode": "recovered_early_stop_finalization",
            "best_checkpoint": display(checkpoint),
            "published_files": published,
            "global_step": step,
            "best_metric": best_metric,
            "train_metrics": {
                "epoch": float(state.get("epoch", 0.0)),
                "early_stopped": True,
                "completed_fraction": step / float(state.get("max_steps", step)),
            },
            "early_stop_recovery": {
                "patience": patience,
                "patience_counter": counter,
                "should_training_stop": True,
                "independent_eval_loss": eval_loss,
                "validation_audit": display(validation_path),
                "validation_audit_hash": sha256_file(validation_path),
                "failure_audit": display(failure_path),
                "failure_audit_hash": sha256_file(failure_path),
                "failure_was_memory_threshold": False,
                "checkpoint_files": {
                    name: {"bytes": (checkpoint / name).stat().st_size, "sha256": sha256_file(checkpoint / name)}
                    for name in REQUIRED_CHECKPOINT_FILES
                },
            },
            "finalizer_hash": sha256_file(Path(__file__)),
            "errors": [],
        }
    )
    report.pop("checkpoint", None)
    report.pop("checkpoint_step", None)
    report.pop("checkpoint_state_hash", None)
    report.pop("evaluation_metrics", None)
    report.pop("evaluated_rows", None)
    report.pop("report_hash", None)
    report["report_hash"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_json(audit_path, report)
    print(f"Published verified early-stop checkpoint: {display(checkpoint)}")
    print(f"Wrote {display(audit_path)} status=passed eval_loss={eval_loss}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-B1 early-stop finalization failed: {exc}")
        raise SystemExit(1)
