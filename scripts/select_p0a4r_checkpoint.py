#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4_remediation.json"
TASKS = ("humaneval", "cmmlu")
FORBIDDEN_MARKERS = ("humaneval/", "gsm8k/test/", "cmmlu/test/", "official_full", "selection170", "smoke96")


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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"Missing audit: {display_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionError(f"Expected object: {display_path(path)}")
    return value


def load_trace(path: Path, task: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SelectionError(f"Missing internal trace: {display_path(path)}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            identity = " ".join(
                str(row.get(key, "")).lower()
                for key in ("sample_id", "validation_group_id")
            )
            if any(marker in identity for marker in FORBIDDEN_MARKERS):
                raise SelectionError(
                    f"Forbidden evaluation identity in internal trace line {line_number}: {sample_id}"
                )
            if row.get("dataset_key") != task or not sample_id or sample_id in rows:
                raise SelectionError(f"Invalid or duplicate internal trace row: {sample_id}")
            if row.get("generation_error"):
                raise SelectionError(f"Generation error in internal trace: {sample_id}")
            rows[sample_id] = row
    if not rows:
        raise SelectionError("Internal trace is empty")
    return rows


def verify_eval_audit(path: Path, task: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = load_json(path)
    if audit.get("status") != "passed":
        raise SelectionError(f"Internal evaluation did not complete: {display_path(path)}")
    if audit.get("formal_test_labels_used") is not False:
        raise SelectionError("Formal labels are forbidden for remediation selection")
    counts = audit.get("dataset_counts", {})
    if set(counts) != {task} or int(counts.get(task, 0)) <= 0:
        raise SelectionError(f"Audit is not a single-task {task} evaluation")
    if int(audit.get("generation_error_count", -1)) != 0:
        raise SelectionError("Internal evaluation contains generation errors")
    validation_path = resolve_path(str(audit.get("validation_data", "")))
    if "p0a4r_" not in validation_path.name or "internal" not in validation_path.name:
        raise SelectionError("Checkpoint selection requires a P0-A4R train-only internal validation set")
    trace_path = resolve_path(str(audit.get("output_trace", "")))
    if sha256_file(trace_path) != audit.get("output_trace_sha256"):
        raise SelectionError("Internal trace hash does not match its audit")
    return audit, load_trace(trace_path, task)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError as exc:
        raise SelectionError(f"Adapter is not an epoch checkpoint: {display_path(path)}") from exc


def scaled_lora_alpha(value: Any, scale: float) -> int:
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"Invalid source lora_alpha: {value!r}") from exc
    scaled = alpha * scale
    rounded = round(scaled)
    if scaled <= 0 or abs(scaled - rounded) > 1e-9:
        raise SelectionError(
            f"Selected adapter scale does not produce a positive integral lora_alpha: "
            f"{alpha} * {scale} = {scaled}"
        )
    return int(rounded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a P0-A4R adapter by train-only task accuracy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.audit)
    try:
        config_path = resolve_path(args.config)
        config = load_json(config_path)
        policy = config["policy"]
        if policy.get("feedback_source") != "train_only_internal_validation":
            raise SelectionError("Checkpoint selection policy must be train-only")
        for key in ("smoke96_item_feedback_used", "selection170_feedback_used", "formal_full_feedback_used"):
            if policy.get(key) is not False:
                raise SelectionError(f"Forbidden feedback policy enabled: {key}")
        baseline_path = resolve_path(args.baseline_audit)
        baseline_audit, baseline_rows = verify_eval_audit(baseline_path, args.task)
        baseline_ids = set(baseline_rows)
        baseline_correct = {
            sample_id for sample_id, row in baseline_rows.items() if row.get("correct") is True
        }
        candidates = []
        validation_hash = baseline_audit.get("validation_data_sha256")
        for raw_path in args.candidate_audit:
            path = resolve_path(raw_path)
            audit, rows = verify_eval_audit(path, args.task)
            if audit.get("validation_data_sha256") != validation_hash or set(rows) != baseline_ids:
                raise SelectionError(f"Candidate does not use the identical internal rows: {display_path(path)}")
            adapter_meta = audit.get("adapter", {})
            adapter_dir = resolve_path(str(adapter_meta.get("path", "")))
            if not adapter_dir.is_dir():
                raise SelectionError(f"Candidate adapter directory is missing: {display_path(adapter_dir)}")
            candidate_correct = {
                sample_id for sample_id, row in rows.items() if row.get("correct") is True
            }
            additions = sorted(candidate_correct - baseline_correct)
            regressions = sorted(baseline_correct - candidate_correct)
            adapter_scale = float(audit.get("adapter_scale", 1.0))
            if not 0 < adapter_scale <= 1:
                raise SelectionError(
                    f"Candidate adapter scale must be in (0, 1]: {display_path(path)}"
                )
            candidates.append(
                {
                    "audit": display_path(path),
                    "audit_hash": sha256_file(path),
                    "adapter": display_path(adapter_dir),
                    "adapter_fingerprint": adapter_meta,
                    "checkpoint_step": checkpoint_step(adapter_dir),
                    "adapter_scale": adapter_scale,
                    "correct_count": len(candidate_correct),
                    "accuracy": len(candidate_correct) / len(rows),
                    "added_correct_count": len(additions),
                    "regressed_correct_count": len(regressions),
                    "net_improvement": len(candidate_correct) - len(baseline_correct),
                    "added_sample_ids_hash": sha256_json(additions),
                    "regressed_sample_ids_hash": sha256_json(regressions),
                }
            )
        minimum_net = int(config["selection"]["require_net_improvement"])
        max_regressions = int(config["selection"]["max_regressions"])
        eligible = [
            candidate
            for candidate in candidates
            if candidate["net_improvement"] >= minimum_net
            and candidate["regressed_correct_count"] <= max_regressions
        ]
        selected = (
            max(
                eligible,
                key=lambda candidate: (
                    candidate["correct_count"],
                    -candidate["regressed_correct_count"],
                    -candidate["checkpoint_step"],
                ),
            )
            if eligible
            else None
        )
        output_dir = resolve_path(args.output_dir)
        if selected is not None:
            source_dir = resolve_path(selected["adapter"])
            selected_scale = float(selected["adapter_scale"])
            source_config_path = source_dir / "adapter_config.json"
            source_config = load_json(source_config_path)
            if selected_scale != 1.0:
                source_config["lora_alpha"] = scaled_lora_alpha(
                    source_config.get("lora_alpha"),
                    selected_scale,
                )
            weight_name = next(
                (
                    name
                    for name in ("adapter_model.safetensors", "adapter_model.bin")
                    if (source_dir / name).is_file()
                ),
                "",
            )
            if not weight_name:
                raise SelectionError("Selected checkpoint is missing PEFT adapter weights")
            already_published = output_dir.exists() and any(output_dir.iterdir())
            if already_published:
                published_config_path = output_dir / "adapter_config.json"
                published_weight_path = output_dir / weight_name
                if (
                    not published_config_path.is_file()
                    or load_json(published_config_path) != source_config
                    or not published_weight_path.is_file()
                    or sha256_file(published_weight_path)
                    != sha256_file(source_dir / weight_name)
                ):
                    raise SelectionError(
                        "Existing selected adapter does not match the newly selected "
                        f"checkpoint/scale: {display_path(output_dir)}"
                    )
                published = [
                    name
                    for name in ("adapter_config.json", weight_name, "README.md")
                    if (output_dir / name).is_file()
                ]
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_dir / weight_name, output_dir / weight_name)
                published_config_path = output_dir / "adapter_config.json"
                published_config_path.write_text(
                    json.dumps(source_config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                published = ["adapter_config.json", weight_name]
                if (source_dir / "README.md").is_file():
                    shutil.copy2(source_dir / "README.md", output_dir / "README.md")
                    published.append("README.md")
            if "adapter_config.json" not in published or not any(
                name.startswith("adapter_model.") for name in published
            ):
                raise SelectionError("Selected checkpoint is missing PEFT adapter files")
        else:
            published = []
        report = {
            "gate": "P0-A4R-INTERNAL-CHECKPOINT-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/select_p0a4r_checkpoint.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "task": args.task,
            "policy": policy,
            "selection_data": baseline_audit.get("validation_data"),
            "selection_data_hash": validation_hash,
            "selection_sample_count": len(baseline_rows),
            "baseline_audit": display_path(baseline_path),
            "baseline_audit_hash": sha256_file(baseline_path),
            "baseline_correct_count": len(baseline_correct),
            "baseline_accuracy": len(baseline_correct) / len(baseline_rows),
            "minimum_net_improvement": minimum_net,
            "maximum_regressions": max_regressions,
            "candidates": candidates,
            "selected_checkpoint": selected["adapter"] if selected else "",
            "selected_checkpoint_step": selected["checkpoint_step"] if selected else None,
            "selected_adapter_scale": selected["adapter_scale"] if selected else None,
            "selected_output": display_path(output_dir) if selected else "",
            "published_files": published,
            "formal_test_reference_count": 0,
            "failures": [] if selected else ["No checkpoint improved the train-only internal execution/choice gate"],
        }
        report["report_hash"] = sha256_json(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {display_path(output_path)}")
        print(
            f"status={report['status']} baseline={report['baseline_correct_count']}/"
            f"{report['selection_sample_count']} selected={report['selected_checkpoint']}"
        )
        return 0 if selected else 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError, SelectionError) as exc:
        print(f"P0-A4R checkpoint selection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
