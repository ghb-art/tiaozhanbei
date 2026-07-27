#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the P0-A4R3 train-only protocol.")
    parser.add_argument(
        "--config", default="configs/p0a4r3_shared_distillation.json"
    )
    parser.add_argument(
        "--output", default="reports/audit/gate_p0a4r3_protocol.json"
    )
    args = parser.parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        errors: list[str] = []
        policy = config["policy"]
        if policy.get("math_status") != "frozen_replay_only":
            errors.append("math_not_frozen")
        if policy.get("adapter_training_allowed") is not False:
            errors.append("adapter_training_enabled")
        if policy.get("shared_training_only") is not True:
            errors.append("shared_training_not_required")
        if int(policy.get("max_preregistered_candidates", 0)) != 2:
            errors.append("candidate_limit_not_two")
        for forbidden in (
            "old_code42_feedback_used",
            "smoke96_item_feedback_used",
            "selection170_feedback_used",
            "formal_full_feedback_used",
        ):
            if policy.get(forbidden) is not False:
                errors.append(f"forbidden_feedback:{forbidden}")
        training = config["training"]["student_shared"]
        rank1 = int(training["lora_rank"])
        rank2 = int(training["candidate_overrides"]["2"]["lora_rank"])
        if (rank1, rank2) != (8, 16):
            errors.append("candidate_ranks_not_8_16")
        teacher = config["models"]["teacher"]
        if teacher.get("request_model_id") != "distill-teacher-v1":
            errors.append("teacher_identity_not_frozen")
        if teacher.get("fallback_request_model_id") != "auto_base_from_endpoint":
            errors.append("teacher_fallback_identity_not_frozen")
        if int(teacher.get("tensor_parallel_size", 0)) != 4:
            errors.append("teacher_not_four_gpu_tp")
        data = config["data"]
        if int(data.get("math_replay_train_groups", 0)) < 1000:
            errors.append("insufficient_math_replay")
        if int(data["code"].get("minimum_unique_train_groups", 0)) < 3500:
            errors.append("insufficient_code_target")
        if int(data["code"].get("new_validation_groups", 0)) < 256:
            errors.append("insufficient_code_validation_target")
        if int(data["nlp"].get("minimum_unique_train_groups", 0)) < 3000:
            errors.append("insufficient_nlp_target")
        if int(data["nlp"].get("new_validation_groups", 0)) < 256:
            errors.append("insufficient_nlp_validation_target")
        if float(data["nlp"].get("minimum_domain_equal_quota_ratio", 0.0)) < 0.8:
            errors.append("nlp_domain_balance_floor_too_low")
        quantization = config["quantization"]
        if quantization.get("type") != "Q4_K_M":
            errors.append("quantization_not_q4_k_m")
        if quantization.get("kv_cache_type") != "q8_0":
            errors.append("kv_cache_not_q8")
        if quantization.get("imatrix_source") != "training_only":
            errors.append("imatrix_not_train_only")
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True).casefold()
        if "reports/sealed/" in serialized:
            errors.append("sealed_trace_referenced_by_training_protocol")
        report: dict[str, Any] = {
            "gate": "P0-A4R3-PROTOCOL",
            "check_version": "1.0",
            "created_by": "scripts/p0a4r3_protocol.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not errors else "failed",
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_hash": sha256_file(config_path),
            "base_model": config["models"]["student"]["local_dir"],
            "candidate_registry": [
                {"candidate_index": 1, "rank": rank1},
                {"candidate_index": 2, "rank": rank2},
            ],
            "math_policy": policy["math_status"],
            "training_mode": "shared_lora_then_merge",
            "adapter_training_allowed": False,
            "quantization": quantization,
            "formal_test_reference_count": 0,
            "errors": errors,
        }
        report["report_hash"] = hashlib.sha256(
            json.dumps(
                report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"P0-A4R3 protocol status={report['status']} output={output.relative_to(ROOT)}")
        return 0 if not errors else 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"P0-A4R3 protocol failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
