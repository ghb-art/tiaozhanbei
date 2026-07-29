#!/usr/bin/env python3
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
DEFAULT_CONFIG = ROOT / "configs/p0a5_capability.json"


class ProtocolError(RuntimeError):
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the single P0-A5 capability protocol.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="reports/audit/gate_p0a5_protocol.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        errors: list[str] = []
        policy = config["policy"]
        if policy.get("baseline_frozen") is not True:
            errors.append("baseline_not_frozen")
        if policy.get("formal_feedback_for_training") is not False:
            errors.append("formal_feedback_enabled")
        if policy.get("adapters_allowed") is not False:
            errors.append("task_adapters_enabled")
        if policy.get("single_shared_lora") is not True:
            errors.append("shared_lora_not_required")
        if int(policy.get("maximum_preregistered_students", 0)) != 2:
            errors.append("candidate_limit_not_two")
        gate = config["gate300"]
        if gate["expected_counts"] != {"math": 100, "code": 100, "nlp": 100}:
            errors.append("gate_not_100_per_domain")
        if float(gate["initial_ratio"]) != 0.78:
            errors.append("initial_threshold_not_78")
        if float(gate["recommended_full_ratio"]) != 0.82:
            errors.append("recommended_threshold_not_82")
        training = config["student_training"]
        if {
            key: float(value) for key, value in training["task_loss_mass"].items()
        } != {"gsm8k": 0.3, "opencodeinstruct": 0.35, "cmmlu": 0.35}:
            errors.append("task_loss_mass_changed")
        preservation = training["math_preservation"]
        if preservation.get("enabled") is not True:
            errors.append("math_preservation_disabled")
        if preservation.get("method") != "base_student_token_kl":
            errors.append("math_preservation_not_kl")
        if config["quantization"]["weight_type"] != "Q4_K_M":
            errors.append("quantization_not_q4_k_m")
        if config["quantization"]["kv_cache_type"] != "q8_0":
            errors.append("kv_cache_not_q8")
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True).casefold()
        for forbidden in ("mbpp", "mmlu_aux", "smoke96", "selection170", "apps", "code_contests"):
            if forbidden in serialized:
                errors.append(f"retired_protocol_reference:{forbidden}")

        artifacts = config["artifacts"]
        manifest_path = resolve_path(artifacts["split_manifest"])
        train_path = resolve_path(artifacts["source_train"])
        validation_path = resolve_path(artifacts["source_validation"])
        gate_path = resolve_path(artifacts["gate_manifest"])
        for path in (manifest_path, train_path, validation_path, gate_path):
            if not path.is_file():
                errors.append(f"missing_artifact:{display_path(path)}")
        counts: dict[str, Any] = {}
        overlaps: dict[str, int] = {}
        hashes: dict[str, str] = {}
        if all(path.is_file() for path in (manifest_path, train_path, validation_path, gate_path)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train = read_jsonl(train_path)
            validation = read_jsonl(validation_path)
            gate_rows = read_jsonl(gate_path)
            counts = {
                "train": dict(Counter(str(row["dataset_key"]) for row in train)),
                "validation": dict(
                    Counter(str(row["dataset_key"]) for row in validation)
                ),
                "gate": dict(Counter(str(row["domain"]) for row in gate_rows)),
            }
            expected_train = {
                "gsm8k": int(config["datasets"]["math"]["train_rows"]),
                "opencodeinstruct": int(config["datasets"]["code"]["train_rows"]),
                "cmmlu": int(config["datasets"]["nlp"]["train_rows"]),
            }
            expected_validation = {
                "gsm8k": int(config["datasets"]["math"]["internal_validation_rows"]),
                "opencodeinstruct": int(
                    config["datasets"]["code"]["internal_validation_rows"]
                ),
                "cmmlu": int(config["datasets"]["nlp"]["internal_validation_rows"]),
            }
            if counts["train"] != expected_train:
                errors.append(f"train_counts_changed:{counts['train']}")
            if counts["validation"] != expected_validation:
                errors.append(f"validation_counts_changed:{counts['validation']}")
            if counts["gate"] != {"math": 100, "code": 100, "nlp": 100}:
                errors.append(f"gate_counts_changed:{counts['gate']}")
            train_ids = {str(row["sample_id"]) for row in train}
            validation_ids = {str(row["sample_id"]) for row in validation}
            gate_ids = {str(row["sample_id"]) for row in gate_rows}
            overlaps = {
                "train_validation": len(train_ids & validation_ids),
                "train_gate": len(train_ids & gate_ids),
                "validation_gate": len(validation_ids & gate_ids),
            }
            if any(overlaps.values()):
                errors.append(f"split_overlap:{overlaps}")
            hashes = {
                "manifest": sha256_file(manifest_path),
                "train": sha256_file(train_path),
                "validation": sha256_file(validation_path),
                "gate": sha256_file(gate_path),
            }
            if manifest.get("source_train_hash") != hashes["train"]:
                errors.append("manifest_train_hash_mismatch")
            if manifest.get("source_validation_hash") != hashes["validation"]:
                errors.append("manifest_validation_hash_mismatch")
            if manifest.get("gate_manifest_hash") != hashes["gate"]:
                errors.append("manifest_gate_hash_mismatch")
        report = {
            "gate": "P0-A5-PROTOCOL",
            "check_version": "1.0",
            "created_by": "scripts/p0a5_protocol.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not errors else "failed",
            "config": display_path(config_path),
            "config_hash": sha256_file(config_path),
            "counts": counts,
            "overlaps": overlaps,
            "artifact_hashes": hashes,
            "active_gate_count": 1,
            "active_gate_name": "gate300",
            "formal_test_reference_count": 0,
            "errors": errors,
        }
        report["report_hash"] = hashlib.sha256(
            json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"P0-A5 protocol status={report['status']} output={display_path(output)}")
        if errors:
            print("\n".join(errors), file=sys.stderr)
        return 0 if report["status"] == "passed" else 1
    except (ProtocolError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A5 protocol failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
