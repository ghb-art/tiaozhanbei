#!/usr/bin/env python3
"""Build a train-only curriculum from Student execution failures and replay."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from build_p0a21_data import sha256_file, sha256_text, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a25_code_failure_mining.json"
DATA_AUDIT = ROOT / "reports/audit/gate_p0a25_data.json"
MINING_AUDIT = ROOT / "reports/audit/p0a25/mining.json"
MINING_TRACE = ROOT / "reports/audit/p0a25/mining_trace.jsonl"
TRAIN_POOL = ROOT / "data/p0a25/code_train_pool.jsonl"
OUTPUT = ROOT / "data/p0a25/code_curriculum.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a25_curriculum.json"


class CurriculumError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise CurriculumError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        data_audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
        mining_audit = json.loads(MINING_AUDIT.read_text(encoding="utf-8"))
        if data_audit.get("status") != "passed":
            raise CurriculumError("P0-A25 data audit did not pass")
        expected_mining = {
            "status": "passed",
            "gate": "P0-A25-TRAIN-ONLY-FAILURE-MINING",
            "sample_count": 6000,
            "generation_error_count": 0,
            "thinking": "off",
            "max_tokens": 768,
            "gate300_loaded": False,
            "formal_full_loaded": False,
        }
        for key, expected in expected_mining.items():
            if mining_audit.get(key) != expected:
                raise CurriculumError(
                    f"Mining audit {key}={mining_audit.get(key)!r}, expected {expected!r}"
                )
        if mining_audit.get("trace_hash") != sha256_file(MINING_TRACE):
            raise CurriculumError("Mining trace hash mismatch")
        pool = read_jsonl(TRAIN_POOL)
        trace = read_jsonl(MINING_TRACE)
        if len(pool) != 6000 or len(trace) != 6000:
            raise CurriculumError("Unexpected pool/trace size")
        pool_by_id = {str(row["sample_id"]): row for row in pool}
        trace_by_id = {str(row["sample_id"]): row for row in trace}
        if len(pool_by_id) != 6000 or set(pool_by_id) != set(trace_by_id):
            raise CurriculumError("Mining trace does not match train pool")
        failures = sorted(
            (sample_id for sample_id, row in trace_by_id.items() if not row["correct"]),
            key=lambda value: sha256_text(f"p0a25:failure:{value}"),
        )
        correct = sorted(
            (sample_id for sample_id, row in trace_by_id.items() if row["correct"]),
            key=lambda value: sha256_text(f"p0a25:replay:{value}"),
        )
        replay_count = min(len(correct), round(len(failures) * 0.5))
        replay = correct[:replay_count]
        if len(failures) < 1000 or replay_count < 500:
            raise CurriculumError(
                f"Insufficient failure/replay curriculum: {len(failures)}/{replay_count}"
            )
        curriculum: list[dict] = []
        for sample_id, role, weight in (
            *((value, "student_execution_failure", 2.0) for value in failures),
            *((value, "student_correct_replay", 1.0) for value in replay),
        ):
            row = dict(pool_by_id[sample_id])
            row["training_weight"] = weight
            row["kl_weight"] = 0.10
            row["p0a25_role"] = role
            curriculum.append(row)
        curriculum.sort(key=lambda row: sha256_text(f"p0a25:curriculum:{row['sample_id']}"))
        write_jsonl(OUTPUT, curriculum)
        audit = {
            "gate": "P0-A25-FAILURE-CURRICULUM",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0a25_curriculum.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "source_rows": len(pool),
            "student_correct_count": len(correct),
            "student_failure_count": len(failures),
            "student_accuracy": len(correct) / len(pool),
            "failure_rows": len(failures),
            "correct_replay_rows": replay_count,
            "curriculum_rows": len(curriculum),
            "failure_training_weight": 2.0,
            "correct_replay_training_weight": 1.0,
            "kl_weight": 0.10,
            "per_item_feedback_scope": "train_only_p0a25_mining_pool",
            "p0a24_gate_trace_loaded": False,
            "formal_full_loaded": False,
            "inputs": {
                CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
                DATA_AUDIT.relative_to(ROOT).as_posix(): sha256_file(DATA_AUDIT),
                MINING_AUDIT.relative_to(ROOT).as_posix(): sha256_file(MINING_AUDIT),
                MINING_TRACE.relative_to(ROOT).as_posix(): sha256_file(MINING_TRACE),
                TRAIN_POOL.relative_to(ROOT).as_posix(): sha256_file(TRAIN_POOL),
            },
            "output": {
                "path": OUTPUT.relative_to(ROOT).as_posix(),
                "rows": len(curriculum),
                "sha256": sha256_file(OUTPUT),
            },
            "errors": [],
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
        write_json(AUDIT, audit)
        print(
            f"P0-A25 curriculum passed failures={len(failures)} "
            f"replay={replay_count} rows={len(curriculum)}"
        )
        print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
        return 0
    except (CurriculumError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A25 curriculum failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
