#!/usr/bin/env python3
"""Build P0-A47 failure-heavy Code continuation data from train-only mining."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a47_code_failure_aligned.json"
SOURCE = ROOT / "data/p0a44/code_train.jsonl"
MINING_AUDIT = ROOT / "reports/audit/p0a47/mining.json"
MINING_TRACE = ROOT / "reports/audit/p0a47/mining_trace.jsonl"
OUTPUT = ROOT / "data/p0a47/code_curriculum.jsonl"
AUDIT = ROOT / "reports/audit/gate_p0a47_curriculum.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["training"]
    mining = json.loads(MINING_AUDIT.read_text(encoding="utf-8"))
    if mining.get("status") != "passed" or int(mining.get("generation_error_count", -1)) != 0:
        raise RuntimeError("P0-A47 mining audit is not passed")
    source_rows = read(SOURCE)
    trace_rows = read(MINING_TRACE)
    source = {str(row["sample_id"]): row for row in source_rows}
    trace = {str(row["sample_id"]): row for row in trace_rows}
    if len(source) != len(source_rows) or set(source) != set(trace):
        raise RuntimeError("P0-A47 source/mining identity mismatch")
    failures = sorted((key for key, row in trace.items() if not row["correct"]), key=str)
    correct = sorted((key for key, row in trace.items() if row["correct"]), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    replay_count = min(int(cfg["correct_replay_limit"]), len(correct), max(1, len(failures) // 2))
    selected = [(sample_id, "failure", float(cfg["failure_weight"])) for sample_id in failures]
    selected += [(sample_id, "correct_replay", float(cfg["correct_replay_weight"])) for sample_id in correct[:replay_count]]
    curriculum: list[dict] = []
    for sample_id, role, weight in selected:
        row = dict(source[sample_id])
        row["sample_id"] = "p0a47/" + str(row["sample_id"]).rsplit("/", 1)[-1]
        row["source"] = "P0-A47-train-only-HumanEval-aligned-failure-repair"
        row["p0a47_role"] = role
        row["training_weight"] = weight
        row["kl_weight"] = float(cfg["kl_weight"])
        curriculum.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in curriculum:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    audit = {
        "gate": "P0-A47-CODE-CURRICULUM",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a47_code_curriculum.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source_rows": len(source_rows),
        "student_failure_rows": len(failures),
        "correct_replay_rows": replay_count,
        "curriculum_rows": len(curriculum),
        "failure_weight": float(cfg["failure_weight"]),
        "correct_replay_weight": float(cfg["correct_replay_weight"]),
        "kl_weight": float(cfg["kl_weight"]),
        "inputs": {str(SOURCE.relative_to(ROOT)): sha(SOURCE), str(MINING_TRACE.relative_to(ROOT)): sha(MINING_TRACE)},
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "rows": len(curriculum), "sha256": sha(OUTPUT)},
        "formal_test_items_loaded": 0,
    }
    audit["report_hash"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT.with_name(f".{AUDIT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, AUDIT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} rows={len(curriculum)} failures={len(failures)} replay={replay_count}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
