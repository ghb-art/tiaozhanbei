#!/usr/bin/env python3
"""Archive the single P0-A19 context-limit failure and seed a safe resume."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_p0a6_internal import sha256_file, sha256_text, write_json_atomic, write_jsonl_atomic


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "reports/audit/p0a19"
AUDIT = AUDIT_ROOT / "base_code.json"
TRACE = AUDIT_ROOT / "base_code_trace.jsonl"
PROGRESS = AUDIT_ROOT / "base_code_trace_progress.jsonl"
ARCHIVE = AUDIT_ROOT / "infra_invalid_http400_ctx1536"
REPAIR_AUDIT = ROOT / "reports/audit/gate_p0a19_infra_repair.json"
EXPECTED_ERROR = (
    "EvaluationError: Endpoint request failed: "
    "http://127.0.0.1:18473/v1/chat/completions: HTTP Error 400: Bad Request"
)


class RepairError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    try:
        if REPAIR_AUDIT.exists() or ARCHIVE.exists() or PROGRESS.exists():
            raise RepairError("P0-A19 infrastructure repair was already applied")
        if not AUDIT.is_file() or not TRACE.is_file():
            raise RepairError("Missing failed P0-A19 base artifacts")
        if any((AUDIT_ROOT / name).exists() for name in ("code_128.json", "code_256.json", "code_selection.json")):
            raise RepairError("Candidate evaluation already started; automatic repair refused")
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        trace = read_jsonl(TRACE)
        errors = [row for row in trace if row.get("generation_error")]
        if (
            audit.get("status") != "failed"
            or audit.get("sample_count") != 255
            or audit.get("generation_error_count") != 1
            or len(trace) != 255
            or len(errors) != 1
            or errors[0].get("generation_error") != EXPECTED_ERROR
        ):
            raise RepairError("Failed audit is not the registered single HTTP-400 context error")
        successful = [row for row in trace if not row.get("generation_error")]
        if len(successful) != 254:
            raise RepairError("Expected exactly 254 reusable rows")
        audit_hash = sha256_file(AUDIT)
        trace_hash = sha256_file(TRACE)
        ARCHIVE.mkdir(parents=True, exist_ok=False)
        os.replace(AUDIT, ARCHIVE / AUDIT.name)
        os.replace(TRACE, ARCHIVE / TRACE.name)
        write_jsonl_atomic(PROGRESS, successful)
        report = {
            "gate": "P0-A19-INFRASTRUCTURE-REPAIR",
            "check_version": "1.0",
            "created_by": "scripts/repair_p0a19_infra.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "reason": "one prompt plus max_tokens exceeded vLLM context 1536",
            "old_context_tokens": 1536,
            "new_context_tokens": 2048,
            "max_tokens_unchanged": 768,
            "prompt_and_weights_unchanged": True,
            "archived_audit": (ARCHIVE / AUDIT.name).relative_to(ROOT).as_posix(),
            "archived_audit_hash": audit_hash,
            "archived_trace": (ARCHIVE / TRACE.name).relative_to(ROOT).as_posix(),
            "archived_trace_hash": trace_hash,
            "valid_rows_reused": len(successful),
            "rows_retried": 1,
            "candidate_evaluations_started": 0,
            "formal_full_opened": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json_atomic(REPAIR_AUDIT, report)
        print("P0-A19 infrastructure repair passed: reused=254 retry=1 context=2048")
        print(f"Wrote {REPAIR_AUDIT.relative_to(ROOT)}")
        return 0
    except (RepairError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A19 infrastructure repair failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
