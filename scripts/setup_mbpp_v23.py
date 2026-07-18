#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
SOURCE_REVISION = "google-research/google-research:master:mbpp/mbpp.jsonl"
EXPECTED_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
DEFAULT_OUTPUT = ROOT / "data" / "datasets" / "mbpp" / "mbpp.jsonl"
DEFAULT_METADATA = ROOT / "data" / "datasets" / "mbpp" / "SOURCE.json"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_mbpp_dataset_v23.json"


class MbppSetupError(RuntimeError):
    pass


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rows(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MbppSetupError(f"Invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise MbppSetupError(f"Expected object at line {line_number}")
        rows.append(row)
    return rows


def validate_payload(payload: bytes) -> tuple[list[dict[str, Any]], str]:
    digest = sha256_bytes(payload)
    if digest != EXPECTED_SHA256:
        raise MbppSetupError(f"MBPP SHA256 mismatch: expected={EXPECTED_SHA256} actual={digest}")
    rows = load_rows(payload)
    task_ids = [int(row.get("task_id", -1)) for row in rows]
    if len(rows) != 974 or sorted(task_ids) != list(range(1, 975)):
        raise MbppSetupError(
            f"Unexpected MBPP task inventory: rows={len(rows)} min={min(task_ids, default=-1)} "
            f"max={max(task_ids, default=-1)}"
        )
    required = {"task_id", "text", "code", "test_list", "test_setup_code", "challenge_test_list"}
    missing = sorted(required - set.intersection(*(set(row) for row in rows)))
    if missing:
        raise MbppSetupError(f"MBPP rows are missing required fields: {', '.join(missing)}")
    return rows, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and hash-freeze official Google Research MBPP for v23.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_sec <= 0:
        print("--timeout-sec must be positive", file=sys.stderr)
        return 2
    output_path = resolve_path(args.output)
    metadata_path = resolve_path(args.metadata)
    audit_path = resolve_path(args.audit)
    try:
        if output_path.is_file() and not args.force_download:
            payload = output_path.read_bytes()
            acquisition = "existing_verified"
        elif args.verify_only:
            raise MbppSetupError(f"Missing MBPP file for --verify-only: {display_path(output_path)}")
        else:
            request = Request(SOURCE_URL, headers={"User-Agent": "tiaozhanbei-mbpp-v23/1.0"})
            try:
                with urlopen(request, timeout=args.timeout_sec) as response:
                    payload = response.read()
            except (HTTPError, URLError, OSError) as exc:
                raise MbppSetupError(f"Failed to download official MBPP: {exc}") from exc
            acquisition = "downloaded"
        rows, digest = validate_payload(payload)
    except (MbppSetupError, UnicodeDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if acquisition == "downloaded":
        output_path.write_bytes(payload)
    created_ts = datetime.now(timezone.utc).isoformat()
    metadata = {
        "dataset": "Mostly Basic Python Problems (MBPP)",
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_sha256": digest,
        "license": "CC BY 4.0",
        "usage_boundary": {
            "v23_train_task_ids": "601-974",
            "v23_development_task_ids": "511-600",
            "excluded_task_ids": "1-510",
            "human_eval_test_used": False,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "gate": "G-DATA-MBPP-v23",
        "check_version": "1.0",
        "created_by": "scripts/setup_mbpp_v23.py",
        "created_ts": created_ts,
        "status": "passed",
        "acquisition": acquisition,
        "dataset_path": display_path(output_path),
        "metadata_path": display_path(metadata_path),
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_sha256": digest,
        "expected_source_sha256": EXPECTED_SHA256,
        "row_count": len(rows),
        "task_id_min": 1,
        "task_id_max": 974,
        "license": "CC BY 4.0",
        "errors": [],
    }
    report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(metadata_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"rows={len(rows)} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
