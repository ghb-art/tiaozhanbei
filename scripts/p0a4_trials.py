#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a4_distillation.json"


class TrialError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ledger_version": "1.0", "created_by": "scripts/p0a4_trials.py", "trials": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("trials"), list):
        raise TrialError("Invalid P0-A4 trial ledger")
    return value


def write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    ledger["updated_ts"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def phase_limit(config: dict[str, Any], phase: str) -> int:
    if phase == "selection170":
        return int(config["gates"]["selection170"]["max_student_versions"])
    if phase == "official_full_student":
        return int(config["gates"]["official_full"]["student_attempts"])
    if phase == "baseline_full":
        return 1
    raise TrialError(f"Unknown trial phase: {phase}")


def reserve(
    ledger: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    version: str,
    resume_reserved: bool = False,
) -> None:
    existing = [item for item in ledger["trials"] if item.get("phase") == phase]
    same = [item for item in existing if item.get("version") == version]
    if resume_reserved and len(same) == 1 and same[0].get("status") == "reserved":
        same[0]["last_resume_ts"] = now()
        return
    if same:
        raise TrialError(f"Trial already exists: {phase}/{version}")
    if phase == "official_full_student" and existing:
        raise TrialError("The one official-full Student attempt has already been reserved")
    if len(existing) >= phase_limit(config, phase):
        raise TrialError(f"Trial limit reached for {phase}: {len(existing)}")
    ledger["trials"].append(
        {"phase": phase, "version": version, "status": "reserved", "reserved_ts": now()}
    )


def finalize(
    ledger: dict[str, Any], phase: str, version: str, status: str, audit_path: Path
) -> None:
    matches = [
        item
        for item in ledger["trials"]
        if item.get("phase") == phase and item.get("version") == version
    ]
    if len(matches) != 1 or matches[0].get("status") != "reserved":
        raise TrialError(f"No unique reserved trial: {phase}/{version}")
    if not audit_path.is_file():
        raise TrialError(f"Missing trial audit: {audit_path}")
    item = matches[0]
    item.update(
        {
            "status": status,
            "completed_ts": now(),
            "audit_path": audit_path.relative_to(ROOT).as_posix(),
            "audit_hash": sha256_file(audit_path),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce P0-A4 selection and formal-test trial limits.")
    parser.add_argument("action", choices=("reserve", "complete", "fail", "show"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=("selection170", "official_full_student", "baseline_full"))
    parser.add_argument("--version")
    parser.add_argument("--audit")
    parser.add_argument("--seal-trace")
    parser.add_argument(
        "--resume-reserved",
        action="store_true",
        help="Resume the exact same unfinished version after infrastructure interruption.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ledger_path = resolve_path(config["artifacts"]["trial_ledger"])
        ledger = load_ledger(ledger_path)
        if args.action == "show":
            print(json.dumps(ledger, indent=2, ensure_ascii=False))
            return 0
        if not args.phase or not args.version:
            raise TrialError("--phase and --version are required")
        if args.action == "reserve":
            reserve(ledger, config, args.phase, args.version, args.resume_reserved)
        else:
            if not args.audit:
                raise TrialError("--audit is required when finalizing a trial")
            finalize(
                ledger,
                args.phase,
                args.version,
                "completed" if args.action == "complete" else "failed",
                resolve_path(args.audit),
            )
            if args.seal_trace:
                trace = resolve_path(args.seal_trace)
                if not trace.is_file():
                    raise TrialError(f"Missing trace to seal: {trace}")
                trace.chmod(0o440)
        write_ledger(ledger_path, ledger)
    except (OSError, KeyError, ValueError, TrialError, json.JSONDecodeError) as exc:
        print(f"P0-A4 trial guard failed: {exc}", file=sys.stderr)
        return 1
    print(f"P0-A4 trial ledger updated: {ledger_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
