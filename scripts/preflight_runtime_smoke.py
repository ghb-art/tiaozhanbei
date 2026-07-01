from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "preflight" / "runtime_smoke.json"
MIN_PYTHON = (3, 10)
MAX_CAPTURE_LINES = 40

REQUIRED_DIRS = [
    "configs",
    "scripts",
    "docs",
    "reports/preflight",
    "runtime",
    "runtime/state_cache",
    "runtime/outbox",
    "data/datasets",
]

REQUIRED_READABLE_FILES = [
    "README.md",
    "IMPLEMENTATION_PLAN.md",
    "docs/TODO.md",
    "docs/PROJECT_STATUS.md",
    "docs/REVISION_LOG.md",
    "docs/DATASET_SOURCES.md",
    "configs/network_profiles.yaml",
    "configs/workload_profiles.yaml",
    "configs/models.yaml",
    "configs/final_config_dev.yaml",
    "dataset_manifest.template.json",
    "manifest.template.json",
    "conflict_gt_manifest.template.json",
    "scripts/validate_project_structure.py",
    "scripts/validate_dataset_presence.py",
    "scripts/validate_manifest_files.py",
    "scripts/inspect_datasets.py",
    "scripts/preflight_runtime_smoke.py",
    "scripts/verify_gate_cloud.py",
]

TEMPLATE_FILES = [
    "dataset_manifest.template.json",
    "manifest.template.json",
    "conflict_gt_manifest.template.json",
]

SMOKE_COMMANDS = [
    {
        "name": "project_structure",
        "command": [sys.executable, "scripts/validate_project_structure.py"],
    },
    {
        "name": "dataset_presence_allow_empty",
        "command": [sys.executable, "scripts/validate_dataset_presence.py", "--allow-empty"],
    },
    {
        "name": "formal_manifest_allow_missing",
        "command": [sys.executable, "scripts/validate_manifest_files.py", "--allow-missing"],
    },
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def to_relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def tail_lines(text: str, max_lines: int = MAX_CAPTURE_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def record_check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "details": details or {},
    }


def check_python_version() -> tuple[list[dict[str, Any]], list[str]]:
    current = sys.version_info[:3]
    passed = current >= MIN_PYTHON
    details = {
        "required_minimum": ".".join(str(part) for part in MIN_PYTHON),
        "current": ".".join(str(part) for part in current),
        "executable": sys.executable,
    }
    checks = [record_check("python_version", passed, details)]
    if passed:
        ok(f"Python version is {details['current']}")
        return checks, []

    return checks, [f"Python must be >= {details['required_minimum']}, got {details['current']}"]


def check_directories() -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for relative_path in REQUIRED_DIRS:
        path = ROOT / relative_path
        passed = path.is_dir()
        checks.append(record_check(f"directory:{relative_path}", passed, {"path": relative_path}))
        if passed:
            ok(f"Directory exists: {relative_path}")
        else:
            errors.append(f"Missing directory: {relative_path}")

    return checks, errors


def check_readable_files() -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for relative_path in REQUIRED_READABLE_FILES:
        path = ROOT / relative_path
        details: dict[str, Any] = {"path": relative_path}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            checks.append(record_check(f"readable_file:{relative_path}", False, details))
            errors.append(f"Cannot read file: {relative_path} ({exc})")
            continue
        except UnicodeDecodeError as exc:
            checks.append(record_check(f"readable_file:{relative_path}", False, details))
            errors.append(f"File is not UTF-8 readable: {relative_path} ({exc})")
            continue

        details["size_chars"] = len(text)
        passed = bool(text.strip())
        checks.append(record_check(f"readable_file:{relative_path}", passed, details))
        if passed:
            ok(f"Readable file: {relative_path}")
        else:
            errors.append(f"File is empty: {relative_path}")

    return checks, errors


def check_manifest_templates() -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for relative_path in TEMPLATE_FILES:
        path = ROOT / relative_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(record_check(f"template_json:{relative_path}", False, {"path": relative_path}))
            errors.append(f"Manifest template is not valid JSON: {relative_path} ({exc})")
            continue

        passed = isinstance(data, dict) and data.get("template_only") is True
        checks.append(
            record_check(
                f"template_json:{relative_path}",
                passed,
                {
                    "path": relative_path,
                    "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else [],
                },
            )
        )
        if passed:
            ok(f"Manifest template loads: {relative_path}")
        else:
            errors.append(f"Manifest template missing template_only=true: {relative_path}")

    return checks, errors


def run_smoke_command(command_item: dict[str, Any], timeout_sec: int) -> tuple[dict[str, Any], list[str]]:
    name = command_item["name"]
    command = command_item["command"]
    started = datetime.now(timezone.utc)

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        finished = datetime.now(timezone.utc)
    except subprocess.TimeoutExpired as exc:
        details = {
            "command": command,
            "timeout_sec": timeout_sec,
            "started_ts": started.isoformat(),
            "stdout_tail": tail_lines(exc.stdout or ""),
            "stderr_tail": tail_lines(exc.stderr or ""),
        }
        return record_check(f"command:{name}", False, details), [f"Command timed out: {name}"]

    passed = completed.returncode == 0
    details = {
        "command": command,
        "returncode": completed.returncode,
        "started_ts": started.isoformat(),
        "finished_ts": finished.isoformat(),
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
    }
    if passed:
        ok(f"Command passed: {name}")
        return record_check(f"command:{name}", True, details), []

    return record_check(f"command:{name}", False, details), [f"Command failed: {name}"]


def run_smoke_commands(timeout_sec: int) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for command_item in SMOKE_COMMANDS:
        check, command_errors = run_smoke_command(command_item, timeout_sec)
        checks.append(check)
        errors.extend(command_errors)

    return checks, errors


def build_report(checks: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    return {
        "report_version": "0.1",
        "created_by": "scripts/preflight_runtime_smoke.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "summary": {
            "passed": not errors,
            "check_count": len(checks),
            "failed_check_count": sum(1 for check in checks if not check["passed"]),
            "errors": errors,
        },
        "checks": checks,
    }


def write_report(report: dict[str, Any], output: Path) -> str:
    report_for_hash = dict(report)
    canonical = json.dumps(
        report_for_hash,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    report["report_hash"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report["report_hash"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight runtime preflight checks for the project scaffold."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON report path to write.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="Timeout for each subprocess smoke command.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Run checks without writing a report file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    print(f"Project root: {ROOT}")
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for check_group in [
        check_python_version,
        check_directories,
        check_readable_files,
        check_manifest_templates,
    ]:
        group_checks, group_errors = check_group()
        checks.extend(group_checks)
        errors.extend(group_errors)

    command_checks, command_errors = run_smoke_commands(args.timeout_sec)
    checks.extend(command_checks)
    errors.extend(command_errors)

    report = build_report(checks, errors)
    if not args.no_output:
        report_hash = write_report(report, output)
        ok(f"Wrote report: {to_relative(output)}")
        ok(f"Report hash: {report_hash}")

    if errors:
        print()
        print("Runtime smoke failed:")
        for error in errors:
            fail(error)
        return 1

    print()
    print("Runtime smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
