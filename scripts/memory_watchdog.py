#!/usr/bin/env python3
"""Run one command behind a host-memory safety guard.

The guard starts the command in its own process group. If host RAM reaches the
configured percentage for N consecutive samples, only that process group is
terminated. GPU memory is intentionally not sampled or used as a trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD_PERCENT = 60.0
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_CONSECUTIVE_SAMPLES = 2
DEFAULT_GRACE_SECONDS = 10.0
MEMORY_LIMIT_EXIT_CODE = 75


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentage(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 100.0:
        raise argparse.ArgumentTypeError("percentage must be in the range (0, 100]")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


@dataclass(frozen=True)
class HostMemory:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    used_percent: float


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp: str
    host: HostMemory


def parse_meminfo(text: str) -> HostMemory:
    values_kib: dict[str, int] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            values_kib[key] = int(fields[0])
        except ValueError:
            continue

    total_kib = values_kib.get("MemTotal")
    available_kib = values_kib.get("MemAvailable")
    if total_kib is None or total_kib <= 0:
        raise ValueError("MemTotal is missing or invalid in /proc/meminfo")
    if available_kib is None:
        fallback_keys = ("MemFree", "Buffers", "Cached", "SReclaimable")
        if not all(key in values_kib for key in fallback_keys):
            raise ValueError("MemAvailable and its fallback fields are missing")
        available_kib = sum(values_kib[key] for key in fallback_keys)

    available_kib = min(max(available_kib, 0), total_kib)
    used_kib = total_kib - available_kib
    return HostMemory(
        total_bytes=total_kib * 1024,
        available_bytes=available_kib * 1024,
        used_bytes=used_kib * 1024,
        used_percent=used_kib * 100.0 / total_kib,
    )


def read_host_memory(meminfo_path: Path = Path("/proc/meminfo")) -> HostMemory:
    return parse_meminfo(meminfo_path.read_text(encoding="utf-8"))


def collect_snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=utc_now(),
        host=read_host_memory(),
    )


def threshold_violations(
    snapshot: MemorySnapshot, threshold_percent: float
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if snapshot.host.used_percent >= threshold_percent:
        violations.append(
            {
                "resource": "host_ram",
                "used_percent": snapshot.host.used_percent,
                "threshold_percent": threshold_percent,
            }
        )
    return violations


@dataclass
class ConsecutiveBreachTracker:
    required_samples: int
    count: int = 0

    def update(self, violations: Sequence[dict[str, Any]]) -> bool:
        self.count = self.count + 1 if violations else 0
        return self.count >= self.required_samples


def snapshot_to_dict(snapshot: MemorySnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def format_usage(snapshot: MemorySnapshot) -> str:
    return f"host_ram={snapshot.host.used_percent:.2f}%"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc_now()} {message}\n")


def default_audit_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (
        PROJECT_ROOT
        / "reports"
        / "runtime"
        / f"memory_watchdog_{stamp}_{os.getpid()}.json"
    )


def update_peaks(audit: dict[str, Any], snapshot: MemorySnapshot) -> None:
    audit["peak_host_used_percent"] = max(
        float(audit.get("peak_host_used_percent", 0.0)),
        snapshot.host.used_percent,
    )


def normalized_process_exit_code(return_code: int) -> int:
    if return_code >= 0:
        return return_code
    return 128 + abs(return_code)


def signal_process_group(process: subprocess.Popen[Any], signum: int) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False


def process_start_time_ticks(pid: int) -> int | None:
    """Return the Linux process start-time identity, or None after exit."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        raise ValueError(f"Invalid /proc/{pid}/stat")
    fields_after_comm = stat[closing_paren + 2 :].split()
    if len(fields_after_comm) <= 19:
        raise ValueError(f"Incomplete /proc/{pid}/stat")
    return int(fields_after_comm[19])


def process_identity_matches(pid: int, start_time_ticks: int) -> bool:
    try:
        return process_start_time_ticks(pid) == start_time_ticks
    except (OSError, ValueError):
        return False


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_existing_process_group(
    process_group_id: int,
    initial_signal: int,
    grace_seconds: float,
) -> dict[str, Any]:
    sent = False
    try:
        os.killpg(process_group_id, initial_signal)
        sent = True
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while sent and process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    forced = False
    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            forced = True
        except ProcessLookupError:
            pass
    return {
        "initial_signal": signal.Signals(initial_signal).name,
        "initial_signal_sent": sent,
        "forced_sigkill": forced,
    }


def terminate_process_group(
    process: subprocess.Popen[Any],
    initial_signal: int,
    grace_seconds: float,
) -> dict[str, Any]:
    sent = signal_process_group(process, initial_signal)
    deadline = time.monotonic() + grace_seconds
    while sent and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    forced = False
    if process.poll() is None:
        forced = signal_process_group(process, signal.SIGKILL)
    try:
        return_code = process.wait(timeout=max(grace_seconds, 1.0))
    except subprocess.TimeoutExpired:
        return_code = process.poll()
    return {
        "initial_signal": signal.Signals(initial_signal).name,
        "initial_signal_sent": sent,
        "forced_sigkill": forced,
        "child_return_code": return_code,
    }


def monitor_existing_process(args: argparse.Namespace) -> int:
    pid = int(args.pid)
    if pid == os.getpid():
        raise SystemExit("memory watchdog cannot monitor itself")
    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        raise SystemExit(f"target process does not exist: pid={pid}")
    if process_group_id == os.getpgrp():
        raise SystemExit(
            "refusing to monitor a target in the watchdog's own process group"
        )
    start_time_ticks = process_start_time_ticks(pid)
    if start_time_ticks is None:
        raise SystemExit(f"target process does not exist: pid={pid}")

    log_path = Path(args.log_file)
    audit_path = Path(args.audit_file) if args.audit_file else default_audit_path()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "mode": "attached_process",
        "created_by": "scripts/memory_watchdog.py",
        "started_at": utc_now(),
        "threshold_percent": args.threshold_percent,
        "interval_seconds": args.interval_seconds,
        "consecutive_samples_required": args.consecutive_samples,
        "grace_seconds": args.grace_seconds,
        "target_pid": pid,
        "process_group_id": process_group_id,
        "target_start_time_ticks": start_time_ticks,
        "audit_file": str(audit_path),
        "log_file": str(log_path),
        "sample_count": 0,
        "peak_host_used_percent": 0.0,
    }
    atomic_write_json(audit_path, audit)
    start_message = (
        f"guard_attached pid={pid} pgid={process_group_id} "
        f"threshold={args.threshold_percent:.2f}% "
        f"interval={args.interval_seconds:.2f}s "
        f"consecutive={args.consecutive_samples} audit={audit_path}"
    )
    append_log(log_path, start_message)
    print(start_message, flush=True)

    tracker = ConsecutiveBreachTracker(args.consecutive_samples)
    interrupted: list[int] = []
    original_handlers: dict[int, Any] = {}

    def remember_signal(signum: int, _frame: Any) -> None:
        interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, remember_signal)
    original_handlers[signal.SIGHUP] = signal.getsignal(signal.SIGHUP)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    try:
        while True:
            if interrupted:
                signum = interrupted[0]
                audit.update(
                    {
                        "status": "monitor_interrupted",
                        "finished_at": utc_now(),
                        "received_signal": signal.Signals(signum).name,
                        "target_terminated": False,
                    }
                )
                atomic_write_json(audit_path, audit)
                append_log(
                    log_path,
                    f"attached_guard_interrupted pid={pid} "
                    f"signal={signal.Signals(signum).name}",
                )
                return 128 + signum

            if not process_identity_matches(pid, start_time_ticks):
                audit.update(
                    {
                        "status": "completed",
                        "finished_at": utc_now(),
                        "completion_reason": "target_exited",
                    }
                )
                atomic_write_json(audit_path, audit)
                append_log(log_path, f"attached_guard_finished pid={pid}")
                print(
                    f"guard_finished pid={pid} audit={audit_path}",
                    flush=True,
                )
                return 0

            try:
                snapshot = collect_snapshot()
            except (OSError, ValueError) as exc:
                audit["probe_error"] = str(exc)
                audit["last_probe_error_at"] = utc_now()
                atomic_write_json(audit_path, audit)
                append_log(log_path, f"host_memory_probe_error error={exc}")
                time.sleep(args.interval_seconds)
                continue

            violations = threshold_violations(snapshot, args.threshold_percent)
            reached_limit = tracker.update(violations)
            audit["sample_count"] += 1
            audit["last_snapshot"] = snapshot_to_dict(snapshot)
            audit["active_violations"] = violations
            audit["consecutive_breach_count"] = tracker.count
            update_peaks(audit, snapshot)
            atomic_write_json(audit_path, audit)

            if violations:
                append_log(
                    log_path,
                    f"threshold_sample pid={pid} "
                    f"count={tracker.count}/{args.consecutive_samples} "
                    f"{format_usage(snapshot)}",
                )
            if reached_limit:
                warning = (
                    f"MEMORY LIMIT: pid={pid} {format_usage(snapshot)}; "
                    "terminating attached process group"
                )
                print(warning, file=sys.stderr, flush=True)
                append_log(log_path, warning)
                action = terminate_existing_process_group(
                    process_group_id, signal.SIGTERM, args.grace_seconds
                )
                audit.update(
                    {
                        "status": "terminated_by_memory_guard",
                        "finished_at": utc_now(),
                        "trigger_snapshot": snapshot_to_dict(snapshot),
                        "trigger_violations": violations,
                        "termination": action,
                        "guard_exit_code": MEMORY_LIMIT_EXIT_CODE,
                    }
                )
                atomic_write_json(audit_path, audit)
                return MEMORY_LIMIT_EXIT_CODE

            time.sleep(args.interval_seconds)
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def run_guarded_command(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("memory_watchdog.py run requires a command after --")

    log_path = Path(args.log_file)
    audit_path = Path(args.audit_file) if args.audit_file else default_audit_path()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "mode": "guarded_command",
        "created_by": "scripts/memory_watchdog.py",
        "started_at": utc_now(),
        "threshold_percent": args.threshold_percent,
        "interval_seconds": args.interval_seconds,
        "consecutive_samples_required": args.consecutive_samples,
        "grace_seconds": args.grace_seconds,
        "command": command,
        "audit_file": str(audit_path),
        "log_file": str(log_path),
        "sample_count": 0,
        "peak_host_used_percent": 0.0,
    }
    atomic_write_json(audit_path, audit)

    try:
        process = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        audit.update(
            {
                "status": "launch_failed",
                "finished_at": utc_now(),
                "error": str(exc),
            }
        )
        atomic_write_json(audit_path, audit)
        append_log(log_path, f"launch_failed command={command!r} error={exc}")
        print(f"Memory guard could not start command: {exc}", file=sys.stderr)
        return 127

    audit.update(
        {
            "status": "running",
            "child_pid": process.pid,
            "process_group_id": process.pid,
        }
    )
    atomic_write_json(audit_path, audit)
    start_message = (
        f"guard_started pid={process.pid} threshold={args.threshold_percent:.2f}% "
        f"interval={args.interval_seconds:.2f}s "
        f"consecutive={args.consecutive_samples} audit={audit_path}"
    )
    append_log(log_path, start_message)
    print(start_message, flush=True)

    tracker = ConsecutiveBreachTracker(args.consecutive_samples)
    original_handlers: dict[int, Any] = {}
    received_signal: list[int] = []

    def remember_signal(signum: int, _frame: Any) -> None:
        received_signal.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        original_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, remember_signal)

    try:
        while True:
            if received_signal:
                signum = received_signal[0]
                action = terminate_process_group(
                    process, signum, args.grace_seconds
                )
                audit.update(
                    {
                        "status": "interrupted",
                        "finished_at": utc_now(),
                        "received_signal": signal.Signals(signum).name,
                        "termination": action,
                    }
                )
                atomic_write_json(audit_path, audit)
                append_log(
                    log_path,
                    f"guard_interrupted pid={process.pid} "
                    f"signal={signal.Signals(signum).name}",
                )
                return 128 + signum

            return_code = process.poll()
            if return_code is not None:
                audit.update(
                    {
                        "status": (
                            "completed" if return_code == 0 else "command_failed"
                        ),
                        "finished_at": utc_now(),
                        "child_return_code": return_code,
                    }
                )
                atomic_write_json(audit_path, audit)
                append_log(
                    log_path,
                    f"guard_finished pid={process.pid} return_code={return_code}",
                )
                print(
                    f"guard_finished pid={process.pid} return_code={return_code} "
                    f"audit={audit_path}",
                    flush=True,
                )
                return normalized_process_exit_code(return_code)

            try:
                snapshot = collect_snapshot()
            except (OSError, ValueError) as exc:
                audit["probe_error"] = str(exc)
                audit["last_probe_error_at"] = utc_now()
                atomic_write_json(audit_path, audit)
                append_log(log_path, f"host_memory_probe_error error={exc}")
                time.sleep(args.interval_seconds)
                continue

            violations = threshold_violations(
                snapshot, args.threshold_percent
            )
            reached_limit = tracker.update(violations)
            audit["sample_count"] += 1
            audit["last_snapshot"] = snapshot_to_dict(snapshot)
            audit["active_violations"] = violations
            audit["consecutive_breach_count"] = tracker.count
            update_peaks(audit, snapshot)
            atomic_write_json(audit_path, audit)

            if violations:
                append_log(
                    log_path,
                    f"threshold_sample pid={process.pid} "
                    f"count={tracker.count}/{args.consecutive_samples} "
                    f"{format_usage(snapshot)}",
                )
            if reached_limit:
                warning = (
                    f"MEMORY LIMIT: pid={process.pid} "
                    f"{format_usage(snapshot)}; terminating guarded process group"
                )
                print(warning, file=sys.stderr, flush=True)
                append_log(log_path, warning)
                action = terminate_process_group(
                    process, signal.SIGTERM, args.grace_seconds
                )
                audit.update(
                    {
                        "status": "terminated_by_memory_guard",
                        "finished_at": utc_now(),
                        "trigger_snapshot": snapshot_to_dict(snapshot),
                        "trigger_violations": violations,
                        "termination": action,
                        "guard_exit_code": MEMORY_LIMIT_EXIT_CODE,
                    }
                )
                atomic_write_json(audit_path, audit)
                return MEMORY_LIMIT_EXIT_CODE

            time.sleep(args.interval_seconds)
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def print_snapshot(args: argparse.Namespace) -> int:
    snapshot = collect_snapshot()
    value = snapshot_to_dict(snapshot)
    value["threshold_percent"] = args.threshold_percent
    value["violations"] = threshold_violations(
        snapshot, args.threshold_percent
    )
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0 if not value["violations"] else MEMORY_LIMIT_EXIT_CODE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Protect one command from host RAM exhaustion. "
            "Only the command launched by this guard can be terminated."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="print current host RAM usage once"
    )
    snapshot_parser.add_argument(
        "--threshold-percent",
        type=percentage,
        default=DEFAULT_THRESHOLD_PERCENT,
    )
    snapshot_parser.set_defaults(func=print_snapshot)

    run_parser = subparsers.add_parser(
        "run", help="run a command and terminate it on sustained memory pressure"
    )
    run_parser.add_argument(
        "--threshold-percent",
        type=percentage,
        default=DEFAULT_THRESHOLD_PERCENT,
        help="host RAM threshold (default: 60)",
    )
    run_parser.add_argument(
        "--interval-seconds",
        type=positive_float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="sampling interval (default: 2 seconds)",
    )
    run_parser.add_argument(
        "--consecutive-samples",
        type=positive_int,
        default=DEFAULT_CONSECUTIVE_SAMPLES,
        help="consecutive over-limit samples before termination (default: 2)",
    )
    run_parser.add_argument(
        "--grace-seconds",
        type=positive_float,
        default=DEFAULT_GRACE_SECONDS,
        help="SIGTERM grace period before SIGKILL (default: 10 seconds)",
    )
    run_parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs" / "memory_watchdog.log"),
    )
    run_parser.add_argument(
        "--audit-file",
        default=None,
        help="JSON status path; defaults to reports/runtime with a timestamp",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to guard; place it after --",
    )
    run_parser.set_defaults(func=run_guarded_command)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="attach the RAM guard to an existing, separate process group",
    )
    monitor_parser.add_argument("--pid", type=positive_int, required=True)
    monitor_parser.add_argument(
        "--threshold-percent",
        type=percentage,
        default=DEFAULT_THRESHOLD_PERCENT,
    )
    monitor_parser.add_argument(
        "--interval-seconds",
        type=positive_float,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    monitor_parser.add_argument(
        "--consecutive-samples",
        type=positive_int,
        default=DEFAULT_CONSECUTIVE_SAMPLES,
    )
    monitor_parser.add_argument(
        "--grace-seconds",
        type=positive_float,
        default=DEFAULT_GRACE_SECONDS,
    )
    monitor_parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / "logs" / "memory_watchdog.log"),
    )
    monitor_parser.add_argument("--audit-file", default=None)
    monitor_parser.set_defaults(func=monitor_existing_process)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
