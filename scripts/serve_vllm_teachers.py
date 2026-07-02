from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "Qwen--Qwen2.5-14B-Instruct-AWQ"
DEFAULT_VLLM_BIN = ROOT / ".venv" / "bin" / "vllm"

STOP_SIGNAL: int | None = None


@dataclass(frozen=True)
class TeacherSpec:
    gpu: str
    port: int


@dataclass
class TeacherProcess:
    spec: TeacherSpec
    process: subprocess.Popen[bytes]


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_csv_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def parse_ports(values: list[str], base_port: int, count: int) -> list[int]:
    parsed = parse_csv_values(values)
    if not parsed:
        return [base_port + index for index in range(count)]
    ports: list[int] = []
    for value in parsed:
        try:
            port = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid port: {value}") from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"Port out of range: {port}")
        ports.append(port)
    return ports


def port_is_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((bind_host, port)) != 0


def build_command(args: argparse.Namespace, port: int) -> list[str]:
    vllm_bin = Path(args.vllm_bin)
    executable = str(vllm_bin if vllm_bin.is_absolute() else ROOT / vllm_bin)
    if not Path(executable).is_file():
        executable = args.vllm_bin
    model_dir = Path(args.model_dir)
    model_arg = str(model_dir if model_dir.is_absolute() else ROOT / model_dir)
    command = [
        executable,
        "serve",
        model_arg,
        "--quantization",
        args.quantization,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--host",
        args.host,
        "--port",
        str(port),
    ]
    if args.disable_log_requests:
        command.append("--disable-log-requests")
    return command


def request_stop(signum: int, _frame: object) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def start_teacher(args: argparse.Namespace, spec: TeacherSpec) -> TeacherProcess:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = build_command(args, spec.port)
    print(
        f"[START] gpu={spec.gpu} port={spec.port} command={' '.join(command)}",
        flush=True,
    )
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    return TeacherProcess(spec=spec, process=process)


def terminate_processes(processes: list[TeacherProcess], grace_sec: float) -> None:
    running = [item for item in processes if item.process.poll() is None]
    if not running:
        return
    print(f"[STOP] Sending SIGTERM to {len(running)} vLLM process group(s)...", flush=True)
    for item in running:
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if all(item.process.poll() is not None for item in running):
            return
        time.sleep(0.2)

    stuck = [item for item in running if item.process.poll() is None]
    if not stuck:
        return
    print(f"[STOP] {len(stuck)} process group(s) still running; sending SIGKILL.", flush=True)
    for item in stuck:
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start one or more 14B vLLM teacher servers in the foreground; Ctrl+C stops all children."
    )
    parser.add_argument("--gpu", "--gpus", action="append", default=[], help="GPU id(s), repeat or comma-separate.")
    parser.add_argument("--port", "--ports", action="append", default=[], help="Port(s), repeat or comma-separate.")
    parser.add_argument("--base-port", type=int, default=8000, help="First port when --port is omitted.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Local teacher model directory.")
    parser.add_argument("--vllm-bin", default=str(DEFAULT_VLLM_BIN), help="vLLM executable.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--grace-sec", type=float, default=20.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    parser.add_argument("--disable-log-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-port-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print launch plan and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = parse_csv_values(args.gpu) or ["0"]
    try:
        ports = parse_ports(args.port, args.base_port, len(gpus))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len(ports) != len(gpus):
        print(f"GPU count ({len(gpus)}) must match port count ({len(ports)}).", file=sys.stderr)
        return 2
    specs = [TeacherSpec(gpu=gpu, port=port) for gpu, port in zip(gpus, ports)]

    model_dir = Path(args.model_dir)
    resolved_model_dir = model_dir if model_dir.is_absolute() else ROOT / model_dir
    if not resolved_model_dir.is_dir():
        print(f"Missing model directory: {display_path(resolved_model_dir)}", file=sys.stderr)
        return 2

    if not args.skip_port_check and not args.dry_run:
        busy = [spec.port for spec in specs if not port_is_free(args.host, spec.port)]
        if busy:
            print(f"Port(s) already in use: {', '.join(str(port) for port in busy)}", file=sys.stderr)
            return 2

    for spec in specs:
        print(f"[PLAN] gpu={spec.gpu} port={spec.port}")
    if args.dry_run:
        for spec in specs:
            print(f"[DRY-RUN] CUDA_VISIBLE_DEVICES={spec.gpu} {' '.join(build_command(args, spec.port))}")
        return 0

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    processes: list[TeacherProcess] = []
    exit_code = 0
    try:
        for spec in specs:
            processes.append(start_teacher(args, spec))
            time.sleep(1.0)
        print("[READY] vLLM teacher launcher is running. Press Ctrl+C to stop all teacher servers.", flush=True)
        while True:
            if STOP_SIGNAL is not None:
                print(f"[STOP] Received signal {STOP_SIGNAL}.", flush=True)
                break
            failed = [item for item in processes if item.process.poll() not in {None, 0}]
            if failed:
                for item in failed:
                    print(
                        f"[FAIL] gpu={item.spec.gpu} port={item.spec.port} exited with code {item.process.returncode}.",
                        flush=True,
                    )
                exit_code = 1
                break
            if all(item.process.poll() == 0 for item in processes):
                break
            time.sleep(1.0)
    finally:
        terminate_processes(processes, args.grace_sec)
        for item in processes:
            item.process.wait()
        print("[DONE] All vLLM teacher processes stopped.", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
