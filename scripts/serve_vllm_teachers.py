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
    gpus: tuple[str, ...]
    port: int
    tensor_parallel_size: int

    @property
    def gpu_group(self) -> str:
        return ",".join(self.gpus)


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


def parse_gpu_groups(group_values: list[str], legacy_gpu_values: list[str]) -> list[tuple[str, ...]]:
    if group_values and legacy_gpu_values:
        raise ValueError("Use --gpu-group or legacy --gpu/--gpus, not both.")
    if group_values:
        groups = [tuple(parse_csv_values([value])) for value in group_values]
    elif legacy_gpu_values:
        groups = [(value,) for value in parse_csv_values(legacy_gpu_values)]
    else:
        groups = [("0",)]
    if any(not group for group in groups):
        raise ValueError("GPU groups must not be empty.")
    flattened = [gpu for group in groups for gpu in group]
    duplicates = sorted({gpu for gpu in flattened if flattened.count(gpu) > 1})
    if duplicates:
        raise ValueError(f"GPU ids cannot be reused across teacher groups: {duplicates}")
    return groups


def resolve_tensor_parallel_size(value: str, gpu_count: int) -> int:
    raw = value.strip().lower()
    if raw == "auto":
        return gpu_count
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid tensor parallel size: {value}") from exc
    if size <= 0:
        raise ValueError("Tensor parallel size must be positive or 'auto'.")
    if size != gpu_count:
        raise ValueError(
            f"Tensor parallel size ({size}) must equal visible GPUs in each group ({gpu_count})."
        )
    return size


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


def build_command(args: argparse.Namespace, spec: TeacherSpec) -> list[str]:
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
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--host",
        args.host,
        "--port",
        str(spec.port),
    ]
    quantization = str(getattr(args, "quantization", "")).strip()
    if quantization and quantization.lower() not in {"none", "null", "bf16"}:
        command.extend(["--quantization", quantization])
    lora_modules = list(getattr(args, "lora_module", []) or [])
    if lora_modules:
        command.append("--enable-lora")
        command.extend(["--lora-modules", *lora_modules])
    if args.disable_log_requests:
        command.append("--disable-log-requests")
    return command


def request_stop(signum: int, _frame: object) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def start_teacher(args: argparse.Namespace, spec: TeacherSpec) -> TeacherProcess:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_group
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = build_command(args, spec)
    print(
        f"[START] gpu_group={spec.gpu_group} tensor_parallel={spec.tensor_parallel_size} "
        f"port={spec.port} command={' '.join(command)}",
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
    parser.add_argument(
        "--gpu-group",
        action="append",
        default=[],
        help="CUDA devices for one tensor-parallel endpoint, e.g. 0,1,2,3; repeat for multiple endpoints.",
    )
    parser.add_argument(
        "--gpu",
        "--gpus",
        action="append",
        default=[],
        help="Legacy single-GPU endpoint id(s), repeat or comma-separate.",
    )
    parser.add_argument("--port", "--ports", action="append", default=[], help="Port(s), repeat or comma-separate.")
    parser.add_argument("--base-port", type=int, default=8000, help="First port when --port is omitted.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Local teacher model directory.")
    parser.add_argument("--vllm-bin", default=str(DEFAULT_VLLM_BIN), help="vLLM executable.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument(
        "--lora-module",
        action="append",
        default=[],
        help="vLLM mapping name=adapter_path; repeatable. Enables LoRA serving.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--tensor-parallel-size",
        default="auto",
        help="Tensor parallel workers per endpoint; 'auto' uses every GPU in its group.",
    )
    parser.add_argument("--grace-sec", type=float, default=20.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    parser.add_argument("--disable-log-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-port-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print launch plan and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        gpu_groups = parse_gpu_groups(args.gpu_group, args.gpu)
        ports = parse_ports(args.port, args.base_port, len(gpu_groups))
        tensor_parallel_sizes = [
            resolve_tensor_parallel_size(args.tensor_parallel_size, len(group))
            for group in gpu_groups
        ]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len(ports) != len(gpu_groups):
        print(
            f"GPU group count ({len(gpu_groups)}) must match port count ({len(ports)}).",
            file=sys.stderr,
        )
        return 2
    specs = [
        TeacherSpec(gpus=group, port=port, tensor_parallel_size=tensor_parallel_size)
        for group, port, tensor_parallel_size in zip(
            gpu_groups, ports, tensor_parallel_sizes
        )
    ]

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
        print(
            f"[PLAN] gpu_group={spec.gpu_group} "
            f"tensor_parallel={spec.tensor_parallel_size} port={spec.port}"
        )
    if args.dry_run:
        for spec in specs:
            print(
                f"[DRY-RUN] CUDA_VISIBLE_DEVICES={spec.gpu_group} "
                f"{' '.join(build_command(args, spec))}"
            )
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
                        f"[FAIL] gpu_group={item.spec.gpu_group} port={item.spec.port} "
                        f"exited with code {item.process.returncode}.",
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
