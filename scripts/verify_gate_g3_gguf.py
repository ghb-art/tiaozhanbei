from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_CPP = ROOT / "external" / "llama.cpp"
DEFAULT_GGUF = ROOT / "models" / "quantized" / "db4ai-edge-3b-kd-q4_k_m.gguf"
DEFAULT_TEACHER_TRACE = ROOT / "data" / "distill" / "teacher_decision_trace.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_g3_memory_gguf.json"
PROMPT_CACHE_SERVER_ARGS = {
    "-cram",
    "--cache-ram",
    "--cache-idle-slots",
    "--no-cache-idle-slots",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_comma_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def prompt_cache_server_args(cache_ram_mib: int, cache_idle_slots: bool) -> list[str]:
    if cache_ram_mib < 0:
        raise ValueError("--host-prompt-cache-mib must be >= 0 for a bounded memory gate")
    if cache_idle_slots and cache_ram_mib == 0:
        raise ValueError("--cache-idle-slots requires --host-prompt-cache-mib > 0")
    return [
        "--cache-ram",
        str(cache_ram_mib),
        "--cache-idle-slots" if cache_idle_slots else "--no-cache-idle-slots",
    ]


def validate_server_extra_args(values: list[str]) -> None:
    for value in values:
        option = value.split("=", 1)[0]
        if option in PROMPT_CACHE_SERVER_ARGS:
            raise ValueError(
                f"{option} is reserved; use --host-prompt-cache-mib and "
                "--cache-idle-slots/--no-cache-idle-slots"
            )


def select_rows(rows: list[dict[str, Any]], dataset_filter: set[str] | None, total: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if dataset_filter is None or str(row.get("dataset_key", "")) in dataset_filter]
    if not filtered:
        raise RuntimeError("No rows selected")
    return filtered[:total]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def find_one(root: Path, name: str) -> Path:
    candidates = sorted(path for path in root.rglob(name) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"Cannot find {name} under {display_path(root)}")
    executable = [path for path in candidates if path.stat().st_mode & 0o111]
    return executable[0] if executable else candidates[0]


def request_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def health_ok(base_url: str, timeout_sec: float) -> bool:
    try:
        with urlopen(Request(f"{base_url}/health", method="GET"), timeout=timeout_sec) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def process_tree(process: psutil.Process) -> list[psutil.Process]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    return processes


def process_memory_mb(process: psutil.Process) -> tuple[float, float]:
    rss_total = 0
    pss_total = 0
    for item in process_tree(process):
        try:
            rss_total += item.memory_info().rss
            try:
                pss_total += item.memory_full_info().pss
            except (AttributeError, psutil.Error):
                pass
        except psutil.Error:
            pass
    return rss_total / 1_000_000, pss_total / 1_000_000


def rss_mb(process: psutil.Process) -> float:
    """Backward-compatible helper used by older callers and tests."""
    return process_memory_mb(process)[0]


def cgroup_memory_mb(process: psutil.Process) -> float | None:
    try:
        lines = Path(f"/proc/{process.pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except (OSError, psutil.Error):
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0":
            continue
        relative = parts[2].lstrip("/")
        path = Path("/sys/fs/cgroup") / relative / "memory.current"
        try:
            return int(path.read_text(encoding="utf-8").strip()) / 1_000_000
        except (OSError, ValueError):
            return None
    return None


def gpu_memory_mb(process: psutil.Process) -> tuple[float | None, str]:
    pids = {item.pid for item in process_tree(process)}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, completed.stderr.strip()[-300:] or f"returncode={completed.returncode}"
    total = 0.0
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
            used = float(fields[1].split()[0])
        except (ValueError, IndexError):
            continue
        if pid in pids:
            total += used * 1024 * 1024 / 1_000_000
    return total, ""


class MemorySampler:
    def __init__(self, process: psutil.Process, interval_ms: int, include_gpu: bool) -> None:
        self.process = process
        self.interval_sec = interval_ms / 1000
        self.include_gpu = include_gpu
        self.samples: list[dict[str, float | int | None]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="g3-memory-sampler", daemon=True)

    def sample_once(self) -> None:
        try:
            rss, pss = process_memory_mb(self.process)
            cgroup = cgroup_memory_mb(self.process)
            gpu: float | None = 0.0
            if self.include_gpu:
                gpu, gpu_error = gpu_memory_mb(self.process)
                if gpu_error and gpu_error not in self.errors:
                    self.errors.append(gpu_error)
            total = rss + gpu if gpu is not None else None
            self.samples.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "rss_mb_decimal": rss,
                    "pss_mb_decimal": pss,
                    "gpu_memory_mb_decimal": gpu,
                    "total_memory_mb_decimal": total,
                    "cgroup_memory_mb_decimal": cgroup,
                }
            )
        except (psutil.Error, OSError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message not in self.errors:
                self.errors.append(message)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_sec * 4))
        if self.process.is_running():
            self.sample_once()


def build_prompt(row: dict[str, Any]) -> str:
    context = json.dumps(row.get("sample_context", {}), ensure_ascii=False, sort_keys=True)
    return (
        "Return one compact JSON object with fields event_type, risk_attr, action, confidence, "
        "review_intent and short_rationale. No markdown.\n"
        f"Sample context: {context}\nJSON:"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify peak G3 memory using a llama.cpp GGUF backend.")
    parser.add_argument("--gguf", default=str(DEFAULT_GGUF))
    parser.add_argument("--llama-cpp-dir", "--llama_cpp_dir", default=str(DEFAULT_LLAMA_CPP))
    parser.add_argument("--teacher-trace", "--teacher_trace", default=str(DEFAULT_TEACHER_TRACE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dataset", action="append", default=[], help="Dataset filter; repeat or comma-separate.")
    parser.add_argument("--warmup-requests", "--warmup_requests", type=int, default=20)
    parser.add_argument("--measure-requests", "--measure_requests", type=int, default=100)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=96)
    parser.add_argument("--ctx-size", "--ctx_size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent llama.cpp slots; G3 requires 1.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--ubatch-size", "--ubatch_size", type=int, default=128)
    parser.add_argument("--cache-type-k", "--cache_type_k", default="q8_0")
    parser.add_argument("--cache-type-v", "--cache_type_v", default="q8_0")
    parser.add_argument(
        "--host-prompt-cache-mib",
        "--host_prompt_cache_mib",
        type=int,
        default=0,
        help=(
            "Maximum llama-server host prompt cache in MiB. The memory-gate default is 0 "
            "because cached idle-slot states are optional deployment acceleration state."
        ),
    )
    parser.add_argument(
        "--cache-idle-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save idle slots to the host prompt cache (default: disabled).",
    )
    parser.add_argument("--flash-attn", "--flash_attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--no-repack", "--no_repack", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout-sec", "--timeout_sec", type=float, default=120.0)
    parser.add_argument(
        "--max-total-memory-mb",
        "--max_total_memory_mb",
        "--max-rss-mb",
        "--max_rss_mb",
        dest="max_total_memory_mb",
        type=float,
        default=1500.0,
        help="Hard limit for peak process-tree RSS plus device memory (decimal MB).",
    )
    parser.add_argument("--n-gpu-layers", "--n_gpu_layers", type=int, default=0)
    parser.add_argument("--sample-interval-ms", "--sample_interval_ms", type=int, default=50)
    parser.add_argument("--quantization-label", "--quantization_label", default="")
    parser.add_argument("--keep-server-log", "--keep_server_log", default="logs/g3/llama_server.log")
    parser.add_argument("--memory-trace-csv", "--memory_trace_csv", default="")
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Additional audited llama-server argv item; repeat for each item.",
    )
    parser.add_argument(
        "--request-extra-json",
        default="{}",
        help="Additional audited /completion request fields.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup_requests < 0 or args.measure_requests <= 0:
        print("--warmup-requests must be >= 0 and --measure-requests must be > 0", file=sys.stderr)
        return 2
    if args.sample_interval_ms < 10 or args.sample_interval_ms > 100:
        print("--sample-interval-ms must be in [10, 100]", file=sys.stderr)
        return 2
    if args.parallel != 1:
        print("G3 batch_size=1 requires --parallel 1", file=sys.stderr)
        return 2
    try:
        prompt_cache_args = prompt_cache_server_args(
            args.host_prompt_cache_mib,
            args.cache_idle_slots,
        )
        validate_server_extra_args(args.server_extra_arg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    gguf_path = resolve_path(args.gguf)
    llama_cpp_dir = resolve_path(args.llama_cpp_dir)
    teacher_trace_path = resolve_path(args.teacher_trace)
    audit_path = resolve_path(args.audit)
    memory_trace_path = resolve_path(args.memory_trace_csv) if args.memory_trace_csv else audit_path.with_suffix(".samples.csv")
    server_log_path = resolve_path(args.keep_server_log)
    dataset_filter = set(parse_comma_values(args.dataset)) if args.dataset else None
    try:
        request_extra = json.loads(args.request_extra_json)
    except json.JSONDecodeError as exc:
        print(f"Invalid --request-extra-json: {exc}", file=sys.stderr)
        return 2
    if not isinstance(request_extra, dict):
        print("--request-extra-json must decode to an object", file=sys.stderr)
        return 2
    total_requests = args.warmup_requests + args.measure_requests
    rows = select_rows(load_jsonl(teacher_trace_path), dataset_filter, total_requests)
    server_bin = find_one(llama_cpp_dir / "build", "llama-server")
    if not gguf_path.is_file():
        print(f"Missing GGUF file: {display_path(gguf_path)}", file=sys.stderr)
        return 2
    base_url = f"http://{args.host}:{args.port}"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server_bin),
        "--model",
        str(gguf_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--parallel",
        str(args.parallel),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--flash-attn",
        args.flash_attn,
        "--n-gpu-layers",
        str(args.n_gpu_layers),
    ]
    command.extend(prompt_cache_args)
    if args.no_repack:
        command.append("--no-repack")
    command.extend(args.server_extra_arg)
    created_ts = datetime.now(timezone.utc).isoformat()
    log_handle = server_log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    ps_proc = psutil.Process(proc.pid)
    errors: list[dict[str, str]] = []
    warmup_latencies: list[float] = []
    measure_latencies: list[float] = []
    sampler: MemorySampler | None = None
    early_abort_reason = ""
    startup_deadline = time.time() + args.timeout_sec
    try:
        while time.time() < startup_deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
            if health_ok(base_url, 2.0):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("llama-server health check timed out")

        startup_rss = rss_mb(ps_proc)
        started = time.perf_counter()
        if startup_rss > args.max_total_memory_mb:
            sampler = MemorySampler(ps_proc, args.sample_interval_ms, args.n_gpu_layers > 0)
            sampler.start()
            early_abort_reason = (
                "startup_process_tree_rss_over_limit: "
                f"{startup_rss:.6f} > {args.max_total_memory_mb:.6f} MB(decimal)"
            )
            time.sleep(max(0.5, args.sample_interval_ms / 1000 * 5))
            print(f"[EARLY FAIL] {early_abort_reason}", flush=True)
        else:
            if args.warmup_requests == 0:
                sampler = MemorySampler(ps_proc, args.sample_interval_ms, args.n_gpu_layers > 0)
                sampler.start()
            for index, row in enumerate(rows, start=1):
                payload = {
                    "prompt": build_prompt(row),
                    "n_predict": args.max_tokens,
                    "temperature": 0,
                    "cache_prompt": False,
                }
                payload.update(request_extra)
                phase = "warmup" if index <= args.warmup_requests else "measure"
                current_rss: float | None = None
                try:
                    request_started = time.perf_counter()
                    request_json(f"{base_url}/completion", payload, args.timeout_sec)
                    latency_ms = (time.perf_counter() - request_started) * 1000
                    current_rss = rss_mb(ps_proc)
                    if phase == "warmup":
                        warmup_latencies.append(latency_ms)
                    else:
                        measure_latencies.append(latency_ms)
                    print(
                        f"[{phase}] {index}/{len(rows)} {row.get('sample_id')} "
                        f"rss_mb={current_rss:.1f} latency_ms={latency_ms:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    errors.append(
                        {"sample_id": str(row.get("sample_id", "")), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print(f"[FAIL] {index}/{len(rows)} {row.get('sample_id')}: {exc}", flush=True)
                if index == args.warmup_requests and sampler is None:
                    sampler = MemorySampler(ps_proc, args.sample_interval_ms, args.n_gpu_layers > 0)
                    sampler.start()
                if current_rss is not None and current_rss > args.max_total_memory_mb:
                    if sampler is None:
                        sampler = MemorySampler(ps_proc, args.sample_interval_ms, args.n_gpu_layers > 0)
                        sampler.start()
                    early_abort_reason = (
                        "inference_process_tree_rss_over_limit: "
                        f"request={index}, {current_rss:.6f} > {args.max_total_memory_mb:.6f} MB(decimal)"
                    )
                    time.sleep(max(0.5, args.sample_interval_ms / 1000 * 5))
                    print(f"[EARLY FAIL] {early_abort_reason}", flush=True)
                    break
        elapsed = time.perf_counter() - started
    except Exception as exc:
        errors.append({"sample_id": "__server__", "error": f"{type(exc).__name__}: {exc}"})
        startup_rss = rss_mb(ps_proc) if proc.poll() is None else 0.0
        elapsed = 0.0
    finally:
        if sampler is not None:
            sampler.stop()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log_handle.close()

    memory_samples = sampler.samples if sampler is not None else []
    measure_rss = [float(sample["rss_mb_decimal"] or 0.0) for sample in memory_samples]
    measure_pss = [float(sample["pss_mb_decimal"] or 0.0) for sample in memory_samples]
    measure_gpu = [float(sample["gpu_memory_mb_decimal"] or 0.0) for sample in memory_samples]
    measure_total = [
        float(sample["total_memory_mb_decimal"])
        for sample in memory_samples
        if sample["total_memory_mb_decimal"] is not None
    ]
    measure_cgroup = [
        float(sample["cgroup_memory_mb_decimal"])
        for sample in memory_samples
        if sample["cgroup_memory_mb_decimal"] is not None
    ]
    p95_rss = percentile(measure_rss, 0.95)
    peak_total = max(measure_total) if measure_total else 0.0
    sampler_errors = sampler.errors if sampler is not None else []
    complete_requests = len(measure_latencies) == args.measure_requests
    status = (
        "passed"
        if memory_samples
        and measure_total
        and complete_requests
        and not errors
        and not sampler_errors
        and peak_total <= args.max_total_memory_mb
        else "failed"
    )
    memory_trace_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_trace_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "elapsed_ms",
            "rss_mb_decimal",
            "pss_mb_decimal",
            "gpu_memory_mb_decimal",
            "total_memory_mb_decimal",
            "cgroup_memory_mb_decimal",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        first_ns = int(memory_samples[0]["monotonic_ns"]) if memory_samples else 0
        for sample in memory_samples:
            writer.writerow(
                {
                    "elapsed_ms": (int(sample["monotonic_ns"]) - first_ns) / 1_000_000 if first_ns else 0.0,
                    **{key: sample.get(key) for key in fieldnames if key != "elapsed_ms"},
                }
            )
    audit = {
        "gate": "G3-memory-gguf",
        "check_version": "2.1",
        "created_by": "scripts/verify_gate_g3_gguf.py",
        "created_ts": created_ts,
        "status": status,
        "gguf_path": display_path(gguf_path),
        "gguf_hash": sha256_file(gguf_path),
        "gguf_bytes": gguf_path.stat().st_size,
        "llama_cpp_dir": display_path(llama_cpp_dir),
        "server_bin": display_path(server_bin),
        "server_command": command,
        "server_extra_args": args.server_extra_arg,
        "request_extra_hash": sha256_text(
            json.dumps(request_extra, ensure_ascii=False, sort_keys=True)
        ),
        "server_log": display_path(server_log_path),
        "teacher_trace_path": display_path(teacher_trace_path),
        "teacher_trace_hash": sha256_file(teacher_trace_path),
        "quantization_backend": "llama_cpp_gguf",
        "quantization_label": args.quantization_label,
        "quantization_scope_note": "The supplied standalone GGUF is measured as-is and identified by its preparation audit.",
        "warmup_requests": args.warmup_requests,
        "successful_warmup_requests": len(warmup_latencies),
        "measure_requests": len(measure_latencies),
        "requested_measure_requests": args.measure_requests,
        "max_tokens": args.max_tokens,
        "ctx_size": args.ctx_size,
        "threads": args.threads,
        "parallel": args.parallel,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "host_prompt_cache_mib": args.host_prompt_cache_mib,
        "host_prompt_cache_enabled": args.host_prompt_cache_mib > 0,
        "cache_idle_slots": args.cache_idle_slots,
        "host_prompt_cache_policy": (
            "bounded_enabled" if args.host_prompt_cache_mib > 0 else "disabled_for_edge_memory_gate"
        ),
        "flash_attn": args.flash_attn,
        "repack_enabled": not args.no_repack,
        "n_gpu_layers": args.n_gpu_layers,
        "memory_gate_metric": "peak_process_tree_rss_plus_device_memory_mb_decimal",
        "sample_interval_ms": args.sample_interval_ms,
        "memory_sample_count": len(memory_samples),
        "memory_trace_csv": display_path(memory_trace_path),
        "memory_trace_hash": sha256_file(memory_trace_path),
        "startup_rss_mb_decimal": startup_rss,
        "startup_over_limit": startup_rss > args.max_total_memory_mb,
        "early_abort_reason": early_abort_reason,
        "max_total_memory_mb_decimal": args.max_total_memory_mb,
        "max_rss_mb_decimal": args.max_total_memory_mb,
        "p50_rss_mb_decimal": statistics.median(measure_rss) if measure_rss else 0.0,
        "p95_rss_mb_decimal": p95_rss,
        "p95_rss_mib_binary": p95_rss * 1_000_000 / (1024 * 1024),
        "peak_rss_mb_decimal": max(measure_rss) if measure_rss else 0.0,
        "p95_pss_mb_decimal": percentile(measure_pss, 0.95),
        "peak_pss_mb_decimal": max(measure_pss) if measure_pss else 0.0,
        "p95_gpu_memory_mb_decimal": percentile(measure_gpu, 0.95),
        "peak_gpu_memory_mb_decimal": max(measure_gpu) if measure_gpu else 0.0,
        "p95_total_memory_mb_decimal": percentile(measure_total, 0.95),
        "peak_total_memory_mb_decimal": peak_total,
        "p95_cgroup_memory_mb_decimal": percentile(measure_cgroup, 0.95),
        "peak_cgroup_memory_mb_decimal": max(measure_cgroup) if measure_cgroup else 0.0,
        "mean_latency_ms": statistics.mean(measure_latencies) if measure_latencies else 0.0,
        "p95_latency_ms": percentile(measure_latencies, 0.95),
        "elapsed_sec": elapsed,
        "selected_sample_ids_hash": sha256_text("\n".join(str(row.get("sample_id", "")) for row in rows) + "\n"),
        "errors": errors,
        "sampler_errors": sampler_errors,
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)
    print(f"Wrote {display_path(audit_path)}")
    print(f"status={status}")
    print(f"peak_total_memory_mb_decimal={peak_total:.2f}")
    print(f"p95_rss_mb_decimal={p95_rss:.2f}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
