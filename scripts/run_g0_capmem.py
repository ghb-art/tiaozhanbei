from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "g0_capmem_candidates.json"
DEFAULT_OUTPUT = ROOT / "reports" / "audit" / "gate_g0_capmem.json"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if log_path is None:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout_tail = "\n".join(completed.stdout.splitlines()[-40:])
        stderr_tail = "\n".join(completed.stderr.splitlines()[-40:])
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        stdout_tail = ""
        stderr_tail = ""
    return {
        "command": command,
        "started_ts": started,
        "finished_ts": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "log_path": display_path(log_path) if log_path else "",
    }


def find_llama_server(llama_cpp_dir: Path) -> Path:
    candidates = sorted(path for path in (llama_cpp_dir / "build").rglob("llama-server") if path.is_file())
    executable = [path for path in candidates if path.stat().st_mode & 0o111]
    if not executable:
        raise FileNotFoundError(f"Cannot find executable llama-server under {display_path(llama_cpp_dir)}")
    return executable[0]


def health_ok(base_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        with urlopen(Request(f"{base_url}/health", method="GET"), timeout=timeout_sec) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def prepare_candidate(candidate: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/prepare_edge_gguf.py",
        "--merged-hf-dir",
        str(candidate["hf_model_dir"]),
        "--llama-cpp-dir",
        str(common.get("llama_cpp_dir", "external/llama.cpp")),
        "--f16-gguf",
        str(candidate["f16_gguf"]),
        "--quantized-gguf",
        str(candidate["gguf"]),
        "--audit",
        str(candidate["prepare_audit"]),
        "--quant-type",
        str(candidate["quant_type"]),
        "--skip-f16-if-exists",
        "--skip-quantized-if-exists",
    ]
    max_bytes = int(candidate.get("max_artifact_bytes", 0))
    if max_bytes > 0:
        command.extend(["--max-quantized-bytes", str(max_bytes)])
    if candidate.get("imatrix"):
        command.extend(["--imatrix", str(candidate["imatrix"])])
    return run_command(command)


def prepare_importance_matrix(common: dict[str, Any]) -> list[dict[str, Any]]:
    imatrix_value = common.get("imatrix", "")
    if not imatrix_value:
        return []
    imatrix = resolve_path(str(imatrix_value))
    if imatrix.is_file():
        return [{"stage": "imatrix", "returncode": 0, "status": "reused", "path": display_path(imatrix)}]

    llama_cpp_dir = resolve_path(str(common.get("llama_cpp_dir", "external/llama.cpp")))
    f16 = resolve_path(str(common["imatrix_f16_gguf"]))
    hf_model_dir = resolve_path(str(common["imatrix_hf_model_dir"]))
    calibration = resolve_path(str(common.get("imatrix_calibration", "data/distill/g0_imatrix_calibration.txt")))
    calibration_audit = resolve_path(str(common.get("imatrix_calibration_audit", "reports/audit/g0_imatrix_calibration.json")))
    commands: list[dict[str, Any]] = []

    build_calibration = [sys.executable, "scripts/build_imatrix_calibration.py"]
    for source in common.get("imatrix_sources", []):
        build_calibration.extend(["--source", str(source)])
    build_calibration.extend(
        [
            "--output",
            str(calibration),
            "--audit",
            str(calibration_audit),
            "--rows-per-source",
            str(common.get("imatrix_rows_per_source", 256)),
        ]
    )
    commands.append({"stage": "imatrix_calibration", **run_command(build_calibration)})
    if commands[-1]["returncode"] != 0:
        return commands

    if not f16.is_file():
        convert = llama_cpp_dir / "convert_hf_to_gguf.py"
        f16.parent.mkdir(parents=True, exist_ok=True)
        conversion = [sys.executable, str(convert), str(hf_model_dir), "--outfile", str(f16), "--outtype", "f16"]
        commands.append({"stage": "imatrix_f16", **run_command(conversion)})
        if commands[-1]["returncode"] != 0:
            return commands

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    if not imatrix_bin.is_file():
        cmake = ROOT / ".venv" / "bin" / "cmake"
        build = [str(cmake), "--build", str(llama_cpp_dir / "build"), "--config", "Release", "--target", "llama-imatrix"]
        commands.append({"stage": "imatrix_binary", **run_command(build)})
        if commands[-1]["returncode"] != 0:
            return commands

    imatrix.parent.mkdir(parents=True, exist_ok=True)
    partial_imatrix = imatrix.with_suffix(imatrix.suffix + ".partial")
    if partial_imatrix.exists():
        partial_imatrix.unlink()
    command = [
        str(imatrix_bin),
        "--model",
        str(f16),
        "--file",
        str(calibration),
        "--output",
        str(partial_imatrix),
        "--chunks",
        str(common.get("imatrix_chunks", 64)),
        "--ctx-size",
        str(common.get("ctx_size", 512)),
        "--threads",
        str(common.get("imatrix_threads", common.get("threads", 8))),
        "--no-ppl",
        "--output-frequency",
        str(common.get("imatrix_output_frequency", 16)),
    ]
    result = run_command(command, resolve_path(str(common.get("imatrix_log", "logs/g0/imatrix.log"))))
    if result["returncode"] == 0 and partial_imatrix.is_file():
        os.replace(partial_imatrix, imatrix)
        result["output"] = display_path(imatrix)
        result["output_hash"] = sha256_file(imatrix)
    commands.append({"stage": "imatrix", **result})
    return commands


def measure_candidate(candidate: dict[str, Any], common: dict[str, Any], index: int) -> dict[str, Any]:
    port = int(common.get("memory_port_base", 18200)) + index
    command = [
        sys.executable,
        "scripts/verify_gate_g3_gguf.py",
        "--gguf",
        str(candidate["gguf"]),
        "--llama-cpp-dir",
        str(common.get("llama_cpp_dir", "external/llama.cpp")),
        "--teacher-trace",
        str(common.get("memory_trace", "data/distill/teacher_decision_trace.jsonl")),
        "--audit",
        str(candidate["memory_audit"]),
        "--warmup-requests",
        str(common.get("warmup_requests", 20)),
        "--measure-requests",
        str(common.get("measure_requests", 100)),
        "--max-tokens",
        str(common.get("memory_max_tokens", 96)),
        "--ctx-size",
        str(common.get("ctx_size", 512)),
        "--threads",
        str(common.get("threads", 8)),
        "--parallel",
        str(common.get("parallel", 1)),
        "--batch-size",
        str(common.get("batch_size", 128)),
        "--ubatch-size",
        str(common.get("ubatch_size", 128)),
        "--cache-type-k",
        str(common.get("cache_type_k", "q8_0")),
        "--cache-type-v",
        str(common.get("cache_type_v", "q8_0")),
        "--flash-attn",
        str(common.get("flash_attn", "on")),
        "--port",
        str(port),
        "--max-total-memory-mb",
        str(common.get("max_memory_mb", 1500)),
        "--sample-interval-ms",
        str(common.get("sample_interval_ms", 50)),
        "--n-gpu-layers",
        str(common.get("n_gpu_layers", 0)),
        "--quantization-label",
        str(candidate.get("quant_type", "")),
        "--keep-server-log",
        str(candidate.get("memory_log", f"logs/g0/{candidate['name']}_memory_server.log")),
    ]
    if not bool(common.get("repack", False)):
        command.append("--no-repack")
    return run_command(command)


def capability_smoke_candidate(candidate: dict[str, Any], common: dict[str, Any], index: int) -> dict[str, Any]:
    llama_cpp_dir = resolve_path(str(common.get("llama_cpp_dir", "external/llama.cpp")))
    server_bin = find_llama_server(llama_cpp_dir)
    gguf = resolve_path(str(candidate["gguf"]))
    port = int(common.get("capability_port_base", 18300)) + index
    base_url = f"http://127.0.0.1:{port}"
    server_log = resolve_path(str(candidate.get("capability_server_log", f"logs/g0/{candidate['name']}_capability_server.log")))
    server_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(server_bin),
        "--model",
        str(gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(common.get("capability_ctx_size", common.get("ctx_size", 512))),
        "--threads",
        str(common.get("threads", 8)),
        "--parallel",
        str(common.get("parallel", 1)),
        "--batch-size",
        str(common.get("batch_size", 128)),
        "--ubatch-size",
        str(common.get("ubatch_size", 128)),
        "--cache-type-k",
        str(common.get("cache_type_k", "q8_0")),
        "--cache-type-v",
        str(common.get("cache_type_v", "q8_0")),
        "--flash-attn",
        str(common.get("flash_attn", "on")),
        "--n-gpu-layers",
        str(common.get("capability_n_gpu_layers", common.get("n_gpu_layers", 0))),
    ]
    if not bool(common.get("repack", False)):
        command.append("--no-repack")
    command.extend(str(value) for value in candidate.get("capability_server_args", []))
    created = datetime.now(timezone.utc).isoformat()
    handle = server_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    errors: list[str] = []
    eval_result: dict[str, Any] = {}
    try:
        deadline = time.time() + float(common.get("server_start_timeout_sec", 120))
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
            if health_ok(base_url):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("llama-server health check timed out")

        eval_command = [
            sys.executable,
            "scripts/evaluate_chapter2_capability.py",
            "--student-url",
            base_url,
            "--student-model-id",
            candidate["name"],
            "--dataset",
            "gsm8k,humaneval,cmmlu",
            "--sample-limit-per-dataset",
            str(common.get("smoke_samples_per_dataset", 34)),
            "--output-trace",
            str(candidate["capability_trace"]),
            "--audit",
            str(candidate["capability_audit"]),
            "--max-new-tokens",
            str(candidate.get("capability_max_tokens", common.get("capability_max_tokens", 384))),
            "--humaneval-timeout-sec",
            str(common.get("humaneval_timeout_sec", 5)),
            "--timeout-sec",
            str(common.get("capability_timeout_sec", 240)),
            "--prompt-style-map",
            "gsm8k=v11,humaneval=v11,cmmlu=v11",
        ]
        max_tokens_map = candidate.get(
            "capability_max_tokens_map", common.get("capability_max_tokens_map", {})
        )
        if max_tokens_map:
            eval_command.extend(
                [
                    "--max-new-tokens-map",
                    ",".join(f"{key}={value}" for key, value in sorted(max_tokens_map.items())),
                ]
            )
        fail_fast_map = candidate.get(
            "capability_fail_fast_min_accuracy_map",
            common.get("capability_fail_fast_min_accuracy_map", {}),
        )
        if fail_fast_map:
            eval_command.extend(
                [
                    "--fail-fast-min-accuracy-map",
                    ",".join(f"{key}={value}" for key, value in sorted(fail_fast_map.items())),
                ]
            )
        eval_result = run_command(eval_command, resolve_path(str(candidate.get("capability_log", f"logs/g0/{candidate['name']}_capability.log"))))
        if eval_result["returncode"] != 0:
            errors.append("capability evaluator failed")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        handle.close()
    return {
        "command": command,
        "created_ts": created,
        "server_log": display_path(server_log),
        "eval": eval_result,
        "errors": errors,
        "returncode": 0 if not errors else 1,
    }


def ratios_from_trace(edge_trace: Path, cloud_trace: Path, min_ratio: float) -> dict[str, Any]:
    edge_rows = load_jsonl(edge_trace)
    cloud_by_id = {str(row["sample_id"]): row for row in load_jsonl(cloud_trace)}
    edge_counts = Counter(str(row.get("dataset_key", "")) for row in edge_rows)
    edge_correct = Counter(str(row.get("dataset_key", "")) for row in edge_rows if row.get("correct") is True)
    cloud_counts: Counter[str] = Counter()
    cloud_correct: Counter[str] = Counter()
    missing_ids: list[str] = []
    for row in edge_rows:
        sample_id = str(row.get("sample_id", ""))
        cloud = cloud_by_id.get(sample_id)
        if cloud is None:
            missing_ids.append(sample_id)
            continue
        dataset = str(row.get("dataset_key", ""))
        cloud_counts[dataset] += 1
        if cloud.get("correct") is True:
            cloud_correct[dataset] += 1

    dataset_to_ratio = {"gsm8k": "math_ratio", "humaneval": "code_ratio", "cmmlu": "nlp_ratio"}
    ratios: dict[str, float] = {}
    edge_accuracy: dict[str, float] = {}
    cloud_accuracy: dict[str, float] = {}
    for dataset, ratio_name in dataset_to_ratio.items():
        edge_acc = edge_correct[dataset] / edge_counts[dataset] if edge_counts[dataset] else 0.0
        cloud_acc = cloud_correct[dataset] / cloud_counts[dataset] if cloud_counts[dataset] else 0.0
        edge_accuracy[dataset] = edge_acc
        cloud_accuracy[dataset] = cloud_acc
        ratios[ratio_name] = edge_acc / cloud_acc if cloud_acc else 0.0
    overall = sum(min(value, 1.0) for value in ratios.values()) / len(ratios)
    passed = not missing_ids and all(value >= min_ratio for value in ratios.values()) and overall >= min_ratio
    return {
        "source": "matched_smoke_trace",
        "edge_trace": display_path(edge_trace),
        "edge_trace_hash": sha256_file(edge_trace),
        "cloud_trace": display_path(cloud_trace),
        "cloud_trace_hash": sha256_file(cloud_trace),
        "sample_count": len(edge_rows),
        "missing_cloud_sample_count": len(missing_ids),
        "missing_cloud_sample_ids_hash": sha256_text("\n".join(missing_ids) + "\n"),
        "edge_accuracy_by_dataset": edge_accuracy,
        "cloud_accuracy_by_dataset": cloud_accuracy,
        "ratios": ratios,
        "overall_r_cap": overall,
        "passed": passed,
    }


def capability_result(candidate: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    min_ratio = float(common.get("min_capability_ratio", 0.8))
    trace_value = candidate.get("capability_trace", "")
    trace = resolve_path(str(trace_value)) if trace_value else None
    cloud_trace = resolve_path(str(common.get("cloud_trace", "data/eval/chapter2_capability_cloud_v17.jsonl")))
    if trace is not None and trace.is_file() and cloud_trace.is_file():
        return ratios_from_trace(trace, cloud_trace, min_ratio)

    audit_value = candidate.get("existing_capability_audit", "")
    audit_path = resolve_path(str(audit_value)) if audit_value else None
    if audit_path is not None and audit_path.is_file():
        audit = load_json(audit_path)
        ratios = {key: float(value) for key, value in audit.get("ratios", {}).items()}
        overall = float(audit.get("overall_r_cap", 0.0))
        passed = all(ratios.get(key, 0.0) >= min_ratio for key in ("math_ratio", "code_ratio", "nlp_ratio")) and overall >= min_ratio
        return {
            "source": "existing_capability_audit",
            "audit": display_path(audit_path),
            "audit_hash": sha256_file(audit_path),
            "ratios": ratios,
            "overall_r_cap": overall,
            "passed": passed,
        }
    return {"source": "missing", "passed": None}


def memory_result(candidate: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    audit_value = candidate.get("memory_audit", "")
    audit_path = resolve_path(str(audit_value)) if audit_value else None
    if audit_path is None or not audit_path.is_file():
        return {"source": "missing", "passed": None}
    audit = load_json(audit_path)
    peak = audit.get("peak_total_memory_mb_decimal")
    metric = str(audit.get("memory_gate_metric", ""))
    complete_metric = peak is not None and metric == "peak_process_tree_rss_plus_device_memory_mb_decimal"
    passed = complete_metric and float(peak) <= float(common.get("max_memory_mb", 1500)) and audit.get("status") == "passed"
    return {
        "source": "memory_audit",
        "audit": display_path(audit_path),
        "audit_hash": sha256_file(audit_path),
        "audit_status": audit.get("status", ""),
        "metric": metric,
        "complete_metric": complete_metric,
        "peak_total_memory_mb_decimal": float(peak) if peak is not None else None,
        "p95_total_memory_mb_decimal": audit.get("p95_total_memory_mb_decimal"),
        "passed": passed,
    }


def candidate_result(candidate: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    gguf_value = candidate.get("gguf", "")
    gguf = resolve_path(str(gguf_value)) if gguf_value else None
    artifact_exists = bool(gguf and gguf.is_file())
    artifact_bytes = gguf.stat().st_size if artifact_exists and gguf is not None else 0
    capability = capability_result(candidate, common)
    memory = memory_result(candidate, common)
    cap_pass = capability["passed"]
    mem_pass = memory["passed"]
    decision_status = str(candidate.get("decision_status", "")).strip().lower()
    pruned = decision_status == "pruned"
    feasible = not pruned and artifact_exists and cap_pass is True and mem_pass is True
    if pruned:
        status = "pruned"
    elif feasible:
        status = "passed"
    elif cap_pass is False or mem_pass is False:
        status = "failed"
    else:
        status = "pending"
    return {
        "name": candidate["name"],
        "role": candidate.get("role", "candidate"),
        "model_id": candidate.get("model_id", ""),
        "quant_type": candidate.get("quant_type", ""),
        "gguf": display_path(gguf) if gguf else "",
        "artifact_exists": artifact_exists,
        "artifact_bytes": artifact_bytes,
        "artifact_mb_decimal": artifact_bytes / 1_000_000,
        "capability": capability,
        "memory": memory,
        "joint_feasible": feasible,
        "status": status,
        "prune_reason": candidate.get("prune_reason", ""),
        "notes": candidate.get("notes", ""),
    }


def recommended_action(results: list[dict[str, Any]]) -> str:
    if any(item["joint_feasible"] for item in results):
        return "freeze_the_highest_margin_feasible_candidate"
    active = [item for item in results if item.get("status") != "pruned"]
    if active and all(item.get("status") == "failed" for item in active):
        return "close_g0_and_start_capability_recovery_on_a_memory_safe_base_or_structured_tool_head"
    if any(not item["artifact_exists"] for item in active):
        return "prepare_quantized_artifacts_then_run_peak_memory_and_matched_capability_smoke"
    if any(item["memory"]["passed"] is None for item in active):
        return "run_peak_total_memory_gate_for_existing_artifacts"
    if any(item["capability"]["passed"] is None for item in active):
        return "run_matched_capability_smoke_for_memory_feasible_artifacts"
    if any(item["memory"]["passed"] is False for item in active):
        return "reject_memory_failures_and_test_smaller_or_lower_bit_candidates"
    if any(item["capability"]["passed"] is False for item in active):
        return "keep_memory_feasible_candidates_only_and_change_model_capacity_or_training_route"
    return "introduce_a_2b_class_candidate_or_change_the_edge_model_architecture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, measure and summarize the G0 capability-memory candidate race.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute-memory", action="store_true")
    parser.add_argument("--execute-capability-smoke", action="store_true")
    parser.add_argument("--candidate", action="append", default=[], help="Only run named candidates; repeatable.")
    parser.add_argument("--require-feasible", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)
    config = load_json(config_path)
    common = dict(config.get("common", {}))
    selected = set(args.candidate)
    candidates = [item for item in config.get("candidates", []) if not selected or item.get("name") in selected]
    if not candidates:
        print("No G0 candidates selected.", file=sys.stderr)
        return 2

    commands: list[dict[str, Any]] = []
    if args.prepare:
        commands.extend(prepare_importance_matrix(common))
    for index, candidate in enumerate(candidates):
        name = str(candidate.get("name", ""))
        if not name:
            print("Every candidate must have a name.", file=sys.stderr)
            return 2
        if args.prepare:
            print(f"[G0 prepare] {name}", flush=True)
            commands.append({"stage": "prepare", "candidate": name, **prepare_candidate(candidate, common)})
        gguf = resolve_path(str(candidate.get("gguf", "")))
        if args.execute_memory and gguf.is_file():
            print(f"[G0 memory] {name}", flush=True)
            commands.append({"stage": "memory", "candidate": name, **measure_candidate(candidate, common, index)})
        if args.execute_capability_smoke and gguf.is_file():
            print(f"[G0 capability smoke] {name}", flush=True)
            commands.append({"stage": "capability", "candidate": name, **capability_smoke_candidate(candidate, common, index)})

    results = [candidate_result(candidate, common) for candidate in candidates]
    feasible = [item for item in results if item["joint_feasible"]]
    if feasible:
        status = "passed"
    elif all(item["status"] in {"failed", "pruned"} for item in results):
        status = "failed"
    else:
        status = "pending"
    report = {
        "gate": "G0-CAPMEM",
        "check_version": "1.1",
        "created_by": "scripts/run_g0_capmem.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config": display_path(config_path),
        "config_hash": sha256_file(config_path),
        "thresholds": {
            "min_capability_ratio": float(common.get("min_capability_ratio", 0.8)),
            "max_peak_total_memory_mb_decimal": float(common.get("max_memory_mb", 1500)),
        },
        "candidate_count": len(results),
        "feasible_candidate_count": len(feasible),
        "feasible_candidates": [item["name"] for item in feasible],
        "recommended_action": recommended_action(results),
        "candidates": results,
        "commands": commands,
    }
    report["report_hash"] = sha256_text(json.dumps({key: value for key, value in report.items() if key != "report_hash"}, sort_keys=True))
    write_json(output_path, report)
    print(f"Wrote {display_path(output_path)}")
    print(f"status={status} feasible={len(feasible)}/{len(results)}")
    if args.require_feasible and not feasible:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
