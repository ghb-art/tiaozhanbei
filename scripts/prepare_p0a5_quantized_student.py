#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class QuantizationError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QuantizationError(f"Missing audit: {display_path(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_passed(path: Path) -> dict[str, Any]:
    report = read_json(path)
    if report.get("status") != "passed":
        raise QuantizationError(
            f"Audit is not passed: {display_path(path)} "
            f"status={report.get('status')}"
        )
    return report


def write_audit(path: Path, report: dict[str, Any]) -> None:
    report["updated_ts"] = datetime.now(timezone.utc).isoformat()
    report["report_hash"] = sha256_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "report_hash"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_binary(llama_cpp: Path, name: str) -> Path:
    preferred = llama_cpp / "build" / "bin" / name
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return preferred
    candidates = sorted(
        path
        for path in (llama_cpp / "build").rglob(name)
        if path.is_file() and os.access(path, os.X_OK)
    )
    if not candidates:
        raise QuantizationError(f"Missing executable: {name}")
    return candidates[0]


def run_streaming(
    command: list[str],
    report: dict[str, Any],
    stage: str,
    env: dict[str, str] | None = None,
) -> None:
    record = {
        "stage": stage,
        "command": command,
        "started_ts": datetime.now(timezone.utc).isoformat(),
    }
    report.setdefault("commands", []).append(record)
    print(f"[{stage}] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
    )
    record["finished_ts"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise QuantizationError(
            f"{stage} failed with return code {completed.returncode}"
        )


def reusable_artifact(
    old_report: dict[str, Any],
    stage: str,
    path: Path,
) -> bool:
    old_stage = old_report.get("stages", {}).get(stage, {})
    if old_stage.get("status") != "passed" or not path.is_file():
        return False
    expected_hash = str(old_stage.get("sha256", ""))
    if not expected_hash:
        return False
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise QuantizationError(
            f"Existing {stage} artifact hash changed: {display_path(path)}"
        )
    print(f"[resume] Reusing {stage}: {display_path(path)}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert, calibrate and Q4_K_M-quantize one P0-A5 Student."
    )
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--config", default="configs/p0a5_capability.json")
    parser.add_argument("--merged-dir", default="")
    parser.add_argument("--corpus", default="")
    parser.add_argument("--merge-audit", default="")
    parser.add_argument("--corpus-audit", default="")
    parser.add_argument("--train-audit", default="")
    parser.add_argument("--audit", default="")
    parser.add_argument("--f16-output", default="")
    parser.add_argument("--imatrix-output", default="")
    parser.add_argument("--q4-output", default="")
    parser.add_argument("--gate-name", default="P0-A5-STUDENT-QUANTIZATION")
    parser.add_argument("--llama-cpp", default="external/llama.cpp")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--chunks", type=int, default=170)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-q4-bytes", type=int, default=1_200_000_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = args.candidate
    config_path = resolve_path(args.config)
    merged_dir = resolve_path(
        args.merged_dir
        or f"models/checkpoints/p0a5/student-candidate-{candidate}-merged"
    )
    corpus_path = resolve_path(
        args.corpus or "data/capability_v2/imatrix_calibration.txt"
    )
    merge_audit_path = resolve_path(
        args.merge_audit
        or f"reports/audit/gate_p0a5_merge_student_candidate_{candidate}.json"
    )
    corpus_audit_path = resolve_path(
        args.corpus_audit or "reports/audit/gate_p0a5_imatrix_calibration.json"
    )
    train_audit_path = resolve_path(
        args.train_audit
        or f"reports/audit/gate_p0a5_train_student_candidate_{candidate}.json"
    )
    audit_path = resolve_path(
        args.audit
        or f"reports/audit/gate_p0a5_quantize_student_candidate_{candidate}.json"
    )
    f16_path = resolve_path(
        args.f16_output
        or f"models/quantized/p0a5-student-candidate-{candidate}-f16.gguf"
    )
    imatrix_path = resolve_path(
        args.imatrix_output
        or f"models/quantized/p0a5-student-candidate-{candidate}-imatrix.gguf"
    )
    q4_path = resolve_path(
        args.q4_output
        or f"models/quantized/p0a5-student-candidate-{candidate}-q4_k_m.gguf"
    )
    llama_cpp = resolve_path(args.llama_cpp)
    partial_paths = {
        "f16": f16_path.with_suffix(f16_path.suffix + ".partial"),
        "imatrix": imatrix_path.with_suffix(imatrix_path.suffix + ".partial"),
        "q4_k_m": q4_path.with_suffix(q4_path.suffix + ".partial"),
    }
    try:
        if args.chunks <= 0:
            raise QuantizationError("--chunks must be positive")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["quantization"]["weight_type"] != "Q4_K_M":
            raise QuantizationError("P0-A5 quantization type is not Q4_K_M")
        if config["quantization"]["imatrix_source"] != "training_only":
            raise QuantizationError("P0-A5 imatrix source is not training_only")
        train_audit = require_passed(train_audit_path)
        merge_audit = require_passed(merge_audit_path)
        corpus_audit = require_passed(corpus_audit_path)
        if display_path(merged_dir) != str(merge_audit.get("output", "")):
            raise QuantizationError("Merged Student path differs from merge audit")
        if not merged_dir.is_dir():
            raise QuantizationError(
                f"Missing merged Student: {display_path(merged_dir)}"
            )
        if not corpus_path.is_file():
            raise QuantizationError(
                f"Missing imatrix corpus: {display_path(corpus_path)}"
            )
        if sha256_file(corpus_path) != corpus_audit.get("output_hash"):
            raise QuantizationError("Imatrix calibration corpus hash changed")
        if int(corpus_audit.get("formal_test_reference_count", -1)) != 0:
            raise QuantizationError("Imatrix corpus contains formal-test references")
        convert_script = llama_cpp / "convert_hf_to_gguf.py"
        if not convert_script.is_file():
            raise QuantizationError(
                f"Missing converter: {display_path(convert_script)}"
            )
        imatrix_bin = find_binary(llama_cpp, "llama-imatrix")
        quantize_bin = find_binary(llama_cpp, "llama-quantize")
        inputs = {
            "config": display_path(config_path),
            "config_hash": sha256_file(config_path),
            "train_audit": display_path(train_audit_path),
            "train_audit_hash": sha256_file(train_audit_path),
            "merge_audit": display_path(merge_audit_path),
            "merge_audit_hash": sha256_file(merge_audit_path),
            "merged_hf": display_path(merged_dir),
            "merged_hf_hash": merge_audit["output_hash"],
            "corpus_audit": display_path(corpus_audit_path),
            "corpus_audit_hash": sha256_file(corpus_audit_path),
            "corpus": display_path(corpus_path),
            "corpus_hash": corpus_audit["output_hash"],
        }
        settings = {
            "weight_type": "Q4_K_M",
            "kv_cache_type_for_runtime": "q8_0",
            "imatrix_gpu": str(args.gpu),
            "imatrix_chunks": args.chunks,
            "imatrix_ctx_size": args.ctx_size,
            "imatrix_batch_size": args.batch_size,
            "imatrix_ubatch_size": args.ubatch_size,
            "imatrix_threads": args.threads,
            "max_q4_bytes": args.max_q4_bytes,
        }
        old_report: dict[str, Any] = {}
        if audit_path.is_file():
            old_report = read_json(audit_path)
            if old_report.get("inputs") != inputs:
                raise QuantizationError(
                    "Existing quantization audit belongs to different inputs"
                )
            if old_report.get("settings") != settings:
                raise QuantizationError(
                    "Existing quantization audit uses different settings"
                )
        report: dict[str, Any] = {
            "gate": args.gate_name,
            "check_version": "1.0",
            "created_by": "scripts/prepare_p0a5_quantized_student.py",
            "created_ts": old_report.get(
                "created_ts", datetime.now(timezone.utc).isoformat()
            ),
            "status": "dry_run_passed" if args.dry_run else "running",
            "candidate": candidate,
            "inputs": inputs,
            "settings": settings,
            "tools": {
                "llama_cpp_head": subprocess.run(
                    ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip(),
                "convert_script": display_path(convert_script),
                "convert_script_hash": sha256_file(convert_script),
                "imatrix_bin": display_path(imatrix_bin),
                "imatrix_bin_hash": sha256_file(imatrix_bin),
                "quantize_bin": display_path(quantize_bin),
                "quantize_bin_hash": sha256_file(quantize_bin),
            },
            "stages": dict(old_report.get("stages", {})),
            "commands": list(old_report.get("commands", [])),
            "errors": [],
        }
        commands = {
            "f16": [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(partial_paths["f16"]),
                "--outtype",
                "f16",
            ],
            "imatrix": [
                str(imatrix_bin),
                "--model",
                str(f16_path),
                "--file",
                str(corpus_path),
                "--output",
                str(partial_paths["imatrix"]),
                "--output-format",
                "gguf",
                "--chunks",
                str(args.chunks),
                "--ctx-size",
                str(args.ctx_size),
                "--batch-size",
                str(args.batch_size),
                "--ubatch-size",
                str(args.ubatch_size),
                "--threads",
                str(args.threads),
                "--n-gpu-layers",
                "all",
                "--no-ppl",
            ],
            "q4_k_m": [
                str(quantize_bin),
                "--imatrix",
                str(imatrix_path),
                str(f16_path),
                str(partial_paths["q4_k_m"]),
                "Q4_K_M",
            ],
        }
        if args.dry_run:
            for stage, command in commands.items():
                print(f"[dry-run:{stage}] {' '.join(command)}")
            return 0
        f16_path.parent.mkdir(parents=True, exist_ok=True)
        for partial in partial_paths.values():
            if partial.exists():
                partial.unlink()
        stages = (
            ("f16", f16_path),
            ("imatrix", imatrix_path),
            ("q4_k_m", q4_path),
        )
        for stage, output_path in stages:
            if reusable_artifact(old_report, stage, output_path):
                report["stages"][stage] = old_report["stages"][stage]
                continue
            env = os.environ.copy()
            if stage == "imatrix":
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            run_streaming(commands[stage], report, stage, env=env)
            partial = partial_paths[stage]
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise QuantizationError(
                    f"{stage} did not produce an artifact: {display_path(partial)}"
                )
            os.replace(partial, output_path)
            stage_report = {
                "status": "passed",
                "path": display_path(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
            if stage == "q4_k_m":
                stage_report["within_size_limit"] = (
                    output_path.stat().st_size <= args.max_q4_bytes
                )
                if not stage_report["within_size_limit"]:
                    raise QuantizationError(
                        f"Q4_K_M artifact too large: {output_path.stat().st_size} "
                        f"> {args.max_q4_bytes}"
                    )
            report["stages"][stage] = stage_report
            write_audit(audit_path, report)
        report["status"] = "passed"
        report["errors"] = []
        write_audit(audit_path, report)
        print(f"Wrote {display_path(audit_path)}")
        print(
            f"status=passed q4_k_m={display_path(q4_path)} "
            f"bytes={q4_path.stat().st_size}"
        )
        return 0
    except (
        QuantizationError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        message = str(exc)
        print(f"P0-A5 quantization failed: {message}", file=sys.stderr)
        if not args.dry_run:
            try:
                report
            except UnboundLocalError:
                report = {
                    "gate": args.gate_name,
                    "check_version": "1.0",
                    "created_by": "scripts/prepare_p0a5_quantized_student.py",
                    "created_ts": datetime.now(timezone.utc).isoformat(),
                    "candidate": candidate,
                    "stages": {},
                    "commands": [],
                }
            report["status"] = "failed"
            report.setdefault("errors", []).append(message)
            write_audit(audit_path, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
