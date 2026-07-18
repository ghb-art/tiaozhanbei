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
DEFAULT_MERGED_HF = ROOT / "models" / "checkpoints" / "db4ai-edge-3b-kd-merged"
DEFAULT_LLAMA_CPP = ROOT / "external" / "llama.cpp"
DEFAULT_F16 = ROOT / "models" / "quantized" / "db4ai-edge-3b-kd-f16.gguf"
DEFAULT_QUANTIZED = ROOT / "models" / "quantized" / "db4ai-edge-3b-kd-q4_k_m.gguf"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_g3_gguf_prepare.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
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


def find_one(root: Path, name: str) -> Path:
    candidates = sorted(path for path in root.rglob(name) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"Cannot find {name} under {display_path(root)}")
    executable = [path for path in candidates if path.stat().st_mode & 0o111]
    return executable[0] if executable else candidates[0]


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": command,
        "cwd": display_path(cwd or ROOT),
        "started_ts": started,
        "finished_ts": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-40:]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an HF edge model to GGUF and apply a selected quantization type.")
    parser.add_argument("--merged-hf-dir", "--merged_hf_dir", default=str(DEFAULT_MERGED_HF))
    parser.add_argument("--llama-cpp-dir", "--llama_cpp_dir", default=str(DEFAULT_LLAMA_CPP))
    parser.add_argument("--f16-gguf", "--f16_gguf", default=str(DEFAULT_F16))
    parser.add_argument(
        "--quantized-gguf",
        "--quantized_gguf",
        "--q4-gguf",
        "--q4_gguf",
        dest="quantized_gguf",
        default=str(DEFAULT_QUANTIZED),
    )
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--quant-type", "--quant_type", default="Q4_K_M")
    parser.add_argument("--imatrix", default="", help="Optional llama.cpp importance matrix for low-bit quantization.")
    parser.add_argument("--skip-f16-if-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-quantized-if-exists",
        "--skip-q4-if-exists",
        dest="skip_quantized_if_exists",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-quantized-bytes",
        "--max_quantized_bytes",
        type=int,
        default=0,
        help="Optional artifact-size pre-gate; 0 disables it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged_hf_dir = resolve_path(args.merged_hf_dir)
    llama_cpp_dir = resolve_path(args.llama_cpp_dir)
    f16_path = resolve_path(args.f16_gguf)
    quantized_path = resolve_path(args.quantized_gguf)
    audit_path = resolve_path(args.audit)
    imatrix_path = resolve_path(args.imatrix) if args.imatrix else None
    errors: list[str] = []
    commands: list[dict[str, Any]] = []

    if not merged_hf_dir.is_dir():
        errors.append(f"Missing merged HF model dir: {display_path(merged_hf_dir)}")
    if not llama_cpp_dir.is_dir():
        errors.append(f"Missing llama.cpp dir: {display_path(llama_cpp_dir)}")
    if errors:
        report = {
            "gate": "G3-gguf-prepare",
            "check_version": "1.0",
            "created_by": "scripts/prepare_edge_gguf.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "errors": errors,
        }
        write_json(audit_path, report)
        print("\n".join(errors), file=sys.stderr)
        return 1

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        print(f"Missing convert script: {display_path(convert_script)}", file=sys.stderr)
        return 2
    quantize_bin = find_one(llama_cpp_dir / "build", "llama-quantize")
    f16_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_path.parent.mkdir(parents=True, exist_ok=True)

    if not (args.skip_f16_if_exists and f16_path.is_file()):
        command = [
            sys.executable,
            str(convert_script),
            str(merged_hf_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            args.outtype,
        ]
        result = run_command(command)
        commands.append(result)
        if result["returncode"] != 0:
            errors.append("convert_hf_to_gguf.py failed")
    if not errors and not (args.skip_quantized_if_exists and quantized_path.is_file()):
        command = [str(quantize_bin)]
        if imatrix_path is not None:
            if not imatrix_path.is_file():
                errors.append(f"Missing importance matrix: {display_path(imatrix_path)}")
            else:
                command.extend(["--imatrix", str(imatrix_path)])
        command.extend([str(f16_path), str(quantized_path), args.quant_type])
        if errors:
            result = {"command": command, "returncode": 2, "stdout_tail": "", "stderr_tail": errors[-1]}
        else:
            result = run_command(command)
        commands.append(result)
        if result["returncode"] != 0:
            errors.append("llama-quantize failed")

    if (
        not errors
        and quantized_path.is_file()
        and args.max_quantized_bytes > 0
        and quantized_path.stat().st_size > args.max_quantized_bytes
    ):
        errors.append(
            f"quantized artifact exceeds size pre-gate: {quantized_path.stat().st_size} > {args.max_quantized_bytes}"
        )

    status = "passed" if not errors and quantized_path.is_file() else "failed"
    audit = {
        "gate": "G3-gguf-prepare",
        "check_version": "2.0",
        "created_by": "scripts/prepare_edge_gguf.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "merged_hf_dir": display_path(merged_hf_dir),
        "merged_hf_hash": sha256_dir(merged_hf_dir),
        "llama_cpp_dir": display_path(llama_cpp_dir),
        "llama_cpp_head": run_command(["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"]).get("stdout_tail", "").strip(),
        "convert_script": display_path(convert_script),
        "quantize_bin": display_path(quantize_bin),
        "f16_gguf": display_path(f16_path),
        "f16_gguf_hash": sha256_file(f16_path) if f16_path.is_file() else "",
        "f16_gguf_bytes": f16_path.stat().st_size if f16_path.is_file() else 0,
        "quantized_gguf": display_path(quantized_path),
        "quantized_gguf_hash": sha256_file(quantized_path) if quantized_path.is_file() else "",
        "quantized_gguf_bytes": quantized_path.stat().st_size if quantized_path.is_file() else 0,
        "max_quantized_bytes": args.max_quantized_bytes,
        "artifact_size_pre_gate_passed": bool(
            quantized_path.is_file()
            and (args.max_quantized_bytes <= 0 or quantized_path.stat().st_size <= args.max_quantized_bytes)
        ),
        "q4_gguf": display_path(quantized_path),
        "q4_gguf_hash": sha256_file(quantized_path) if quantized_path.is_file() else "",
        "q4_gguf_bytes": quantized_path.stat().st_size if quantized_path.is_file() else 0,
        "outtype": args.outtype,
        "quant_type": args.quant_type,
        "imatrix": display_path(imatrix_path) if imatrix_path else "",
        "imatrix_hash": sha256_file(imatrix_path) if imatrix_path and imatrix_path.is_file() else "",
        "commands": commands,
        "errors": errors,
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)
    print(f"Wrote {display_path(audit_path)}")
    print(f"status={status}")
    if quantized_path.is_file():
        print(f"quantized_gguf={display_path(quantized_path)}")
        print(f"quantized_gguf_hash={audit['quantized_gguf_hash']}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
