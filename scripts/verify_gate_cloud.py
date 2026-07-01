from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "pretrained" / "Qwen--Qwen2.5-14B-Instruct-AWQ"
DEFAULT_MODEL_AUDIT = ROOT / "reports" / "audit" / "model_downloads.json"
DEFAULT_OUTPUT = ROOT / "reports" / "audit" / "gate_cloud_smoke.json"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_REPO_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_PROMPT = (
    "Return exactly one compact JSON object with keys event_type, risk_attr, action. "
    "Use event_type=cloud_smoke, risk_attr=low, action=pass."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def model_files(model_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(model_dir).parts
    ]


def local_model_stats(model_dir: Path) -> dict[str, Any]:
    files = model_files(model_dir)
    return {
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def load_audit_entry(audit_path: Path, repo_id: str, model_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not audit_path.is_file():
        return None, [f"Missing model audit file: {display_path(audit_path)}"]

    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"Cannot read model audit JSON: {display_path(audit_path)} ({exc})"]

    if not isinstance(data, list):
        return None, [f"Model audit must be a list: {display_path(audit_path)}"]

    model_dir_rel = display_path(model_dir)
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("repo_id") == repo_id or item.get("local_dir") == model_dir_rel:
            return item, []

    return None, [f"Model audit has no entry for {repo_id} at {model_dir_rel}"]


def verify_model(
    model_dir: Path,
    audit_entry: dict[str, Any] | None,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    hashed_files: list[str] = []

    if not model_dir.is_dir():
        return {
            "passed": False,
            "model_dir": display_path(model_dir),
            "errors": [f"Missing model directory: {display_path(model_dir)}"],
        }

    stats = local_model_stats(model_dir)
    result: dict[str, Any] = {
        "passed": True,
        "model_dir": display_path(model_dir),
        "local_stats": stats,
        "audit_stats": {},
        "checked_files": [],
        "hashed_files": hashed_files,
        "errors": errors,
    }

    required_names = {"config.json", "generation_config.json", "tokenizer_config.json"}
    local_names = {path.name for path in model_files(model_dir)}
    missing_required = sorted(required_names - local_names)
    if missing_required:
        errors.append(f"Missing required model files: {', '.join(missing_required)}")

    if audit_entry is not None:
        result["audit_stats"] = {
            "repo_id": audit_entry.get("repo_id"),
            "role": audit_entry.get("role"),
            "file_count": audit_entry.get("file_count"),
            "bytes": audit_entry.get("bytes"),
        }
        if stats["file_count"] != audit_entry.get("file_count"):
            errors.append(
                f"Model file_count mismatch: expected {audit_entry.get('file_count')}, got {stats['file_count']}"
            )
        if stats["bytes"] != audit_entry.get("bytes"):
            errors.append(f"Model bytes mismatch: expected {audit_entry.get('bytes')}, got {stats['bytes']}")

        for item in list(audit_entry.get("key_files", [])) + list(audit_entry.get("weight_files", [])):
            relative = item.get("path")
            expected_size = item.get("bytes")
            expected_hash = item.get("sha256")
            if not relative:
                errors.append("Audit entry contains a file without path")
                continue
            path = model_dir / relative
            check: dict[str, Any] = {"path": relative}
            if not path.is_file():
                check["passed"] = False
                errors.append(f"Missing audited model file: {relative}")
                result["checked_files"].append(check)
                continue

            actual_size = path.stat().st_size
            check["bytes"] = actual_size
            if actual_size != expected_size:
                check["passed"] = False
                errors.append(f"Audited file size mismatch: {relative}")
            elif verify_file_hashes:
                actual_hash = sha256_file(path)
                hashed_files.append(relative)
                check["sha256"] = actual_hash
                if actual_hash != expected_hash:
                    check["passed"] = False
                    errors.append(f"Audited file hash mismatch: {relative}")
                else:
                    check["passed"] = True
            else:
                check["passed"] = True
            result["checked_files"].append(check)

        result["model_hash"] = stable_hash(
            {
                "repo_id": audit_entry.get("repo_id"),
                "local_dir": audit_entry.get("local_dir"),
                "file_count": audit_entry.get("file_count"),
                "bytes": audit_entry.get("bytes"),
                "key_files": audit_entry.get("key_files", []),
                "weight_files": audit_entry.get("weight_files", []),
            }
        )
    else:
        result["model_hash"] = stable_hash({"model_dir": display_path(model_dir), **stats})

    result["passed"] = not errors
    return result


def request_json(url: str, timeout_sec: float) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout_sec) as response:
            body = response.read()
            elapsed = time.perf_counter() - started
            text = body.decode("utf-8", errors="replace")
            details = {
                "url": url,
                "status_code": response.status,
                "elapsed_sec": elapsed,
                "body_preview": text[:500],
            }
            try:
                return json.loads(text), details
            except json.JSONDecodeError:
                return None, details
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, {"url": url, "status_code": exc.code, "error": body[:500]}
    except URLError as exc:
        return None, {"url": url, "error": f"{type(exc.reason).__name__}: {exc.reason}"}
    except OSError as exc:
        return None, {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def check_health(base_url: str, timeout_sec: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    started = time.perf_counter()
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout_sec) as response:
            body = response.read(500).decode("utf-8", errors="replace")
            return {
                "passed": response.status == 200,
                "url": url,
                "status_code": response.status,
                "elapsed_sec": time.perf_counter() - started,
                "body_preview": body,
            }
    except HTTPError as exc:
        return {"passed": False, "url": url, "status_code": exc.code, "error": exc.reason}
    except URLError as exc:
        return {"passed": False, "url": url, "error": f"{type(exc.reason).__name__}: {exc.reason}"}
    except OSError as exc:
        return {"passed": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def list_served_models(base_url: str, timeout_sec: float) -> dict[str, Any]:
    data, details = request_json(f"{base_url.rstrip('/')}/v1/models", timeout_sec)
    model_ids: list[str] = []
    if isinstance(data, dict):
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
    return {"model_ids": model_ids, **details}


def stream_chat_completion(
    base_url: str,
    api_model: str,
    prompt: str,
    timeout_sec: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_event_sec: float | None = None
    first_token_sec: float | None = None
    pieces: list[str] = []
    event_count = 0

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event_count += 1
                if first_event_sec is None:
                    first_event_sec = time.perf_counter() - started
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                text = ""
                for choice in chunk.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        text += str(delta.get("content") or "")
                    text += str(choice.get("text") or "")
                if text:
                    pieces.append(text)
                    if first_token_sec is None:
                        first_token_sec = time.perf_counter() - started

        completed_sec = time.perf_counter() - started
        response_text = "".join(pieces)
        return {
            "passed": first_token_sec is not None,
            "api_model": api_model,
            "event_count": event_count,
            "first_event_latency_sec": first_event_sec,
            "first_token_latency_sec": first_token_sec,
            "completed_sec": completed_sec,
            "response_preview": response_text[:500],
            "response_chars": len(response_text),
        }
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "passed": False,
            "api_model": api_model,
            "status_code": exc.code,
            "error": error_body[:1000],
        }
    except URLError as exc:
        return {"passed": False, "api_model": api_model, "error": f"{type(exc.reason).__name__}: {exc.reason}"}
    except OSError as exc:
        return {"passed": False, "api_model": api_model, "error": f"{type(exc).__name__}: {exc}"}


def start_command(model_dir: Path, port: int) -> str:
    return (
        f"vllm serve {display_path(model_dir)} --quantization awq --max-model-len 4096 "
        f"--gpu-memory-utilization 0.85 --tensor-parallel-size 1 --port {port}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify G-CLOUD 14B-AWQ vLLM smoke gate.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM/OpenAI-compatible base URL.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Local 14B-AWQ model directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Expected Hugging Face repo id.")
    parser.add_argument("--model-audit", default=str(DEFAULT_MODEL_AUDIT), help="Model download audit JSON.")
    parser.add_argument("--api-model", default="", help="Model id served by vLLM. Defaults to /v1/models first id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Smoke prompt used for first-token latency.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens for smoke response.")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--ttft-threshold-sec", type=float, default=2.0, help="G-CLOUD TTFT threshold.")
    parser.add_argument("--verify-file-hashes", action="store_true", help="Recompute audited model file SHA256 hashes.")
    parser.add_argument(
        "--offline-model-check",
        action="store_true",
        help="Only verify local model files and audit metadata; skip live vLLM checks.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Gate audit JSON output path.")
    parser.add_argument("--no-output", action="store_true", help="Do not write an audit report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    audit_path = Path(args.model_audit)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    audit_entry, audit_errors = load_audit_entry(audit_path, args.repo_id, model_dir)
    model_check = verify_model(model_dir, audit_entry, args.verify_file_hashes)
    model_check["audit_errors"] = audit_errors

    report: dict[str, Any] = {
        "gate": "G-CLOUD",
        "check_version": "1.0",
        "created_by": "scripts/verify_gate_cloud.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "repo_id": args.repo_id,
        "model_dir": display_path(model_dir),
        "model_hash": model_check.get("model_hash", ""),
        "prompt_hash": sha256_text(args.prompt),
        "prompt": args.prompt,
        "ttft_threshold_sec": args.ttft_threshold_sec,
        "offline_model_check": bool(args.offline_model_check),
        "model_check": model_check,
        "service_start_command": start_command(model_dir, 8000),
        "live_gate_pass": False,
        "status": "failed",
        "errors": [],
    }

    if audit_errors:
        report["errors"].extend(audit_errors)
    if not model_check["passed"]:
        report["errors"].extend(model_check["errors"])

    if args.offline_model_check:
        report["status"] = "offline_model_check_passed" if not report["errors"] else "failed"
        report["report_hash"] = stable_hash({k: v for k, v in report.items() if k != "report_hash"})
        if not args.no_output:
            write_json(output_path, report)
            print(f"Wrote {display_path(output_path)}")
        if report["status"] == "offline_model_check_passed":
            print("G-CLOUD offline model check passed.")
            print(f"model_hash={report['model_hash']}")
            print(f"prompt_hash={report['prompt_hash']}")
            print(f"Start command: {report['service_start_command']}")
            return 0
        print("G-CLOUD offline model check failed.")
        for error in report["errors"]:
            print(f"[FAIL] {error}")
        return 1

    health = check_health(args.base_url, args.timeout_sec)
    report["health"] = health
    if not health["passed"]:
        report["errors"].append(f"Health check failed: {health.get('error') or health.get('status_code')}")
    else:
        print(f"[OK] /health {health['status_code']} in {health['elapsed_sec']:.3f}s")

    served_models = list_served_models(args.base_url, args.timeout_sec) if health["passed"] else {"model_ids": []}
    report["served_models"] = served_models
    api_model = args.api_model or (served_models.get("model_ids") or [display_path(model_dir)])[0]
    report["api_model"] = api_model

    if health["passed"]:
        smoke = stream_chat_completion(args.base_url, api_model, args.prompt, args.timeout_sec, args.max_tokens)
        report["smoke"] = smoke
        ttft = smoke.get("first_token_latency_sec")
        ttft_passed = isinstance(ttft, (int, float)) and ttft < args.ttft_threshold_sec
        if not smoke["passed"]:
            report["errors"].append(f"Smoke prompt failed: {smoke.get('error', 'no first token')}")
        elif not ttft_passed:
            report["errors"].append(f"First token latency {ttft:.3f}s exceeds {args.ttft_threshold_sec:.3f}s")
        else:
            print(f"[OK] smoke first_token_latency={ttft:.3f}s")
    else:
        report["smoke"] = {"passed": False, "skipped": True, "reason": "health check failed"}

    report["live_gate_pass"] = not report["errors"]
    report["status"] = "passed" if report["live_gate_pass"] else "failed"
    report["report_hash"] = stable_hash({k: v for k, v in report.items() if k != "report_hash"})

    if not args.no_output:
        write_json(output_path, report)
        print(f"Wrote {display_path(output_path)}")

    if report["live_gate_pass"]:
        print("G-CLOUD live gate passed.")
        print(f"model_hash={report['model_hash']}")
        print(f"prompt_hash={report['prompt_hash']}")
        return 0

    print("G-CLOUD live gate failed.")
    for error in report["errors"]:
        print(f"[FAIL] {error}")
    print(f"Start command: {report['service_start_command']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
