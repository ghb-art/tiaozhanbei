from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from generate_teacher_traces import extract_json_object, normalize_decision, request_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_TRACE = ROOT / "data" / "distill" / "teacher_decision_trace.jsonl"
DEFAULT_OUTPUT_PROBE = ROOT / "data" / "distill" / "student_probe_trace.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_student_probe.json"
DEFAULT_STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROBE_SYSTEM_PROMPT = (
    "You are the DB4AI-EdgeServe edge student. Return only one JSON object. "
    "No markdown, no prose outside JSON."
)
PROBE_PROMPT_TEMPLATE = """Create a compact structured edge decision.

Return this exact JSON schema:
{
  "object_state": "short observable state",
  "event_type": "math_reasoning|knowledge_choice|industrial_normal|surface_defect|traffic_camera",
  "risk_attr": "low|medium|high",
  "action": "pass|inspect|alert|upload",
  "confidence": 0.0,
  "review_intent": "none|verify_reasoning|inspect_quality|sync_tracking",
  "short_rationale": "one short sentence",
  "evidence_items": ["1-3 short evidence strings"]
}

Use only the sample context below.
Sample context:
{sample_context}
"""


class ProbeError(RuntimeError):
    pass


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


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_comma_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def select_rows(
    rows: list[dict[str, Any]],
    dataset_filter: set[str] | None,
    sample_limit: int | None,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if dataset_filter is None or str(row.get("dataset_key", "")) in dataset_filter
    ]
    if not filtered:
        raise ProbeError("No teacher trace rows selected")
    if sample_limit is None:
        return filtered

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        by_dataset.setdefault(str(row.get("dataset_key", "")), []).append(row)

    selected: list[dict[str, Any]] = []
    positions = {key: 0 for key in by_dataset}
    keys = sorted(by_dataset)
    while len(selected) < sample_limit:
        progressed = False
        for key in keys:
            pos = positions[key]
            if pos < len(by_dataset[key]):
                selected.append(by_dataset[key][pos])
                positions[key] += 1
                progressed = True
                if len(selected) >= sample_limit:
                    break
        if not progressed:
            break
    return selected


def get_served_model(base_url: str, timeout_sec: float, fallback: str) -> str:
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/v1/models", method="GET"), timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    except Exception:
        pass
    return fallback


def health_status(base_url: str, timeout_sec: float) -> int | None:
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/health", method="GET"), timeout=timeout_sec) as response:
            return int(response.status)
    except Exception:
        return None


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    sample_context = json.dumps(row.get("sample_context", {}), ensure_ascii=False, sort_keys=True, indent=2)
    prompt = PROBE_PROMPT_TEMPLATE.replace("{sample_context}", sample_context)
    return prompt, sha256_text(PROBE_SYSTEM_PROMPT + "\n" + prompt)


def call_student_endpoint(
    student_url: str,
    student_model: str,
    row: dict[str, Any],
    timeout_sec: float,
    max_tokens: int,
) -> dict[str, Any]:
    prompt, prompt_hash = build_prompt(row)
    payload = {
        "model": student_model,
        "messages": [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = request_json(f"{student_url.rstrip('/')}/v1/chat/completions", payload, timeout_sec)
    latency_ms = (time.perf_counter() - started) * 1000
    choices = response.get("choices", [])
    if not choices:
        raise ProbeError("Student response has no choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    response_text = str(message.get("content", ""))
    parsed = extract_json_object(response_text)
    decision, evidence_items, parse_errors = normalize_decision(parsed)
    return {
        "prompt_hash": prompt_hash,
        "request_hash": sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        "response_text": response_text,
        "parsed_response": parsed,
        "decision_tuple": decision,
        "evidence_items": evidence_items,
        "parse_errors": parse_errors,
        "latency_ms": latency_ms,
    }


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ProbeError(f"Unsupported dtype: {dtype_name}")


def load_local_student(
    model_dir: Path,
    adapter_path: Path | None,
    device: str,
    dtype_name: str,
    quantize_adapter: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_utils import load_lora_adapter

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=resolve_torch_dtype(dtype_name),
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapter_config: dict[str, Any] = {}
    if adapter_path is not None:
        adapter_config = load_lora_adapter(model, adapter_path, quantize_adapter=quantize_adapter)
    model.to(device)
    model.eval()
    return tokenizer, model, adapter_config


def call_local_student(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    device: str,
    max_tokens: int,
) -> dict[str, Any]:
    import torch

    prompt, prompt_hash = build_prompt(row)
    messages = [
        {"role": "system", "content": PROBE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True).to(device)
    input_length = int(inputs["input_ids"].shape[1])
    payload = {
        "backend": "local_transformers",
        "messages": messages,
        "max_new_tokens": max_tokens,
        "do_sample": False,
    }

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = (time.perf_counter() - started) * 1000
    response_ids = output_ids[0, input_length:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    parsed: dict[str, Any] = {}
    decision: dict[str, Any] = {}
    evidence_items: list[str] = []
    parse_errors: list[str] = []
    try:
        parsed = extract_json_object(response_text)
        decision, evidence_items, parse_errors = normalize_decision(parsed)
    except Exception as exc:
        parse_errors = [f"{type(exc).__name__}: {exc}"]

    return {
        "prompt_hash": prompt_hash,
        "request_hash": sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        "response_text": response_text,
        "parsed_response": parsed,
        "decision_tuple": decision,
        "evidence_items": evidence_items,
        "parse_errors": parse_errors,
        "latency_ms": latency_ms,
    }


def dry_run_student(row: dict[str, Any], dry_run_mismatch_mod: int) -> dict[str, Any]:
    _, prompt_hash = build_prompt(row)
    teacher_decision = dict(row.get("decision_tuple", {}))
    decision = dict(teacher_decision)
    evidence_items = [str(item) for item in row.get("evidence_items", [])[:3]] or ["teacher trace evidence"]
    sample_hash = int(sha256_text(str(row.get("sample_id", "")))[:8], 16)

    confidence = float(decision.get("confidence", 0.7) or 0.7)
    decision["confidence"] = max(0.05, min(1.0, confidence - 0.12))
    decision["short_rationale"] = "dry-run student probe replay"

    if dry_run_mismatch_mod > 0 and sample_hash % dry_run_mismatch_mod == 0:
        action = str(decision.get("action", "pass"))
        decision["action"] = "inspect" if action == "pass" else "pass"
        decision["confidence"] = min(decision["confidence"], 0.55)
        decision["review_intent"] = "inspect_quality"

    parsed = dict(decision)
    parsed["evidence_items"] = evidence_items
    return {
        "prompt_hash": prompt_hash,
        "request_hash": sha256_text("dry-run::" + str(row.get("sample_id", ""))),
        "response_text": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        "parsed_response": parsed,
        "decision_tuple": decision,
        "evidence_items": evidence_items,
        "parse_errors": [],
        "latency_ms": 0.0,
    }


def agreement_flags(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, bool]:
    return {
        "event_type_match": teacher.get("event_type") == student.get("event_type"),
        "risk_attr_match": teacher.get("risk_attr") == student.get("risk_attr"),
        "action_match": teacher.get("action") == student.get("action"),
        "review_intent_match": teacher.get("review_intent") == student.get("review_intent"),
    }


def repair_reasons(
    teacher: dict[str, Any],
    student: dict[str, Any],
    parse_errors: list[str],
    min_confidence: float,
    max_confidence_gap: float,
) -> list[str]:
    reasons: list[str] = []
    if parse_errors:
        reasons.append("parse_error")
    if teacher.get("event_type") != student.get("event_type"):
        reasons.append("event_type_mismatch")
    if teacher.get("risk_attr") != student.get("risk_attr"):
        reasons.append("risk_attr_mismatch")
    if teacher.get("action") != student.get("action"):
        reasons.append("action_mismatch")
    if teacher.get("review_intent") != student.get("review_intent"):
        reasons.append("review_intent_mismatch")

    student_confidence = float(student.get("confidence", 0.0) or 0.0)
    teacher_confidence = float(teacher.get("confidence", 0.0) or 0.0)
    if student_confidence < min_confidence:
        reasons.append("low_student_confidence")
    if abs(teacher_confidence - student_confidence) > max_confidence_gap:
        reasons.append("confidence_gap")
    return reasons


def build_probe_row(
    teacher_row: dict[str, Any],
    student_result: dict[str, Any],
    backend: str,
    student_model_id: str,
    adapter_path: str,
    created_ts: str,
    min_confidence: float,
    max_confidence_gap: float,
) -> dict[str, Any]:
    teacher_decision = dict(teacher_row.get("decision_tuple", {}))
    student_decision = dict(student_result["decision_tuple"])
    parse_errors = [str(item) for item in student_result.get("parse_errors", [])]
    reasons = repair_reasons(
        teacher_decision,
        student_decision,
        parse_errors,
        min_confidence,
        max_confidence_gap,
    )
    confidence_gap = float(teacher_decision.get("confidence", 0.0) or 0.0) - float(
        student_decision.get("confidence", 0.0) or 0.0
    )
    row = {
        "probe_version": "1.0",
        "created_by": "model_compression/run_student_probe.py",
        "created_ts": created_ts,
        "sample_id": teacher_row["sample_id"],
        "dataset_key": teacher_row["dataset_key"],
        "split": teacher_row["split"],
        "task_type": teacher_row["task_type"],
        "teacher_trace_row_hash": teacher_row.get("trace_row_hash", ""),
        "teacher_prompt_hash": teacher_row.get("prompt_hash", ""),
        "teacher_model_id": teacher_row.get("teacher_model_id", ""),
        "teacher_decision_tuple": teacher_decision,
        "student_model_id": student_model_id,
        "student_adapter_path": adapter_path,
        "probe_backend": backend,
        "student_prompt_hash": student_result["prompt_hash"],
        "student_request_hash": student_result["request_hash"],
        "student_response_text": student_result["response_text"],
        "student_response_json": student_result["parsed_response"],
        "student_decision_tuple": student_decision,
        "student_evidence_items": student_result["evidence_items"],
        "parse_ok": not parse_errors,
        "parse_errors": parse_errors,
        "agreement": agreement_flags(teacher_decision, student_decision),
        "confidence_gap": confidence_gap,
        "repair_candidate": bool(reasons),
        "repair_candidate_reasons": reasons,
        "latency_ms": student_result["latency_ms"],
    }
    row["probe_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Student-Base/CEDD student probe over teacher traces.")
    parser.add_argument("--teacher-trace", "--teacher_trace", default=str(DEFAULT_TEACHER_TRACE))
    parser.add_argument("--output-probe", "--output_probe", default=str(DEFAULT_OUTPUT_PROBE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dataset", action="append", default=[], help="Dataset key filter; repeat or comma-separate.")
    parser.add_argument("--sample-limit", "--sample_limit", type=int, default=None)
    parser.add_argument("--student-url", "--student_url", default="")
    parser.add_argument("--student-model-id", "--student_model_id", default=DEFAULT_STUDENT_MODEL_ID)
    parser.add_argument("--adapter-path", "--adapter_path", default="")
    parser.add_argument("--local-model-dir", "--local_model_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--quantize-adapter",
        "--quantize_adapter",
        action="store_true",
        help="Apply fake INT4 dequantization to LoRA adapter weights during local inference.",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-parse-rate", type=float, default=0.9)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--max-confidence-gap", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic replay probe; no model call.")
    parser.add_argument(
        "--dry-run-mismatch-mod",
        type=int,
        default=7,
        help="Every Nth deterministic dry-run sample gets an action mismatch. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit is not None and args.sample_limit <= 0:
        print("--sample-limit must be positive", file=sys.stderr)
        return 2
    backend_count = sum(bool(item) for item in (args.dry_run, args.student_url, args.local_model_dir))
    if backend_count != 1:
        print("Choose exactly one backend: --dry-run, --student-url, or --local-model-dir.", file=sys.stderr)
        return 2

    teacher_trace_path = resolve_path(args.teacher_trace)
    output_path = resolve_path(args.output_probe)
    audit_path = resolve_path(args.audit)
    local_model_dir = resolve_path(args.local_model_dir) if args.local_model_dir else None
    adapter_path = resolve_path(args.adapter_path) if args.adapter_path else None
    dataset_filter = set(parse_comma_values(args.dataset)) if args.dataset else None

    teacher_rows = load_jsonl(teacher_trace_path)
    selected_rows = select_rows(teacher_rows, dataset_filter, args.sample_limit)
    created_ts = datetime.now(timezone.utc).isoformat()

    backend = "dry_run_replay" if args.dry_run else "openai_compatible"
    served_model_id = args.student_model_id
    health = None
    local_tokenizer = None
    local_model = None
    local_adapter_config: dict[str, Any] = {}
    if local_model_dir is not None:
        backend = "local_transformers_adapter" if adapter_path else "local_transformers_base"
        local_tokenizer, local_model, local_adapter_config = load_local_student(
            local_model_dir,
            adapter_path,
            args.device,
            args.dtype,
            args.quantize_adapter,
        )
    elif not args.dry_run:
        health = health_status(args.student_url, args.timeout_sec)
        if health != 200:
            print(f"Student endpoint health check failed: {args.student_url} status={health}", file=sys.stderr)
            return 1
        served_model_id = get_served_model(args.student_url, args.timeout_sec, args.student_model_id)

    probe_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, teacher_row in enumerate(selected_rows, start=1):
        try:
            if args.dry_run:
                student_result = dry_run_student(teacher_row, args.dry_run_mismatch_mod)
            elif local_model is not None and local_tokenizer is not None:
                student_result = call_local_student(
                    local_tokenizer,
                    local_model,
                    teacher_row,
                    args.device,
                    args.max_tokens,
                )
            else:
                student_result = call_student_endpoint(
                    args.student_url,
                    served_model_id,
                    teacher_row,
                    args.timeout_sec,
                    args.max_tokens,
                )
            probe_row = build_probe_row(
                teacher_row,
                student_result,
                backend,
                served_model_id,
                args.adapter_path,
                created_ts,
                args.min_confidence,
                args.max_confidence_gap,
            )
            probe_rows.append(probe_row)
            print(
                f"[OK] {index}/{len(selected_rows)} {probe_row['sample_id']} "
                f"repair_candidate={probe_row['repair_candidate']} parse_ok={probe_row['parse_ok']}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"sample_id": str(teacher_row.get("sample_id", "")), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {index}/{len(selected_rows)} {teacher_row.get('sample_id')}: {exc}", flush=True)

    write_jsonl(output_path, probe_rows)
    parse_ok_count = sum(1 for row in probe_rows if row.get("parse_ok") is True)
    parse_rate = parse_ok_count / len(probe_rows) if probe_rows else 0.0
    candidate_count = sum(1 for row in probe_rows if row.get("repair_candidate") is True)
    dataset_counts = Counter(row["dataset_key"] for row in probe_rows)
    reason_counts = Counter(reason for row in probe_rows for reason in row.get("repair_candidate_reasons", []))
    action_match_count = sum(1 for row in probe_rows if row.get("agreement", {}).get("action_match") is True)
    status = "passed" if not errors and parse_rate >= args.min_parse_rate and probe_rows else "failed"
    audit = {
        "gate": "G-KD-TRACE-student-probe-smoke" if args.sample_limit else "G-KD-TRACE-student-probe",
        "check_version": "1.0",
        "created_by": "model_compression/run_student_probe.py",
        "created_ts": created_ts,
        "status": status,
        "teacher_trace_path": display_path(teacher_trace_path),
        "teacher_trace_hash": sha256_file(teacher_trace_path),
        "output_probe_path": display_path(output_path),
        "student_probe_trace_hash": sha256_file(output_path),
        "probe_backend": backend,
        "dry_run": bool(args.dry_run),
        "student_url": args.student_url,
        "student_endpoint_health_status": health,
        "local_model_dir": display_path(local_model_dir) if local_model_dir else "",
        "local_adapter_config": local_adapter_config,
        "local_dtype": args.dtype if local_model_dir else "",
        "local_device": args.device if local_model_dir else "",
        "student_model_id": args.student_model_id,
        "served_model_id": served_model_id,
        "adapter_path": args.adapter_path,
        "quantize_adapter": bool(args.quantize_adapter),
        "sample_limit": args.sample_limit,
        "selected_sample_count": len(selected_rows),
        "successful_probe_count": len(probe_rows),
        "failed_probe_count": len(errors),
        "parse_ok_count": parse_ok_count,
        "parse_success_rate": parse_rate,
        "min_parse_rate": args.min_parse_rate,
        "repair_candidate_count": candidate_count,
        "action_match_rate": action_match_count / len(probe_rows) if probe_rows else 0.0,
        "selected_sample_ids_hash": sha256_text("\n".join(str(row.get("sample_id", "")) for row in selected_rows) + "\n"),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "repair_reason_counts": dict(sorted(reason_counts.items())),
        "prompt_template_hash": sha256_text(PROBE_SYSTEM_PROMPT + "\n" + PROBE_PROMPT_TEMPLATE),
        "errors": errors,
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)

    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"student_probe_trace_hash={audit['student_probe_trace_hash']}")
    if status != "passed":
        print("Student probe smoke failed.", file=sys.stderr)
        return 1
    print("Student probe smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
