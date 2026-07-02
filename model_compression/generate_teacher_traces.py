from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "data" / "datasets"
SPLITS = ROOT / "data" / "splits"
DEFAULT_TEACHER_URL = "http://127.0.0.1:8000"
DEFAULT_TEACHER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_TEACHER_TRACE = ROOT / "data" / "distill" / "teacher_decision_trace.jsonl"
DEFAULT_DISTILL = ROOT / "data" / "distill" / "distill_dataset.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_trace_teacher.json"
ALLOWED_SPLITS = {"train", "validation"}

SYSTEM_PROMPT = (
    "You are the DB4AI-EdgeServe cloud teacher. Return only one JSON object. "
    "No markdown, no prose outside JSON."
)

PROMPT_TEMPLATE = """Create a compact structured teacher decision for edge distillation.

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

Rules:
- Use action=pass for correct normal/low-risk cases.
- Use action=inspect for uncertainty or medium risk.
- Use action=alert for high-risk defects or safety-critical events.
- Use action=upload when cloud/global sync is useful.
- confidence must be a number from 0 to 1.

Sample context:
{sample_context}
"""


class TraceError(RuntimeError):
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_trace_rows_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            rows_by_id[sample_id] = row
    return rows_by_id


def read_split_ids(split: str) -> dict[str, list[str]]:
    ids_by_dataset: dict[str, list[str]] = {}
    for path in sorted(SPLITS.glob(f"*_{split}.txt")):
        dataset_key = path.name[: -len(f"_{split}.txt")]
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ids:
            ids_by_dataset[dataset_key] = ids
    return ids_by_dataset


def load_frozen_splits() -> dict[str, Any]:
    path = SPLITS / "frozen_splits.json"
    if not path.is_file():
        raise TraceError("Missing data/splits/frozen_splits.json")
    return json.loads(path.read_text(encoding="utf-8"))


def split_hash_for(frozen: dict[str, Any], dataset_key: str, split: str) -> str:
    for dataset in frozen.get("datasets", []):
        if dataset.get("dataset_key") == dataset_key:
            return str(dataset.get("hashes", {}).get(split, ""))
    return ""


def empty_split_dataset_keys(frozen: dict[str, Any], split: str) -> list[str]:
    empty: list[str] = []
    for dataset in frozen.get("datasets", []):
        dataset_key = str(dataset.get("dataset_key", ""))
        count = dataset.get("counts", {}).get(split, 0)
        if dataset_key and count == 0:
            empty.append(dataset_key)
    return sorted(empty)


def select_sample_ids(
    ids_by_dataset: dict[str, list[str]],
    include_datasets: set[str] | None,
    sample_limit: int | None,
) -> list[str]:
    filtered = {
        key: list(ids)
        for key, ids in sorted(ids_by_dataset.items())
        if ids and (include_datasets is None or key in include_datasets)
    }
    if not filtered:
        raise TraceError("No sample ids selected from split")

    if sample_limit is None:
        return [sample_id for key in sorted(filtered) for sample_id in filtered[key]]

    selected: list[str] = []
    positions = {key: 0 for key in filtered}
    keys = sorted(filtered)
    while len(selected) < sample_limit:
        progressed = False
        for key in keys:
            pos = positions[key]
            if pos < len(filtered[key]):
                selected.append(filtered[key][pos])
                positions[key] += 1
                progressed = True
                if len(selected) >= sample_limit:
                    break
        if not progressed:
            break
    return selected


def apply_shard(sample_ids: list[str], num_shards: int, shard_index: int) -> list[str]:
    if num_shards < 1:
        raise TraceError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise TraceError("--shard-index must be in [0, num_shards)")
    if num_shards == 1:
        return sample_ids
    return [sample_id for index, sample_id in enumerate(sample_ids) if index % num_shards == shard_index]


def csv_rows_for_split(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[1:] if rows else []


def load_gsm8k(sample_id: str) -> dict[str, Any]:
    _, split, index_text = sample_id.split("/")
    path = DATASETS / "gsm8k" / "grade_school_math" / "data" / f"{split}.jsonl"
    index = int(index_text)
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                item = json.loads(line)
                answer = str(item.get("answer", ""))
                final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
                context = {
                    "dataset": "GSM8K",
                    "task_type": "math_reasoning",
                    "question": item.get("question", ""),
                    "reference_final_answer": final_answer,
                }
                return {
                    "sample_id": sample_id,
                    "dataset_key": "gsm8k",
                    "task_type": "math_reasoning",
                    "input_text": f"Math problem: {item.get('question', '')}",
                    "reference": {"answer": answer, "final_answer": final_answer},
                    "sample_context": context,
                }
    raise TraceError(f"GSM8K sample index not found: {sample_id}")


def mmlu_csv_path(split: str, subject: str) -> Path:
    base = DATASETS / "mmlu" / "data" / split
    candidates = [
        base / f"{subject}.csv",
        base / f"{subject}_{split}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise TraceError(f"MMLU CSV not found for {split}/{subject}")


def load_mmlu(sample_id: str) -> dict[str, Any]:
    _, split, subject, index_text = sample_id.split("/")
    rows = csv_rows_for_split(mmlu_csv_path(split, subject))
    index = int(index_text)
    if index >= len(rows):
        raise TraceError(f"MMLU sample index out of range: {sample_id}")
    row = rows[index]
    if len(row) < 6:
        raise TraceError(f"MMLU row has too few columns: {sample_id}")
    question, choices, answer = row[0], row[1:5], row[5]
    context = {
        "dataset": "MMLU",
        "task_type": "knowledge_choice",
        "subject": subject,
        "question": question,
        "choices": dict(zip(["A", "B", "C", "D"], choices)),
        "reference_answer": answer,
    }
    input_text = (
        f"Multiple-choice question ({subject}): {question}\n"
        f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    )
    return {
        "sample_id": sample_id,
        "dataset_key": "mmlu",
        "task_type": "knowledge_choice",
        "input_text": input_text,
        "reference": {"answer": answer},
        "sample_context": context,
    }


def load_mvtec(sample_id: str) -> dict[str, Any]:
    _, split, class_name, defect_type, filename = sample_id.split("/")
    image_path = DATASETS / "mvtec_ad" / "mvtec_anomaly_detection" / class_name / split / defect_type / filename
    if not image_path.is_file():
        raise TraceError(f"MVTec image not found: {sample_id}")
    context = {
        "dataset": "MVTec AD",
        "task_type": "industrial_quality",
        "class_name": class_name,
        "defect_type": defect_type,
        "image_path": display_path(image_path),
        "known_training_label": "normal" if defect_type == "good" else defect_type,
        "image_bytes": image_path.stat().st_size,
    }
    return {
        "sample_id": sample_id,
        "dataset_key": "mvtec_ad",
        "task_type": "industrial_quality",
        "input_text": (
            f"Industrial inspection sample for class={class_name}, split={split}, "
            f"defect_type={defect_type}. Training-good samples are normal references."
        ),
        "reference": {"label": context["known_training_label"]},
        "sample_context": context,
    }


NEU_STEM_RE = re.compile(r"(.+)_([0-9]+)$")


def neu_class_from_stem(stem: str) -> str:
    match = NEU_STEM_RE.match(stem)
    if not match:
        raise TraceError(f"Unexpected NEU-DET stem: {stem}")
    return match.group(1)


def parse_neu_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        bbox = {}
        if box is not None:
            for key in ["xmin", "ymin", "xmax", "ymax"]:
                value = box.findtext(key)
                if value is not None:
                    bbox[key] = int(float(value))
        objects.append({"name": name, "bbox": bbox})
    return {"object_count": len(objects), "objects": objects[:5]}


def load_neu_det(sample_id: str) -> dict[str, Any]:
    _, split, stem = sample_id.split("/")
    class_name = neu_class_from_stem(stem)
    base = DATASETS / "neu_det" / "NEU-DET" / split
    image_path = base / "images" / class_name / f"{stem}.jpg"
    xml_path = base / "annotations" / f"{stem}.xml"
    if not image_path.is_file():
        raise TraceError(f"NEU-DET image not found: {sample_id}")
    if not xml_path.is_file():
        raise TraceError(f"NEU-DET XML not found: {sample_id}")
    annotation = parse_neu_xml(xml_path)
    context = {
        "dataset": "NEU-DET",
        "task_type": "surface_defect",
        "class_name": class_name,
        "image_path": display_path(image_path),
        "annotation_path": display_path(xml_path),
        "annotation": annotation,
    }
    return {
        "sample_id": sample_id,
        "dataset_key": "neu_det",
        "task_type": "surface_defect",
        "input_text": (
            f"Steel surface defect sample class={class_name}. "
            f"Annotation object_count={annotation['object_count']}."
        ),
        "reference": {"label": class_name, "annotation": annotation},
        "sample_context": context,
    }


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def load_cityflow(sample_id: str) -> dict[str, Any]:
    _, split, scene, camera = sample_id.split("/")
    camera_dir = DATASETS / "cityflow" / "AICity22_Track1_MTMC_Tracking" / split / scene / camera
    if not camera_dir.is_dir():
        raise TraceError(f"CityFlow camera directory not found: {sample_id}")
    gt_path = camera_dir / "gt" / "gt.txt"
    det_path = camera_dir / "det" / "det_mask_rcnn.txt"
    calibration_path = camera_dir / "calibration.txt"
    context = {
        "dataset": "CityFlow",
        "task_type": "traffic_camera",
        "scene": scene,
        "camera": camera,
        "camera_dir": display_path(camera_dir),
        "gt_line_count": count_lines(gt_path),
        "det_mask_rcnn_line_count": count_lines(det_path),
        "has_calibration": calibration_path.is_file(),
    }
    return {
        "sample_id": sample_id,
        "dataset_key": "cityflow",
        "task_type": "traffic_camera",
        "input_text": (
            f"Traffic camera sample scene={scene}, camera={camera}, "
            f"gt_lines={context['gt_line_count']}, detections={context['det_mask_rcnn_line_count']}."
        ),
        "reference": {"gt_line_count": context["gt_line_count"]},
        "sample_context": context,
    }


LOADERS = {
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "mvtec_ad": load_mvtec,
    "neu_det": load_neu_det,
    "cityflow": load_cityflow,
}


def dataset_key_from_sample_id(sample_id: str) -> str:
    return sample_id.split("/", 1)[0]


def load_sample(sample_id: str) -> dict[str, Any]:
    key = dataset_key_from_sample_id(sample_id)
    loader = LOADERS.get(key)
    if loader is None:
        raise TraceError(f"No loader implemented for dataset key: {key}")
    return loader(sample_id)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise TraceError("Teacher response does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])
    raise TraceError("Teacher response JSON object is incomplete")


def normalize_decision(data: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    fields = ["object_state", "event_type", "risk_attr", "action", "confidence", "review_intent", "short_rationale"]
    decision: dict[str, Any] = {}
    for field in fields:
        if field not in data:
            errors.append(f"missing field: {field}")
            continue
        decision[field] = data[field]

    if decision.get("risk_attr") not in {"low", "medium", "high"}:
        errors.append("risk_attr must be low|medium|high")
    if decision.get("action") not in {"pass", "inspect", "alert", "upload"}:
        errors.append("action must be pass|inspect|alert|upload")
    if decision.get("review_intent") not in {"none", "verify_reasoning", "inspect_quality", "sync_tracking"}:
        errors.append("review_intent must be none|verify_reasoning|inspect_quality|sync_tracking")
    confidence = decision.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be a number in [0, 1]")
    elif isinstance(confidence, int):
        decision["confidence"] = float(confidence)

    evidence = data.get("evidence_items", [])
    if not isinstance(evidence, list):
        errors.append("evidence_items must be a list")
        evidence = []
    evidence_items = [str(item)[:240] for item in evidence[:5]]
    if not evidence_items:
        errors.append("evidence_items must be non-empty")
    return decision, evidence_items, errors


def request_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            text = response.read().decode("utf-8", errors="replace")
            return json.loads(text)
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise TraceError(f"HTTP {exc.code}: {body_text[:500]}") from exc
    except (URLError, OSError) as exc:
        raise TraceError(f"Teacher request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceError(f"Teacher response is not JSON: {exc}") from exc


def get_health(base_url: str, timeout_sec: float) -> int:
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/health", method="GET"), timeout=timeout_sec) as response:
            return int(response.status)
    except Exception as exc:
        raise TraceError(f"Teacher health check failed: {exc}") from exc


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


def probe_teacher_endpoints(urls: list[str], timeout_sec: float) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for url in urls:
        health_status = get_health(url, timeout_sec)
        if health_status != 200:
            raise TraceError(f"Teacher health check returned HTTP {health_status}: {url}")
        endpoints.append(
            {
                "teacher_url": url,
                "health_status": health_status,
                "served_model_id": get_served_model(url, timeout_sec, DEFAULT_TEACHER_MODEL_ID),
            }
        )
    return endpoints


def call_teacher(
    teacher_url: str,
    api_model: str,
    sample: dict[str, Any],
    timeout_sec: float,
    max_tokens: int,
) -> dict[str, Any]:
    sample_context = json.dumps(sample["sample_context"], ensure_ascii=False, sort_keys=True, indent=2)
    prompt = PROMPT_TEMPLATE.replace("{sample_context}", sample_context)
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = request_json(f"{teacher_url.rstrip('/')}/v1/chat/completions", payload, timeout_sec)
    latency_ms = (time.perf_counter() - started) * 1000
    choices = response.get("choices", [])
    if not choices:
        raise TraceError("Teacher response has no choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    response_text = str(message.get("content", ""))
    parsed = extract_json_object(response_text)
    decision, evidence_items, parse_errors = normalize_decision(parsed)
    prompt_hash = sha256_text(SYSTEM_PROMPT + "\n" + prompt)
    request_hash = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return {
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "request_hash": request_hash,
        "response": response,
        "response_text": response_text,
        "parsed_response": parsed,
        "decision_tuple": decision,
        "evidence_items": evidence_items,
        "parse_errors": parse_errors,
        "latency_ms": latency_ms,
    }


def call_teacher_with_retries(
    teacher_url: str,
    api_model: str,
    sample: dict[str, Any],
    timeout_sec: float,
    max_tokens: int,
    retry_count: int,
    retry_sleep_sec: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            return call_teacher(teacher_url, api_model, sample, timeout_sec, max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_sleep_sec)
    assert last_error is not None
    raise last_error


def dry_run_teacher_result(sample_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_hash": sha256_text("dry-run"),
        "request_hash": sha256_text(sample_id),
        "response_text": "{}",
        "parsed_response": {},
        "decision_tuple": {
            "object_state": "dry_run",
            "event_type": sample["task_type"],
            "risk_attr": "low",
            "action": "pass",
            "confidence": 1.0,
            "review_intent": "none",
            "short_rationale": "dry run",
        },
        "evidence_items": ["dry run"],
        "parse_errors": [],
        "latency_ms": 0.0,
    }


def build_trace_row(
    sample: dict[str, Any],
    teacher_result: dict[str, Any],
    split: str,
    split_hash: str,
    teacher_url: str,
    api_model: str,
    created_ts: str,
) -> dict[str, Any]:
    trace_row = {
        "trace_version": "1.0",
        "created_ts": created_ts,
        "created_by": "model_compression/generate_teacher_traces.py",
        "sample_id": sample["sample_id"],
        "dataset_key": sample["dataset_key"],
        "split": split,
        "split_hash": split_hash,
        "task_type": sample["task_type"],
        "teacher_model_id": DEFAULT_TEACHER_MODEL_ID,
        "served_model_id": api_model,
        "teacher_url": teacher_url,
        "prompt_hash": teacher_result["prompt_hash"],
        "request_hash": teacher_result["request_hash"],
        "input_text": sample["input_text"],
        "reference": sample["reference"],
        "sample_context": sample["sample_context"],
        "teacher_response_text": teacher_result["response_text"],
        "teacher_response_json": teacher_result["parsed_response"],
        "decision_tuple": teacher_result["decision_tuple"],
        "evidence_items": teacher_result["evidence_items"],
        "parse_ok": not teacher_result["parse_errors"],
        "parse_errors": teacher_result["parse_errors"],
        "latency_ms": teacher_result["latency_ms"],
    }
    trace_row["trace_row_hash"] = sha256_text(json.dumps(trace_row, ensure_ascii=False, sort_keys=True))
    return trace_row


def build_distill_row(trace_row: dict[str, Any]) -> dict[str, Any]:
    target = {
        "decision_tuple": trace_row["decision_tuple"],
        "evidence_items": trace_row["evidence_items"],
        "short_rationale": trace_row["decision_tuple"].get("short_rationale", ""),
    }
    row = {
        "distill_version": "1.0",
        "sample_id": trace_row["sample_id"],
        "dataset_key": trace_row["dataset_key"],
        "split": trace_row["split"],
        "task_type": trace_row["task_type"],
        "input_text": trace_row["input_text"],
        "target_json": target,
        "teacher_trace_row_hash": trace_row["trace_row_hash"],
        "prompt_hash": trace_row["prompt_hash"],
        "used_for_training": trace_row["split"] == "train",
    }
    row["distill_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def parse_dataset_filter(values: list[str]) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for value in values:
        selected.update(part.strip() for part in value.split(",") if part.strip())
    return selected or None


def parse_comma_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def parse_teacher_urls(teacher_url_values: list[str], teacher_urls_values: list[str]) -> list[str]:
    urls = parse_comma_values(teacher_url_values) + parse_comma_values(teacher_urls_values)
    if not urls:
        urls = [DEFAULT_TEACHER_URL]

    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            unique_urls.append(normalized)
    return unique_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 14B teacher traces for G-KD-TRACE.")
    parser.add_argument(
        "--teacher-url",
        "--teacher_url",
        dest="teacher_url",
        action="append",
        default=[],
        help="Teacher base URL. Can be repeated for parallel teacher replicas.",
    )
    parser.add_argument(
        "--teacher-urls",
        "--teacher_urls",
        dest="teacher_urls",
        action="append",
        default=[],
        help="Comma-separated teacher base URLs.",
    )
    parser.add_argument("--split", default="train", choices=sorted(ALLOWED_SPLITS | {"test"}))
    parser.add_argument("--sample-limit", "--sample_limit", type=int, default=None)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset key filter. Can be repeated or comma-separated.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent teacher requests.")
    parser.add_argument("--num-shards", "--num_shards", type=int, default=1)
    parser.add_argument("--shard-index", "--shard_index", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retry-count", "--retry_count", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", "--retry_sleep_sec", type=float, default=1.0)
    parser.add_argument("--output-teacher-trace", "--output_teacher_trace", default=str(DEFAULT_TEACHER_TRACE))
    parser.add_argument("--output-distill", "--output_distill", default=str(DEFAULT_DISTILL))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--min-parse-rate", type=float, default=0.9)
    parser.add_argument(
        "--checkpoint-interval",
        "--checkpoint_interval",
        type=int,
        default=25,
        help="Write checkpoint JSONL files and a partial audit every N completed attempts. Use 0 to disable.",
    )
    parser.add_argument(
        "--partial-audit",
        "--partial_audit",
        default=None,
        help="Path for checkpoint audit JSON. Defaults to <audit stem>.partial.json.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing trace rows and process only missing sample ids.")
    parser.add_argument("--dry-run", action="store_true", help="Build samples and audit without calling the teacher service.")
    return parser.parse_args()


def resolve_output(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def default_partial_audit_path(audit_path: Path) -> Path:
    return audit_path.with_name(f"{audit_path.stem}.partial{audit_path.suffix or '.json'}")


def main() -> int:
    args = parse_args()
    if args.split == "test":
        print("Refusing to generate teacher traces from final test split.", file=sys.stderr)
        return 2
    if args.sample_limit is not None and args.sample_limit <= 0:
        print("--sample-limit must be positive", file=sys.stderr)
        return 2
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.retry_count < 0:
        print("--retry-count must be non-negative", file=sys.stderr)
        return 2
    if args.checkpoint_interval < 0:
        print("--checkpoint-interval must be non-negative", file=sys.stderr)
        return 2

    teacher_trace_path = resolve_output(args.output_teacher_trace)
    distill_path = resolve_output(args.output_distill)
    audit_path = resolve_output(args.audit)
    partial_audit_path = resolve_output(args.partial_audit) if args.partial_audit else default_partial_audit_path(audit_path)
    include_datasets = parse_dataset_filter(args.dataset)
    teacher_urls = parse_teacher_urls(args.teacher_url, args.teacher_urls)

    frozen = load_frozen_splits()
    all_ids_by_dataset = read_split_ids(args.split)
    skipped_dataset_keys = sorted(key for key, ids in all_ids_by_dataset.items() if ids and key not in LOADERS)
    ids_by_dataset = {key: ids for key, ids in all_ids_by_dataset.items() if key in LOADERS}
    selected_ids = select_sample_ids(ids_by_dataset, include_datasets, args.sample_limit)
    selected_ids = apply_shard(selected_ids, args.num_shards, args.shard_index)
    if not selected_ids:
        raise TraceError("Shard selection produced no sample ids")

    created_ts = datetime.now(timezone.utc).isoformat()
    teacher_endpoints: list[dict[str, Any]]
    if not args.dry_run:
        teacher_endpoints = probe_teacher_endpoints(teacher_urls, args.timeout_sec)
    else:
        teacher_endpoints = [
            {
                "teacher_url": "dry-run",
                "health_status": None,
                "served_model_id": DEFAULT_TEACHER_MODEL_ID,
            }
        ]

    existing_trace_rows = load_trace_rows_by_sample_id(teacher_trace_path) if args.resume else {}
    selected_id_set = set(selected_ids)
    trace_rows_by_id = {
        sample_id: row
        for sample_id, row in existing_trace_rows.items()
        if sample_id in selected_id_set
    }
    initial_resumed_count = len(trace_rows_by_id)
    tasks = [
        (index, sample_id)
        for index, sample_id in enumerate(selected_ids, start=1)
        if sample_id not in trace_rows_by_id
    ]

    errors: list[dict[str, Any]] = []
    processed_count = 0
    completed_attempt_count = 0

    def generate_one(index: int, sample_id: str) -> dict[str, Any]:
        sample = load_sample(sample_id)
        split_hash = split_hash_for(frozen, sample["dataset_key"], args.split)
        endpoint = teacher_endpoints[(index - 1) % len(teacher_endpoints)]
        teacher_url = str(endpoint["teacher_url"])
        api_model = str(endpoint["served_model_id"])
        if args.dry_run:
            teacher_result = dry_run_teacher_result(sample_id, sample)
        else:
            teacher_result = call_teacher_with_retries(
                teacher_url,
                api_model,
                sample,
                args.timeout_sec,
                args.max_tokens,
                args.retry_count,
                args.retry_sleep_sec,
            )
        return build_trace_row(
            sample,
            teacher_result,
            args.split,
            split_hash,
            teacher_url,
            api_model,
            created_ts,
        )

    def record_success(index: int, trace_row: dict[str, Any]) -> None:
        nonlocal completed_attempt_count, processed_count
        processed_count += 1
        completed_attempt_count += 1
        trace_rows_by_id[str(trace_row["sample_id"])] = trace_row
        print(
            f"[OK] {processed_count}/{len(tasks)} selected={index}/{len(selected_ids)} "
            f"{trace_row['sample_id']} endpoint={trace_row['teacher_url']} "
            f"action={trace_row['decision_tuple'].get('action')} "
            f"parse_ok={trace_row['parse_ok']} latency_ms={trace_row['latency_ms']:.1f}",
            flush=True,
        )

    def record_error(index: int, sample_id: str, exc: Exception) -> None:
        nonlocal completed_attempt_count
        completed_attempt_count += 1
        errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        print(f"[FAIL] selected={index}/{len(selected_ids)} {sample_id}: {exc}", flush=True)

    def current_trace_rows() -> list[dict[str, Any]]:
        return [trace_rows_by_id[sample_id] for sample_id in selected_ids if sample_id in trace_rows_by_id]

    def write_outputs_and_audit(target_audit_path: Path, status: str, is_partial: bool) -> dict[str, Any]:
        trace_rows = current_trace_rows()
        distill_rows = [build_distill_row(trace_row) for trace_row in trace_rows]
        write_jsonl(teacher_trace_path, trace_rows)
        write_jsonl(distill_path, distill_rows)

        parse_ok_count = sum(1 for row in trace_rows if row.get("parse_ok") is True)
        parse_rate = parse_ok_count / len(trace_rows) if trace_rows else 0.0
        dataset_counts = Counter(row["dataset_key"] for row in trace_rows)
        action_counts = Counter(row["decision_tuple"].get("action", "") for row in trace_rows)
        endpoint_counts = Counter(row["teacher_url"] for row in trace_rows)
        sample_hash = sha256_text("\n".join(selected_ids) + "\n")
        first_endpoint = teacher_endpoints[0]
        audit = {
            "gate": "G-KD-TRACE-teacher-smoke" if args.sample_limit else "G-KD-TRACE-teacher",
            "check_version": "1.2",
            "created_by": "model_compression/generate_teacher_traces.py",
            "created_ts": created_ts,
            "updated_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "is_partial": is_partial,
            "teacher_url": first_endpoint["teacher_url"],
            "teacher_urls": [endpoint["teacher_url"] for endpoint in teacher_endpoints],
            "teacher_model_id": DEFAULT_TEACHER_MODEL_ID,
            "served_model_id": first_endpoint["served_model_id"],
            "teacher_endpoints": teacher_endpoints,
            "teacher_health_status": first_endpoint["health_status"],
            "teacher_health_statuses": {
                str(endpoint["teacher_url"]): endpoint["health_status"]
                for endpoint in teacher_endpoints
            },
            "split": args.split,
            "global_split_hash": frozen.get("global_split_hash", ""),
            "sample_limit": args.sample_limit,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "workers": args.workers,
            "retry_count": args.retry_count,
            "resume": bool(args.resume),
            "checkpoint_interval": args.checkpoint_interval,
            "partial_audit_path": display_path(partial_audit_path),
            "available_dataset_keys": sorted(ids_by_dataset),
            "empty_split_dataset_keys": empty_split_dataset_keys(frozen, args.split),
            "skipped_dataset_keys": skipped_dataset_keys,
            "selected_sample_count": len(selected_ids),
            "resumed_trace_count": initial_resumed_count,
            "attempted_trace_count": len(tasks),
            "completed_attempt_count": completed_attempt_count,
            "processed_trace_count": processed_count,
            "successful_trace_count": len(trace_rows),
            "failed_trace_count": len(errors),
            "pending_trace_count": len(selected_ids) - len(trace_rows) - len(errors),
            "parse_ok_count": parse_ok_count,
            "parse_success_rate": parse_rate,
            "min_parse_rate": args.min_parse_rate,
            "selected_sample_ids_hash": sample_hash,
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
            "teacher_trace_path": display_path(teacher_trace_path),
            "teacher_trace_hash": sha256_file(teacher_trace_path),
            "distill_dataset_path": display_path(distill_path),
            "distill_dataset_hash": sha256_file(distill_path),
            "prompt_template_hash": sha256_text(SYSTEM_PROMPT + "\n" + PROMPT_TEMPLATE),
            "dry_run": bool(args.dry_run),
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(
            json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
        )
        write_json(target_audit_path, audit)
        return audit

    def maybe_write_checkpoint() -> None:
        if args.checkpoint_interval <= 0:
            return
        if completed_attempt_count == 0 or completed_attempt_count % args.checkpoint_interval != 0:
            return
        if completed_attempt_count == len(tasks):
            status = "completed_with_errors" if errors else "completed"
        else:
            status = "running_with_errors" if errors else "running"
        audit = write_outputs_and_audit(partial_audit_path, status, is_partial=True)
        print(
            f"[CHECKPOINT] completed_attempts={completed_attempt_count}/{len(tasks)} "
            f"successful={audit['successful_trace_count']} failed={audit['failed_trace_count']} "
            f"wrote={display_path(partial_audit_path)}",
            flush=True,
        )

    if tasks and args.workers == 1:
        for index, sample_id in tasks:
            try:
                record_success(index, generate_one(index, sample_id))
            except Exception as exc:
                record_error(index, sample_id, exc)
            maybe_write_checkpoint()
    elif tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_one, index, sample_id): (index, sample_id)
                for index, sample_id in tasks
            }
            for future in as_completed(futures):
                index, sample_id = futures[future]
                try:
                    record_success(index, future.result())
                except Exception as exc:
                    record_error(index, sample_id, exc)
                maybe_write_checkpoint()

    if not tasks:
        print(f"[OK] resume found all {len(current_trace_rows())} selected trace rows; no teacher calls needed.", flush=True)

    final_trace_rows = current_trace_rows()
    final_parse_ok_count = sum(1 for row in final_trace_rows if row.get("parse_ok") is True)
    final_parse_rate = final_parse_ok_count / len(final_trace_rows) if final_trace_rows else 0.0
    missing_count = len(selected_ids) - len(final_trace_rows) - len(errors)
    final_status = "passed" if not errors and missing_count == 0 and final_parse_rate >= args.min_parse_rate else "failed"
    audit = write_outputs_and_audit(audit_path, final_status, is_partial=False)

    print(f"Wrote {display_path(teacher_trace_path)}")
    print(f"Wrote {display_path(distill_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"teacher_trace_hash={audit['teacher_trace_hash']}")
    print(f"distill_dataset_hash={audit['distill_dataset_hash']}")
    if audit["status"] != "passed":
        print("Teacher trace generation failed gate checks.", file=sys.stderr)
        return 1
    print("Teacher trace generation passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TraceError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
