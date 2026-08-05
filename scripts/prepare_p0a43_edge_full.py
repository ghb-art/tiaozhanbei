#!/usr/bin/env python3
"""Freeze the P0-A43 full-test identity from the sealed 14B baseline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a43_edge_full.json"
AUDIT = ROOT / "reports/audit/gate_p0a43_preflight.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")


class PreflightError(RuntimeError):
    pass


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
        raise PreflightError(f"Missing JSON: {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"JSON root is not an object: {path.relative_to(ROOT)}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreflightError(f"Missing JSONL: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PreflightError(f"Non-object row: {path.relative_to(ROOT)}:{number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    try:
        from evaluate_chapter2_capability import SAMPLE_LOADERS, build_messages

        config = read_json(CONFIG)
        formal = config["formal_test"]
        candidate = config["candidate"]
        baseline_trace = ROOT / formal["baseline_trace"]
        baseline_audit_path = ROOT / formal["baseline_audit"]
        split_dir = ROOT / formal["split_dir"]
        baseline_audit = read_json(baseline_audit_path)
        if baseline_audit.get("status") != "passed":
            raise PreflightError("Sealed 14B baseline audit is not passed")
        if sha256_file(baseline_trace) != baseline_audit.get("sealed_trace_hash"):
            raise PreflightError("Sealed 14B baseline trace hash changed")
        rows = read_jsonl(baseline_trace)
        expected_counts = {key: int(value) for key, value in formal["counts"].items()}
        counts = Counter(str(row.get("dataset_key", "")) for row in rows)
        if dict(counts) != expected_counts:
            raise PreflightError(f"Baseline counts changed: {dict(counts)}")
        ids_by_task = {
            task: [str(row["sample_id"]) for row in rows if row.get("dataset_key") == task]
            for task in TASKS
        }
        all_ids = [sample_id for task in TASKS for sample_id in ids_by_task[task]]
        if len(all_ids) != len(set(all_ids)):
            raise PreflightError("Official-full baseline contains duplicate sample ids")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_hashes: dict[str, str] = {}
        for task, ids in ids_by_task.items():
            path = split_dir / f"{task}_test.txt"
            content = "\n".join(ids) + "\n"
            path.write_text(content, encoding="utf-8", newline="\n")
            split_hashes[task] = sha256_text(content)
        prompt_hashes = [str(row.get("prompt_hash", "")) for row in rows]
        if any(not value for value in prompt_hashes):
            raise PreflightError("Baseline contains an empty prompt hash")
        prompt_mismatches = 0
        for row in rows:
            dataset_key = str(row["dataset_key"])
            sample = SAMPLE_LOADERS[dataset_key](str(row["sample_id"]))
            _, current_prompt_hash = build_messages(sample, str(formal["prompt_style"]))
            if current_prompt_hash != row["prompt_hash"]:
                prompt_mismatches += 1
        if prompt_mismatches:
            raise PreflightError(
                f"Current evaluator differs from the sealed 14B prompts: {prompt_mismatches}"
            )
        selection_path = ROOT / "reports/audit/gate_p0a42_domain_selection.json"
        selection = read_json(selection_path)
        if selection.get("status") != "passed":
            raise PreflightError("P0-A42 domain selection is not passed")
        selected = selection.get("selected", {})
        expected_routes = {
            "math": "original-base",
            "code": "p0a25-code-192",
            "nlp": "p0a10-nlp-136",
        }
        actual_routes = {key: selected.get(key, {}).get("name") for key in expected_routes}
        if actual_routes != expected_routes:
            raise PreflightError(f"Best route identity changed: {actual_routes}")
        required_paths = {
            "base_hf": ROOT / candidate["base_hf"],
            "base_gguf": ROOT / candidate["base_gguf"],
            "code_hf_adapter": ROOT / candidate["code_route"],
            "code_gguf_adapter": ROOT / candidate["code_gguf_adapter"],
            "nlp_hf_adapter": ROOT / candidate["nlp_route"],
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise PreflightError(f"Missing candidate artifacts: {missing}")
        final_paths = (
            ROOT / "reports/sealed/p0a43/edge_best_router_q4_full.jsonl",
            ROOT / "reports/audit/gate_p0a43_edge_best_router_q4_full.json",
            ROOT / "reports/audit/gate_p0a43_edge_best_router_q4_full_retention.json",
        )
        if any(path.exists() for path in final_paths):
            raise PreflightError("P0-A43 full-test artifacts already exist; repeat is forbidden")
        manifest = {
            "stage": "P0-A43",
            "created_by": "scripts/prepare_p0a43_edge_full.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "source_baseline_trace": formal["baseline_trace"],
            "source_baseline_trace_hash": sha256_file(baseline_trace),
            "source_baseline_audit": formal["baseline_audit"],
            "source_baseline_audit_hash": sha256_file(baseline_audit_path),
            "counts": expected_counts,
            "split_hashes": split_hashes,
            "ordered_sample_ids_hash": sha256_text("\n".join(all_ids) + "\n"),
            "ordered_prompt_hashes_hash": sha256_text("\n".join(prompt_hashes) + "\n"),
            "prompt_style": formal["prompt_style"],
            "feedback_policy": "aggregate_only_no_retraining",
        }
        manifest["manifest_hash"] = sha256_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        write_json(split_dir / "manifest.json", manifest)
        report = {
            "gate": "P0-A43-EDGE-OFFICIAL-FULL-PREFLIGHT",
            "check_version": "1.0",
            "created_by": "scripts/prepare_p0a43_edge_full.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "config": CONFIG.relative_to(ROOT).as_posix(),
            "config_hash": sha256_file(CONFIG),
            "selection": selection_path.relative_to(ROOT).as_posix(),
            "selection_hash": sha256_file(selection_path),
            "selected_routes": actual_routes,
            "split_dir": split_dir.relative_to(ROOT).as_posix(),
            "split_manifest_hash": sha256_file(split_dir / "manifest.json"),
            "sample_count": len(rows),
            "dataset_counts": expected_counts,
            "baseline_trace_hash": sha256_file(baseline_trace),
            "prompt_mismatch_count": prompt_mismatches,
            "base_gguf_hash": sha256_file(required_paths["base_gguf"]),
            "code_gguf_adapter_hash": sha256_file(required_paths["code_gguf_adapter"]),
            "nlp_hf_adapter_hash": sha256_file(required_paths["nlp_hf_adapter"] / "adapter_model.safetensors"),
            "formal_full_opened": False,
            "item_level_feedback_allowed_for_training": False,
        }
        report["report_hash"] = sha256_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        write_json(AUDIT, report)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, PreflightError) as exc:
        print(f"P0-A43 preflight failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {AUDIT.relative_to(ROOT)}")
    print(f"status=passed rows={len(rows)} counts={expected_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
