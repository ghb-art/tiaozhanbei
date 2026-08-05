#!/usr/bin/env python3
"""Freeze P0-A44 deployment routes and copy the already-audited full split identity."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a44_aligned_retrain.json"
SELECTION = ROOT / "reports/audit/gate_p0a44_q4_selection.json"
OUT = ROOT / "reports/audit/gate_p0a44_edge_full_preflight.json"
TASKS = ("gsm8k", "humaneval", "cmmlu")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("status") != "passed":
        raise RuntimeError("Q4 selection not passed")
    formal = cfg["formal_full"]
    source = ROOT / formal["split_source"]
    target = ROOT / formal["split_dir"]
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = {key: int(value) for key, value in formal["datasets"].items()}
    target.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for task in TASKS:
        src = source / f"{task}_test.txt"; dst = target / f"{task}_test.txt"
        ids = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != expected[task] or len(ids) != len(set(ids)):
            raise RuntimeError(f"Invalid frozen split {task}: {len(ids)}")
        content = "\n".join(ids) + "\n"; dst.write_text(content, encoding="utf-8", newline="\n")
        if sha(dst) != sha(src): raise RuntimeError(f"Split copy changed: {task}")
        split_hashes[task] = sha(dst)
    routes = selection["selected"]
    base = ROOT / "models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"
    artifacts = {"base": {"path": str(base.relative_to(ROOT)), "sha256": sha(base)}}
    for domain in ("code", "nlp"):
        adapter = str(routes[domain]["adapter"])
        if adapter:
            path = ROOT / adapter
            if not path.is_file(): raise RuntimeError(f"Missing selected adapter: {adapter}")
            artifacts[domain] = {"path": adapter, "sha256": sha(path), "bytes": path.stat().st_size}
    final_paths = (
        ROOT / "reports/sealed/p0a44/edge_aligned_router_q4_full.jsonl",
        ROOT / "reports/audit/gate_p0a44_edge_aligned_router_q4_full.json",
        ROOT / "reports/audit/gate_p0a44_edge_aligned_router_q4_full_retention.json",
    )
    if any(path.exists() for path in final_paths):
        raise RuntimeError("P0-A44 full artifacts already exist; repeat refused")
    manifest = {
        "stage": "P0-A44", "created_ts": datetime.now(timezone.utc).isoformat(),
        "source_split_manifest_hash": sha(source / "manifest.json"),
        "source_ordered_sample_ids_hash": source_manifest["ordered_sample_ids_hash"],
        "counts": expected, "split_hashes": split_hashes, "prompt_style": formal["prompt_style"],
        "feedback_policy": "aggregate_only_no_retraining", "selected_routes": routes,
    }
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "gate": "P0-A44-EDGE-FULL-PREFLIGHT", "check_version": "1.0",
        "created_by": "scripts/prepare_p0a44_edge_full.py", "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "selected_routes": routes, "artifacts": artifacts,
        "dataset_counts": expected, "split_dir": formal["split_dir"],
        "split_manifest_hash": sha(target / "manifest.json"), "formal_full_opened": False,
        "item_level_feedback_allowed_for_training": False,
    }
    audit["report_hash"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} status=passed routes={routes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
