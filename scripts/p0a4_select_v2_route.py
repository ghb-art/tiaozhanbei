#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RATIO_KEYS = ("math_ratio", "code_ratio", "nlp_ratio")


class RouteError(RuntimeError):
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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RouteError(f"Missing route evidence: {display_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RouteError(f"Route evidence must be an object: {display_path(path)}")
    return value


def verify_route(
    route: str,
    evaluation_path: Path,
    retention_path: Path,
    quant_hash: str,
) -> dict[str, Any]:
    evaluation = load(evaluation_path)
    retention = load(retention_path)
    if evaluation.get("status") != "passed":
        raise RouteError(f"{route} did not complete the 96-row evaluation")
    if evaluation.get("dataset_counts") != {"cmmlu": 32, "gsm8k": 32, "humaneval": 32}:
        raise RouteError(f"{route} does not contain exactly 32+32+32 rows")
    if evaluation.get("disable_thinking") is not True or evaluation.get("kv_cache_type") != "q8_0":
        raise RouteError(f"{route} runtime is not thinking=off with Q8 KV")
    if evaluation.get("model", {}).get("sha256") != quant_hash:
        raise RouteError(f"{route} evaluation used a different quantized Student")
    if retention.get("candidate_trace_hash") != evaluation.get("output_trace_sha256"):
        raise RouteError(f"{route} retention report is not bound to its evaluation trace")
    if int(retention.get("generation_error_count", -1)) != 0:
        raise RouteError(f"{route} contains generation errors")
    ratios = {key: float(retention.get("ratios", {}).get(key, 0.0)) for key in RATIO_KEYS}
    macro = float(retention.get("capped_macro_ratio", 0.0))
    return {
        "route": route,
        "evaluation_audit": display_path(evaluation_path),
        "evaluation_audit_hash": sha256_file(evaluation_path),
        "retention_audit": display_path(retention_path),
        "retention_audit_hash": sha256_file(retention_path),
        "retention_gate_status": str(retention.get("status", "")),
        "ratios": ratios,
        "capped_macro_ratio": macro,
        "worst_task_ratio": min(ratios.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the final P0-A4 Student v2 route using aggregate smoke96 only.")
    parser.add_argument("--config", default="configs/p0a4_distillation.json")
    parser.add_argument("--quant-audit", default="reports/audit/gate_p0a4_student_v2_q4_prepare.json")
    parser.add_argument("--shared-eval", default="reports/audit/gate_p0a4_edge_student_v2_q4_k_m_smoke96.json")
    parser.add_argument("--shared-retention", default="reports/audit/gate_p0a4_edge_student_v2_q4_k_m_smoke96_retention.json")
    parser.add_argument("--adapter-eval", default="reports/audit/gate_p0a4_edge_student_v2_adapter_smoke96_eval.json")
    parser.add_argument("--adapter-retention", default="reports/audit/gate_p0a4_edge_student_v2_adapter_smoke96_retention.json")
    parser.add_argument("--adapter-manifest", default="models/adapters/p0a4/v2/router_manifest.json")
    parser.add_argument("--adapter-audit", default="reports/audit/gate_p0a4_adapter_router_prepare_v2.json")
    parser.add_argument("--output", default="reports/audit/gate_p0a4_student_v2_route_selection.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config)
        config = load(config_path)
        gate = config["gates"]["v2_promotion"]
        if gate.get("feedback") != "smoke96_aggregate_only" or gate.get("full_test_feedback_used") is not False:
            raise RouteError("v2 promotion policy must forbid full-test feedback")
        quant_path = resolve_path(args.quant_audit)
        quant = load(quant_path)
        if quant.get("status") != "passed" or quant.get("quant_type") != "Q4_K_M":
            raise RouteError("Student v2 quantization gate did not pass")
        quant_hash = str(quant.get("quantized_gguf_hash", ""))
        routes = [
            verify_route("shared", resolve_path(args.shared_eval), resolve_path(args.shared_retention), quant_hash),
            verify_route("adapter_top1", resolve_path(args.adapter_eval), resolve_path(args.adapter_retention), quant_hash),
        ]
        manifest_path = resolve_path(args.adapter_manifest)
        adapter_audit_path = resolve_path(args.adapter_audit)
        adapter_audit = load(adapter_audit_path)
        if adapter_audit.get("status") != "passed" or adapter_audit.get("manifest_hash") != sha256_file(manifest_path):
            raise RouteError("Adapter manifest is not bound to a passed preparation audit")
        minimum = float(gate["min_ratio_per_task"])
        macro_minimum = float(gate["min_capped_macro_ratio"])
        eligible = [
            route for route in routes
            if route["worst_task_ratio"] >= minimum
            and route["capped_macro_ratio"] >= macro_minimum
        ]
        selected = max(
            eligible,
            key=lambda route: (route["worst_task_ratio"], route["capped_macro_ratio"], route["route"] == "shared"),
        ) if eligible else None
        audit = {
            "gate": "P0-A4-STUDENT-V2-ROUTE-SELECTION",
            "check_version": "1.0",
            "created_by": "scripts/p0a4_select_v2_route.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if selected else "failed",
            "feedback_source": "smoke96_aggregate_only",
            "full_test_feedback_used": False,
            "selection_metric": "maximize worst task retention, then capped macro retention",
            "min_ratio_per_task": minimum,
            "min_capped_macro_ratio": macro_minimum,
            "quantization_audit": display_path(quant_path),
            "quantization_audit_hash": sha256_file(quant_path),
            "quantized_gguf_hash": quant_hash,
            "adapter_manifest": display_path(manifest_path),
            "adapter_manifest_hash": sha256_file(manifest_path),
            "adapter_audit": display_path(adapter_audit_path),
            "adapter_audit_hash": sha256_file(adapter_audit_path),
            "routes": routes,
            "selected_route": selected["route"] if selected else "",
            "failures": [] if selected else ["No v2 route reached the 85% per-task and macro promotion margin"],
        }
        audit["report_hash"] = sha256_json(audit)
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError, json.JSONDecodeError, RouteError) as exc:
        print(f"P0-A4 v2 route selection failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {display_path(output)}")
    print(f"status={audit['status']} selected_route={audit['selected_route']}")
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
