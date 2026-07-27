#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_edge.runtime import (  # noqa: E402
    CloudEdgeRuntime,
    ConflictArbitrator,
    FastPath,
    NetworkState,
    Scheduler,
    TaskRequest,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def request_for(index: int, scenario: str, rng: random.Random) -> TaskRequest:
    confidence = rng.uniform(0.68, 0.99)
    risk = rng.uniform(0.05, 0.95)
    if scenario == "industrial":
        payload: dict[str, Any] = {
            "temperature_c": rng.uniform(40, 98),
            "vibration_mm_s": rng.uniform(1, 14),
            "defect_score": rng.uniform(0.05, 0.99),
        }
    else:
        payload = {
            "queue_length": rng.randint(0, 50),
            "emergency_vehicle": index % 41 == 0,
            "pedestrian_conflict": index % 17 == 0,
        }
    return TaskRequest(
        request_id=f"{scenario}-{index:05d}",
        event_id=f"{scenario}-event-{index // 2:05d}",
        node_id=f"edge-{index % 4}",
        scenario=scenario,
        deadline_ms=200.0,
        complexity=rng.random(),
        confidence=confidence,
        risk=risk,
        payload=payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic industrial/traffic cloud-edge protocol simulation."
    )
    parser.add_argument("--config", default="configs/cloud_edge_runtime.json")
    parser.add_argument("--requests-per-scenario", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202606)
    parser.add_argument(
        "--audit", default="reports/audit/gate_cloud_edge_system_simulation.json"
    )
    args = parser.parse_args()
    try:
        config_path = ROOT / args.config
        config = json.loads(config_path.read_text(encoding="utf-8"))
        scheduler_config = config["scheduler"]
        runtime = CloudEdgeRuntime(
            fast_path=FastPath(config["fast_path"]),
            scheduler=Scheduler(
                float(scheduler_config["edge_model_latency_ms"]),
                float(scheduler_config["cloud_model_latency_ms"]),
            )
        )
        arbitrator = ConflictArbitrator()
        rng = random.Random(args.seed)
        decisions = []
        weak_requests = 0
        weak_completed = 0
        route_counts: Counter[str] = Counter()

        def edge_model(request: TaskRequest) -> tuple[str, float]:
            if request.scenario == "industrial":
                return ("slow_and_inspect" if request.risk >= 0.5 else "continue", 0.82)
            return ("extend_green" if request.risk >= 0.5 else "normal_cycle", 0.81)

        def cloud_model(request: TaskRequest) -> tuple[str, float]:
            if request.scenario == "industrial":
                return ("emergency_stop" if request.risk >= 0.8 else "slow_and_inspect", 0.93)
            return ("hold_vehicle_phase" if request.risk >= 0.8 else "extend_green", 0.92)

        for scenario in ("industrial", "traffic"):
            for index in range(args.requests_per_scenario):
                weak = index % 5 == 0
                offline = index % 19 == 0
                network = NetworkState(
                    available=not offline,
                    rtt_ms=160.0 if weak else 18.0,
                    loss_rate=0.2 if weak else 0.01,
                    bandwidth_mbps=1.0 if weak else 100.0,
                )
                request = request_for(index, scenario, rng)
                if network.weak:
                    weak_requests += 1
                decision = runtime.process(request, network, edge_model, cloud_model)
                decisions.append(decision)
                route_counts[decision.route] += 1
                if network.weak and decision.action:
                    weak_completed += 1

        by_event: dict[str, list[Any]] = defaultdict(list)
        for decision in decisions:
            by_event[decision.event_id].append(decision)
        raw_conflicts = 0
        resolved = 0
        unresolved = 0
        for group in by_event.values():
            if len({decision.action for decision in group}) <= 1:
                continue
            raw_conflicts += 1
            try:
                _, had_conflict = arbitrator.resolve(group)
                resolved += int(had_conflict)
            except ValueError:
                unresolved += 1
        queued_during_weak_network = runtime.outbox.pending_count
        synced_after_recovery = runtime.outbox.flush(
            NetworkState(True, 15.0, 0.0, 100.0),
            lambda _record: True,
        )
        latencies = [decision.latency_ms for decision in decisions]
        weak_rate = weak_completed / weak_requests if weak_requests else 0.0
        resolution_rate = resolved / raw_conflicts if raw_conflicts else 1.0
        post_conflict_ratio = unresolved / len(by_event) if by_event else 0.0
        metrics = {
            "scenario_count": 2,
            "request_count": len(decisions),
            "mean_end_to_end_latency_ms": statistics.fmean(latencies),
            "p95_end_to_end_latency_ms": percentile(latencies, 0.95),
            "weak_network_request_count": weak_requests,
            "weak_network_functionality_rate": weak_rate,
            "raw_conflict_group_count": raw_conflicts,
            "raw_conflict_ratio": raw_conflicts / len(by_event),
            "post_arbitration_unresolved_count": unresolved,
            "post_arbitration_conflict_ratio": post_conflict_ratio,
            "conflict_resolution_success_rate": resolution_rate,
            "route_counts": dict(route_counts),
            "outbox_queued_during_weak_network": queued_during_weak_network,
            "outbox_synced_after_recovery": synced_after_recovery,
            "outbox_pending_count": runtime.outbox.pending_count,
        }
        gates = config["gates"]
        failures: list[str] = []
        if metrics["scenario_count"] < int(gates["scenario_count"]):
            failures.append("scenario_count")
        if metrics["p95_end_to_end_latency_ms"] > float(
            gates["p95_end_to_end_latency_ms"]
        ):
            failures.append("p95_end_to_end_latency")
        if metrics["weak_network_functionality_rate"] < float(
            gates["weak_network_functionality_rate"]
        ):
            failures.append("weak_network_functionality")
        if metrics["post_arbitration_conflict_ratio"] > float(
            gates["post_arbitration_conflict_ratio"]
        ):
            failures.append("post_arbitration_conflict_ratio")
        if metrics["conflict_resolution_success_rate"] < float(
            gates["conflict_resolution_success_rate"]
        ):
            failures.append("conflict_resolution_success")
        report = {
            "gate": "CLOUD-EDGE-SYSTEM-SIMULATION",
            "check_version": "1.0",
            "created_by": "scripts/simulate_cloud_edge.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not failures else "failed",
            "evidence_scope": "deterministic protocol simulation; not formal hardware evidence",
            "scenarios": ["industrial", "traffic"],
            "config": args.config,
            "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "metrics": metrics,
            "gates": gates,
            "failures": failures,
        }
        report["report_hash"] = hashlib.sha256(
            json.dumps(
                report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        output = ROOT / args.audit
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Cloud-edge simulation status={report['status']} metrics={metrics}")
        return 0 if not failures else 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cloud-edge simulation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
