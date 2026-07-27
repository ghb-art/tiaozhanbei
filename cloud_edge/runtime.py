from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class NetworkState:
    available: bool
    rtt_ms: float
    loss_rate: float
    bandwidth_mbps: float

    @property
    def weak(self) -> bool:
        return (
            not self.available
            or self.rtt_ms >= 100.0
            or self.loss_rate >= 0.1
            or self.bandwidth_mbps <= 2.0
        )


@dataclass(frozen=True)
class TaskRequest:
    request_id: str
    event_id: str
    node_id: str
    scenario: str
    deadline_ms: float
    complexity: float
    confidence: float
    risk: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class Route:
    path: str
    reason: str
    estimated_latency_ms: float


@dataclass(frozen=True)
class Decision:
    request_id: str
    event_id: str
    node_id: str
    scenario: str
    action: str
    confidence: float
    risk: float
    route: str
    latency_ms: float
    autonomous: bool
    created_ts: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FastPath:
    """Deterministic millisecond path for obvious industrial/traffic states."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        values = config or {}
        industrial = values.get("industrial", {})
        traffic = values.get("traffic", {})
        self.emergency_temperature_c = float(
            industrial.get("emergency_temperature_c", 90.0)
        )
        self.emergency_vibration_mm_s = float(
            industrial.get("emergency_vibration_mm_s", 12.0)
        )
        self.emergency_defect_score = float(
            industrial.get("emergency_defect_score", 0.95)
        )
        self.congestion_queue_length = float(
            traffic.get("congestion_queue_length", 35.0)
        )

    def decide(self, request: TaskRequest) -> Decision | None:
        payload = request.payload
        action = ""
        if request.scenario == "industrial":
            temperature = float(payload.get("temperature_c", 0.0))
            vibration = float(payload.get("vibration_mm_s", 0.0))
            defect_score = float(payload.get("defect_score", 0.0))
            if (
                temperature >= self.emergency_temperature_c
                or vibration >= self.emergency_vibration_mm_s
                or defect_score >= self.emergency_defect_score
            ):
                action = "emergency_stop"
            elif temperature >= 80.0 or vibration >= 8.0 or defect_score >= 0.85:
                action = "slow_and_inspect"
            elif request.confidence >= 0.95 and request.risk <= 0.2:
                action = "continue"
        elif request.scenario == "traffic":
            emergency = bool(payload.get("emergency_vehicle", False))
            queue = float(payload.get("queue_length", 0.0))
            pedestrian = bool(payload.get("pedestrian_conflict", False))
            if emergency:
                action = "emergency_preemption"
            elif pedestrian and request.confidence >= 0.9:
                action = "hold_vehicle_phase"
            elif queue >= self.congestion_queue_length and request.confidence >= 0.92:
                action = "extend_green"
            elif queue <= 3 and request.confidence >= 0.97:
                action = "normal_cycle"
        if not action:
            return None
        return Decision(
            request_id=request.request_id,
            event_id=request.event_id,
            node_id=request.node_id,
            scenario=request.scenario,
            action=action,
            confidence=request.confidence,
            risk=request.risk,
            route="edge_fast_path",
            latency_ms=8.0,
            autonomous=True,
            created_ts=utc_now(),
        )


class Scheduler:
    def __init__(
        self,
        edge_model_latency_ms: float = 115.0,
        cloud_model_latency_ms: float = 70.0,
    ) -> None:
        self.edge_model_latency_ms = edge_model_latency_ms
        self.cloud_model_latency_ms = cloud_model_latency_ms

    def choose(self, request: TaskRequest, network: NetworkState) -> Route:
        if network.weak:
            return Route(
                "edge_model",
                "weak_or_offline_network",
                min(self.edge_model_latency_ms, request.deadline_ms),
            )
        cloud_estimate = self.cloud_model_latency_ms + 2.0 * network.rtt_ms
        needs_global = request.complexity >= 0.65 or request.confidence < 0.75
        if needs_global and cloud_estimate <= request.deadline_ms:
            return Route("cloud_model", "complex_global_task", cloud_estimate)
        return Route(
            "edge_model",
            "deadline_or_local_task",
            min(self.edge_model_latency_ms, request.deadline_ms),
        )


class WeakNetworkOutbox:
    """Idempotent local decision queue flushed after connectivity recovers."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._pending: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = str(record.get("outbox_id", ""))
                    if not key:
                        continue
                    if record.get("status") == "sent":
                        self._pending.pop(key, None)
                    elif record.get("status") == "pending":
                        self._pending[key] = record

    @staticmethod
    def key(decision: Decision) -> str:
        material = f"{decision.event_id}:{decision.node_id}:{decision.action}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def enqueue(self, decision: Decision) -> None:
        key = self.key(decision)
        if key in self._pending:
            return
        record = {"outbox_id": key, "decision": asdict(decision), "status": "pending"}
        self._pending[key] = record
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )

    def flush(
        self,
        network: NetworkState,
        sender: Callable[[dict[str, Any]], bool],
    ) -> int:
        if network.weak:
            return 0
        sent = 0
        for key, record in list(self._pending.items()):
            if sender(record):
                del self._pending[key]
                if self.path is not None:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"outbox_id": key, "status": "sent"},
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                sent += 1
        return sent

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class ConflictArbitrator:
    ACTION_SAFETY = {
        "emergency_stop": 100,
        "emergency_preemption": 95,
        "hold_vehicle_phase": 80,
        "slow_and_inspect": 75,
        "extend_green": 55,
        "continue": 20,
        "normal_cycle": 10,
    }

    def resolve(self, decisions: Iterable[Decision]) -> tuple[Decision, bool]:
        values = list(decisions)
        if not values:
            raise ValueError("Cannot arbitrate an empty decision group")
        actions = {decision.action for decision in values}
        winner = max(
            values,
            key=lambda decision: (
                self.ACTION_SAFETY.get(decision.action, 0),
                decision.risk,
                decision.confidence,
                decision.created_ts,
                decision.node_id,
            ),
        )
        return winner, len(actions) > 1


class CloudEdgeRuntime:
    def __init__(
        self,
        fast_path: FastPath | None = None,
        scheduler: Scheduler | None = None,
        outbox: WeakNetworkOutbox | None = None,
    ) -> None:
        self.fast_path = fast_path or FastPath()
        self.scheduler = scheduler or Scheduler()
        self.outbox = outbox or WeakNetworkOutbox()

    def process(
        self,
        request: TaskRequest,
        network: NetworkState,
        edge_model: Callable[[TaskRequest], tuple[str, float]],
        cloud_model: Callable[[TaskRequest], tuple[str, float]],
    ) -> Decision:
        decision = self.fast_path.decide(request)
        if decision is None:
            route = self.scheduler.choose(request, network)
            autonomous = route.path == "edge_model"
            if route.path == "cloud_model":
                try:
                    action, confidence = cloud_model(request)
                except Exception:
                    action, confidence = edge_model(request)
                    route = Route("edge_model", "cloud_failure_fallback", 125.0)
                    autonomous = True
            else:
                action, confidence = edge_model(request)
            decision = Decision(
                request_id=request.request_id,
                event_id=request.event_id,
                node_id=request.node_id,
                scenario=request.scenario,
                action=action,
                confidence=float(confidence),
                risk=request.risk,
                route=route.path,
                latency_ms=route.estimated_latency_ms,
                autonomous=autonomous,
                created_ts=utc_now(),
            )
        if network.weak:
            self.outbox.enqueue(decision)
        return decision
