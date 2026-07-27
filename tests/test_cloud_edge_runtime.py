from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cloud_edge.runtime import (
    CloudEdgeRuntime,
    ConflictArbitrator,
    Decision,
    FastPath,
    NetworkState,
    Scheduler,
    TaskRequest,
    WeakNetworkOutbox,
)


def request(scenario: str = "industrial", **payload):
    return TaskRequest(
        request_id="req-1",
        event_id="event-1",
        node_id="edge-1",
        scenario=scenario,
        deadline_ms=200,
        complexity=0.8,
        confidence=0.96,
        risk=0.9,
        payload=payload,
    )


class CloudEdgeRuntimeTests(unittest.TestCase):
    def test_industrial_emergency_uses_fast_path(self) -> None:
        decision = FastPath().decide(
            request(temperature_c=95, vibration_mm_s=2, defect_score=0.1)
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.action, "emergency_stop")
        self.assertEqual(decision.route, "edge_fast_path")
        self.assertLess(decision.latency_ms, 20)

    def test_weak_network_forces_edge_model(self) -> None:
        route = Scheduler().choose(
            request(temperature_c=50, vibration_mm_s=2, defect_score=0.2),
            NetworkState(False, 500, 1.0, 0.0),
        )
        self.assertEqual(route.path, "edge_model")

    def test_outbox_is_idempotent_and_flushes_after_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = WeakNetworkOutbox(Path(directory) / "outbox.jsonl")
            decision = Decision(
                request_id="r",
                event_id="e",
                node_id="n",
                scenario="traffic",
                action="extend_green",
                confidence=0.9,
                risk=0.5,
                route="edge_model",
                latency_ms=100,
                autonomous=True,
                created_ts="2026-01-01T00:00:00+00:00",
            )
            outbox.enqueue(decision)
            outbox.enqueue(decision)
            self.assertEqual(outbox.pending_count, 1)
            self.assertEqual(
                outbox.flush(NetworkState(True, 10, 0, 100), lambda _: True),
                1,
            )
            self.assertEqual(outbox.pending_count, 0)
            recovered = WeakNetworkOutbox(Path(directory) / "outbox.jsonl")
            self.assertEqual(recovered.pending_count, 0)

    def test_outbox_recovers_pending_record_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.jsonl"
            first = WeakNetworkOutbox(path)
            decision = Decision(
                request_id="r",
                event_id="e",
                node_id="n",
                scenario="industrial",
                action="slow_and_inspect",
                confidence=0.8,
                risk=0.7,
                route="edge_model",
                latency_ms=100,
                autonomous=True,
                created_ts="2026-01-01T00:00:00+00:00",
            )
            first.enqueue(decision)
            recovered = WeakNetworkOutbox(path)
            self.assertEqual(recovered.pending_count, 1)
            self.assertEqual(
                recovered.flush(NetworkState(True, 10, 0, 100), lambda _: True),
                1,
            )

    def test_conflict_arbitration_prefers_safer_action(self) -> None:
        base = dict(
            request_id="r",
            event_id="e",
            scenario="industrial",
            confidence=0.9,
            risk=0.8,
            route="edge_model",
            latency_ms=100,
            autonomous=True,
            created_ts="2026-01-01T00:00:00+00:00",
        )
        continue_decision = Decision(
            node_id="edge-1", action="continue", **base
        )
        stop_decision = Decision(
            node_id="edge-2", action="emergency_stop", **base
        )
        winner, had_conflict = ConflictArbitrator().resolve(
            [continue_decision, stop_decision]
        )
        self.assertTrue(had_conflict)
        self.assertEqual(winner.action, "emergency_stop")

    def test_cloud_failure_falls_back_to_edge(self) -> None:
        runtime = CloudEdgeRuntime()
        network = NetworkState(True, 10, 0.0, 100)
        result = runtime.process(
            TaskRequest(
                request_id="r",
                event_id="e",
                node_id="n",
                scenario="industrial",
                deadline_ms=200,
                complexity=0.9,
                confidence=0.5,
                risk=0.4,
                payload={"temperature_c": 50, "vibration_mm_s": 2, "defect_score": 0.2},
            ),
            network,
            lambda _: ("continue", 0.8),
            lambda _: (_ for _ in ()).throw(RuntimeError("cloud down")),
        )
        self.assertEqual(result.route, "edge_model")
        self.assertTrue(result.autonomous)


if __name__ == "__main__":
    unittest.main()
