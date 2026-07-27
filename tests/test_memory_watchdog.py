from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/memory_watchdog.py"
SPEC = importlib.util.spec_from_file_location("project_memory_watchdog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


class MemoryWatchdogTests(unittest.TestCase):
    def test_parse_meminfo_uses_available_memory(self) -> None:
        memory = watchdog.parse_meminfo(
            "MemTotal:       100000 kB\n"
            "MemFree:         10000 kB\n"
            "MemAvailable:    40000 kB\n"
            "Buffers:          1000 kB\n"
            "Cached:          20000 kB\n"
        )
        self.assertEqual(memory.total_bytes, 100000 * 1024)
        self.assertEqual(memory.available_bytes, 40000 * 1024)
        self.assertAlmostEqual(memory.used_percent, 60.0)

    def test_threshold_is_inclusive_for_host_ram(self) -> None:
        snapshot = watchdog.MemorySnapshot(
            timestamp="test",
            host=watchdog.HostMemory(
                total_bytes=100,
                available_bytes=40,
                used_bytes=60,
                used_percent=60.0,
            ),
        )
        violations = watchdog.threshold_violations(snapshot, 60.0)
        self.assertEqual([item["resource"] for item in violations], ["host_ram"])

    def test_consecutive_tracker_resets_after_safe_sample(self) -> None:
        tracker = watchdog.ConsecutiveBreachTracker(required_samples=2)
        violation = [{"resource": "host_ram"}]
        self.assertFalse(tracker.update(violation))
        self.assertEqual(tracker.count, 1)
        self.assertFalse(tracker.update([]))
        self.assertEqual(tracker.count, 0)
        self.assertFalse(tracker.update(violation))
        self.assertTrue(tracker.update(violation))

    def test_wrapper_defaults_to_sixty_percent(self) -> None:
        source = (
            ROOT / "scripts/run_with_memory_guard.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('MEMORY_GUARD_THRESHOLD_PERCENT:-60', source)
        self.assertIn('"$ROOT/scripts/memory_watchdog.py" run', source)
        self.assertIn('-- "$@"', source)


if __name__ == "__main__":
    unittest.main()
