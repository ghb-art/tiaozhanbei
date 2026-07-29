from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("p0a5_builder", "model_compression/build_p0a5_data.py")
gate = load_module("p0a5_gate", "scripts/p0a5_gate.py")


class P0A5ProtocolTests(unittest.TestCase):
    def test_config_has_one_gate_and_no_retired_sources(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a5_capability.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["gate300"]["expected_counts"],
            {"math": 100, "code": 100, "nlp": 100},
        )
        self.assertEqual(config["gate300"]["initial_ratio"], 0.78)
        self.assertEqual(config["gate300"]["recommended_full_ratio"], 0.82)
        self.assertEqual(
            sum(config["datasets"]["nlp"]["category_train_quotas"].values()),
            config["datasets"]["nlp"]["train_rows"],
        )
        self.assertEqual(config["datasets"]["nlp"]["train_rows"], 9500)
        serialized = json.dumps(config, ensure_ascii=False).casefold()
        for retired in ("mbpp", "mmlu_aux", "smoke96", "selection170", "code_contests"):
            self.assertNotIn(retired, serialized)

    def test_nlp_category_mapping_is_multidomain(self) -> None:
        exam = {
            "task_type": {"major": ["试题"]},
            "domain": ["历史"],
        }
        self.assertIn("exam", builder.nlp_eligible_categories(exam))
        self.assertIn(
            "humanities_social_science", builder.nlp_eligible_categories(exam)
        )
        logic = {
            "task_type": {"major": ["自然语言推理"]},
            "domain": ["通用"],
        }
        self.assertIn("language_reasoning", builder.nlp_eligible_categories(logic))

    def test_task_weights_produce_requested_loss_mass(self) -> None:
        rows = [
            *[
                {"dataset_key": "gsm8k", "sample_id": f"m{index}"}
                for index in range(3)
            ],
            *[
                {"dataset_key": "opencodeinstruct", "sample_id": f"c{index}"}
                for index in range(7)
            ],
            *[
                {"dataset_key": "cmmlu", "sample_id": f"n{index}"}
                for index in range(5)
            ],
        ]
        target = {"gsm8k": 0.3, "opencodeinstruct": 0.35, "cmmlu": 0.35}
        builder.add_task_weights(rows, target)
        mass: Counter[str] = Counter()
        for row in rows:
            mass[row["dataset_key"]] += row["training_weight"]
        total = sum(mass.values())
        for task, expected in target.items():
            self.assertAlmostEqual(mass[task] / total, expected)
        self.assertTrue(all(row["preserve_math"] for row in rows[:3]))
        self.assertFalse(any(row["preserve_math"] for row in rows[3:]))

    def test_gate_trace_loader_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            row = {
                "sample_id": "x",
                "domain": "math",
                "prompt_hash": "h",
                "correct": True,
                "generation_error": "",
            }
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(gate.GateError):
                gate.read_trace(path, "test")

    def test_runbook_exposes_cpu_stop_point(self) -> None:
        launcher = (ROOT / "scripts/run_p0a5.sh").read_text(encoding="utf-8")
        self.assertIn("CPU preflight complete. No GPU command was started.", launcher)
        self.assertIn("teacher-train", launcher)
        self.assertNotIn("smoke96", launcher)
        self.assertNotIn("selection170", launcher)


if __name__ == "__main__":
    unittest.main()
