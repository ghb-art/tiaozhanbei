from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "model_compression" / "assemble_p0a4r4_shared_data.py"
SPEC = importlib.util.spec_from_file_location("p0a4r4_assembly", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
assembly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assembly
SPEC.loader.exec_module(assembly)


class P0A4R4LongCodeTests(unittest.TestCase):
    def test_code_sources_receive_equal_loss_mass_without_duplication(self) -> None:
        rows = [
            {
                "dataset_key": "gsm8k",
                "validation_group_id": "math/1",
            },
            *(
                {
                    "dataset_key": "humaneval",
                    "origin": "mbpp",
                    "validation_group_id": f"mbpp/{index}",
                }
                for index in range(2)
            ),
            *(
                {
                    "dataset_key": "humaneval",
                    "origin": "apps",
                    "validation_group_id": f"apps/{index}",
                }
                for index in range(6)
            ),
            *(
                {
                    "dataset_key": "humaneval",
                    "origin": "code_contests",
                    "validation_group_id": f"contest/{index}",
                }
                for index in range(4)
            ),
            {
                "dataset_key": "cmmlu",
                "validation_group_id": "nlp/1",
            },
        ]
        counts, weights, mass = assembly.assign_training_weights(
            rows, "training_weight"
        )
        self.assertEqual(
            counts, {"mbpp": 2, "apps": 6, "code_contests": 4}
        )
        self.assertAlmostEqual(mass["mbpp"], 4.0)
        self.assertAlmostEqual(mass["apps"], 4.0)
        self.assertAlmostEqual(mass["code_contests"], 4.0)
        self.assertEqual(rows[0]["training_weight"], 1.0)
        self.assertEqual(rows[-1]["training_weight"], 1.0)
        self.assertEqual(len(rows), 14)
        self.assertGreater(weights["mbpp"], weights["apps"])

    def test_protocol_preregisters_compact_code_route(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a4r4_long_code_distillation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["policy"]["max_preregistered_candidates"], 2)
        self.assertTrue(
            config["policy"]["predecessor_candidate_limit_exhausted"]
        )
        self.assertEqual(
            config["data"]["code"]["solution_selection"],
            "shortest_tokens",
        )
        self.assertEqual(config["data"]["code"]["max_answer_tokens"], 768)
        self.assertEqual(
            config["evaluation"]["max_new_tokens"]["humaneval"], 768
        )
        self.assertEqual(config["data"]["sample_weight_key"], "training_weight")

    def test_launcher_builds_fresh_validation_before_training(self) -> None:
        source = (ROOT / "scripts/run_p0a4r4.sh").read_text(encoding="utf-8")
        self.assertIn("--solution-selection shortest_tokens", source)
        self.assertIn("--max-answer-tokens 768", source)
        self.assertIn("--sample-weight-key training_weight", source)
        self.assertIn(
            "--exclude-jsonl data/distill/p0a4r3_code_contests_train.jsonl",
            source,
        )
        self.assertNotIn("reports/sealed/p0a4", source)


if __name__ == "__main__":
    unittest.main()
