from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "model_compression/train_p0a4_lora.py"
SPEC = importlib.util.spec_from_file_location("p0a4r2_train_lora", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


class P0A4R2V1CodeTests(unittest.TestCase):
    def test_equal_source_loss_weights_preserve_rows_and_equalize_mass(self) -> None:
        rows = [
            *({"source": "mbpp"} for _ in range(2)),
            *({"source": "apps"} for _ in range(6)),
        ]
        counts, weights = trainer.equal_source_loss_weights(rows, "source")
        self.assertEqual(counts, {"mbpp": 2, "apps": 6})
        self.assertAlmostEqual(counts["mbpp"] * weights["mbpp"], 4.0)
        self.assertAlmostEqual(counts["apps"] * weights["apps"], 4.0)
        self.assertEqual(len(rows), 8)

    def test_config_freezes_nlp_but_routes_v1_nlp_to_shared(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a4r2_v1_code.json").read_text(encoding="utf-8")
        )
        code = config["training"]["code_adapter"]
        self.assertEqual(code["rank"], 4)
        self.assertEqual(code["alpha"], 4)
        self.assertEqual(code["learning_rate"], 1e-5)
        self.assertEqual(code["epochs"], 1)
        self.assertEqual(code["source_balance_key"], "source")
        self.assertFalse(config["frozen_nlp"]["deployment_on_v1"])
        self.assertEqual(config["routing"]["cmmlu"], "shared_v1")
        self.assertEqual(config["routing"]["humaneval"], "selected_code_adapter_on_v1")

    def test_launcher_never_reads_protected_item_traces(self) -> None:
        source = (ROOT / "scripts/run_p0a4r2.sh").read_text(encoding="utf-8")
        self.assertIn("--source-balance-key", source)
        self.assertIn("--external-checkpoint-selection", source)
        self.assertIn("models/checkpoints/p0a4/student-shared-merged", source)
        self.assertNotIn("reports/sealed/p0a4", source)
        self.assertNotIn("p0a4_smoke96.jsonl", source)
        self.assertNotIn("p0a2_recovery_validation.jsonl", source)


if __name__ == "__main__":
    unittest.main()
