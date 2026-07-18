from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "model_compression/build_p0a2_recovery_data.py", "build_p0a2_recovery_data"
)
evaluator = load_module("scripts/evaluate_p0a2_recovery.py", "evaluate_p0a2_recovery")


def read_jsonl(path: Path) -> list[dict]:
    # JSON strings may legally contain Unicode characters that ``splitlines``
    # treats as separators even though they are not JSONL record delimiters.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class P0A2RecoveryTests(unittest.TestCase):
    def test_frozen_data_gate_has_no_training_or_final_leakage(self) -> None:
        audit = json.loads(
            (ROOT / "reports/audit/gate_p0a2_recovery_data.json").read_text(encoding="utf-8")
        )
        train = read_jsonl(ROOT / audit["outputs"]["train"]["path"])
        validation = read_jsonl(ROOT / audit["outputs"]["validation"]["path"])

        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["formal_test_reference_count"], 0)
        self.assertEqual(audit["formal_humaneval_prompt_overlap_count"], 0)
        self.assertEqual(audit["train_validation_group_overlap_count"], 0)
        self.assertTrue(all(row["used_for_training"] is True for row in train))
        self.assertTrue(all(row["used_for_training"] is False for row in validation))
        self.assertTrue(all(row["used_for_final_test"] is False for row in train + validation))

    def test_data_protocol_covers_three_capabilities(self) -> None:
        audit = json.loads(
            (ROOT / "reports/audit/gate_p0a2_recovery_data.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(audit["train_dataset_counts"]), {"gsm8k", "humaneval", "cmmlu"})
        self.assertEqual(
            set(audit["validation_dataset_counts"]), {"gsm8k", "humaneval", "cmmlu"}
        )
        self.assertGreaterEqual(audit["validation_group_count"], 150)

    def test_copy_row_is_deterministic_and_preserves_code_protocol(self) -> None:
        source = {
            "dataset_key": "humaneval",
            "sample_id": "mbpp/train/0601",
            "validation_group_id": "mbpp/task/0601",
            "messages": [{"role": "user", "content": "complete"}],
            "answer": "return 1",
            "code_eval": {"kind": "mbpp_assert_tests_v1", "tests": ["assert f() == 1"]},
        }
        first = builder.copy_row(
            source,
            sample_id="p0a2/train/mbpp/train/0601/exposure/00",
            source="mbpp_official_train_p0a2",
            used_for_training=True,
        )
        second = builder.copy_row(
            source,
            sample_id="p0a2/train/mbpp/train/0601/exposure/00",
            source="mbpp_official_train_p0a2",
            used_for_training=True,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["validation_group_id"], "mbpp/task/0601")
        self.assertEqual(first["code_eval"]["kind"], "mbpp_assert_tests_v1")

    def test_evaluator_maps_validate_ranges(self) -> None:
        self.assertEqual(
            evaluator.parse_map(["gsm8k=256,cmmlu=16"], int, "tokens"),
            {"gsm8k": 256, "cmmlu": 16},
        )
        with self.assertRaises(evaluator.RecoveryEvalError):
            evaluator.parse_map(["unknown=1"], int, "tokens")

    def test_launcher_enforces_guarded_train_and_g0_reentry(self) -> None:
        launcher = (ROOT / "scripts/run_p0a2.sh").read_text(encoding="utf-8")
        self.assertIn("--validation-selection-metric generation", launcher)
        self.assertIn("--parent-preservation-weight 0.1", launcher)
        self.assertIn("--min-generation-validation-improvement 0.01", launcher)
        self.assertIn("--require-feasible", launcher)
        self.assertNotIn("HumanEval.jsonl", launcher)


if __name__ == "__main__":
    unittest.main()
