from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grpo = load_module("p0b2_grpo", "model_compression/train_p0b2_code_grpo.py")


CODE_ROW = {
    "sample_id": "opencodeinstruct/train/test",
    "domain": "code",
    "messages": [
        {"role": "system", "content": "Return only the function body."},
        {
            "role": "user",
            "content": (
                "Complete the Python function correctly. Return only the function body.\n\n"
                "def add(a, b):\n"
                '    """Adds two numbers."""\n'
            ),
        },
    ],
    "metadata": {
        "entry_point": "add",
        "prompt_source": "def add(a, b):\n    \"\"\"Adds two numbers.\"\"\"\n",
        "unit_tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    },
}


class P0B2CodeGrpoTests(unittest.TestCase):
    def test_normalize_body_strips_think_and_dedents(self) -> None:
        self.assertEqual(
            grpo.normalize_body("<think>\n\n</think>\n\n    return a + b"),
            "return a + b",
        )

    def test_wrap_body_uses_signature_and_dedents(self) -> None:
        source = grpo.wrap_body(
            CODE_ROW, "<think>\n\n</think>\n\n    return a + b"
        )
        self.assertEqual(source, "def add(a, b):\n    return a + b")

    def test_reward_correct_body_passes(self) -> None:
        reward, detail = grpo.reward_rollout(CODE_ROW, "return a + b", 5)
        self.assertEqual(reward, 1.0, detail)

    def test_reward_wrong_and_invalid_bodies_fail(self) -> None:
        reward, detail = grpo.reward_rollout(CODE_ROW, "return a - b", 5)
        self.assertEqual(reward, 0.0, detail)
        reward, detail = grpo.reward_rollout(CODE_ROW, "return a +", 5)
        self.assertEqual(reward, 0.0, detail)
        self.assertTrue(detail in ("unsafe_or_invalid_python", "SyntaxError: invalid syntax"))

    def test_group_advantages_normalize_within_group(self) -> None:
        advantages = grpo.group_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[1], 0.0)

    def test_group_advantages_all_equal_are_zero(self) -> None:
        self.assertEqual(grpo.group_advantages([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_config_and_data_audit_are_preregistered(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0b2_code_grpo.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["policy"]["humaneval_used"], False)
        self.assertEqual(config["policy"]["pre_registered_candidate_count"], 1)
        audit = json.loads(
            (ROOT / "reports/audit/gate_p0b2_code_data.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["train_rows"], 16000)
        self.assertEqual(audit["gate_internal_overlap"], 0)
        self.assertEqual(audit["formal_set_references"], 0)


if __name__ == "__main__":
    unittest.main()
