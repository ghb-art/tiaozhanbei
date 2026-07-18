from __future__ import annotations

import unittest

from model_compression.build_capability_rehearsal import V22_CODE_TASKS, code_task_variant
from model_compression.generate_teacher_capability_distill import (
    CODE_VERIFIER_VERSION,
    CONTRACT_CLARIFICATIONS,
    build_repair_messages,
    run_code_equivalence,
    safe_code_ast,
    validate_code_row,
)
from scripts.evaluate_chapter2_capability import build_messages


class TeacherCapabilityDistillTests(unittest.TestCase):
    def test_code_verifier_contract_version(self) -> None:
        self.assertEqual(CODE_VERIFIER_VERSION, "1.2")
        self.assertIn("sorted and unique", CONTRACT_CLARIFICATIONS["compress_ranges"])

    def test_semantic_variant_preserves_arguments_and_docstring(self) -> None:
        prompt = (
            "def min_coins(coins: list[int], amount: int) -> int:\n"
            '    """Return the minimum coins needed for amount, or -1 when impossible."""\n'
        )
        answer = "    best = [amount + 1] * (amount + 1)\n    return best[amount]"

        variant_prompt, _ = code_task_variant(prompt, answer, 4)

        self.assertIn("(coins: list[int], amount: int)", variant_prompt)
        self.assertIn("minimum coins needed for amount", variant_prompt)

    def test_safe_ast_allows_whitelisted_standard_library(self) -> None:
        self.assertTrue(safe_code_ast("from collections import deque\ndef f():\n    return deque()\n"))
        self.assertFalse(safe_code_ast("import os\ndef f():\n    return os.getcwd()\n"))

    def test_sorted_merge_still_rejects_duplicate_loss(self) -> None:
        reference = (
            "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n"
            "    result = []\n"
            "    left = right = 0\n"
            "    while left < len(a) and right < len(b):\n"
            "        if a[left] <= b[right]:\n"
            "            result.append(a[left]); left += 1\n"
            "        else:\n"
            "            result.append(b[right]); right += 1\n"
            "    return result + a[left:] + b[right:]\n"
        )
        candidate = (
            "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n"
            "    return sorted(set(a + b))\n"
        )

        passed, detail = run_code_equivalence(
            reference,
            candidate,
            "merge_sorted_lists",
            "synthetic_code_advanced/merge_sorted_lists",
            5.0,
        )

        self.assertFalse(passed)
        self.assertIn("mismatch", detail)

    def test_two_sum_accepts_any_valid_pair(self) -> None:
        reference = (
            "def two_sum_indices(values: list[int], target: int):\n"
            "    seen = {}\n"
            "    for index, value in enumerate(values):\n"
            "        if target - value in seen:\n"
            "            return (seen[target - value], index)\n"
            "        seen[value] = index\n"
            "    return None\n"
        )
        candidate = (
            "def two_sum_indices(values: list[int], target: int):\n"
            "    for left in range(len(values)):\n"
            "        for right in range(left + 1, len(values)):\n"
            "            if values[left] + values[right] == target:\n"
            "                return (left, right)\n"
            "    return None\n"
        )

        passed, detail = run_code_equivalence(
            reference,
            candidate,
            "two_sum_indices",
            "synthetic_code_advanced/two_sum_indices",
            5.0,
        )

        self.assertTrue(passed, detail)

    def test_binary_search_accepts_any_valid_duplicate_index(self) -> None:
        reference = (
            "def binary_search(values: list[int], target: int) -> int:\n"
            "    try:\n        return values.index(target)\n    except ValueError:\n        return -1\n"
        )
        candidate = (
            "def binary_search(values: list[int], target: int) -> int:\n"
            "    for index in range(len(values) - 1, -1, -1):\n"
            "        if values[index] == target:\n            return index\n"
            "    return -1\n"
        )

        passed, detail = run_code_equivalence(
            reference,
            candidate,
            "binary_search",
            "synthetic_code_advanced/binary_search",
            5.0,
        )

        self.assertTrue(passed, detail)

    def test_repair_prompt_contains_contract_and_failure(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "Return code."},
                {
                    "role": "user",
                    "content": "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n",
                },
            ]
        }

        messages = build_repair_messages(
            row,
            "return sorted(set(a + b))",
            "expected=[1, 1] actual=[1]",
            "Merge two sorted integer lists while preserving duplicates.",
        )

        feedback = messages[-1]["content"]
        self.assertIn("preserving duplicates", feedback)
        self.assertIn("expected=[1, 1] actual=[1]", feedback)
        self.assertEqual(messages[-2]["role"], "assistant")

    def test_v22_canonical_code_families_pass_execution_verifier(self) -> None:
        self.assertEqual(len(V22_CODE_TASKS), 48)
        for sample_id, prompt, answer in V22_CODE_TASKS:
            entry_point = prompt.split("def ", 1)[1].split("(", 1)[0]
            source = f"{prompt.rstrip()}\n{answer.rstrip()}\n"
            passed, detail = run_code_equivalence(
                source,
                source,
                entry_point,
                sample_id,
                5.0,
            )
            self.assertTrue(passed, f"{sample_id}: {detail}")

    def test_mbpp_assert_verifier_uses_formal_body_protocol(self) -> None:
        prompt_source = 'def add_one(value: int) -> int:\n    "Return value plus one."\n'
        sample = {
            "dataset_key": "humaneval",
            "sample_id": "mbpp/dev_gate/0001",
            "prompt": prompt_source,
            "entry_point": "add_one",
        }
        messages, _ = build_messages(sample, "v11")
        row = {
            "dataset_key": "humaneval",
            "sample_id": sample["sample_id"],
            "validation_group_id": "mbpp/task/0001",
            "messages": messages,
            "answer": "    return value + 1",
            "code_eval": {
                "kind": "mbpp_assert_tests_v1",
                "entry_point": "add_one",
                "prompt_source": prompt_source,
                "setup_code": "",
                "tests": ["assert add_one(0) == 1", "assert add_one(-2) == -1"],
            },
        }

        passed, detail, normalized = validate_code_row(row, "return value + 1", 5.0)
        failed, _, _ = validate_code_row(row, "return value - 1", 5.0)

        self.assertTrue(passed, detail)
        self.assertEqual(normalized, "    return value + 1")
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
