from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPRESSION = ROOT / "model_compression"
if str(MODEL_COMPRESSION) not in sys.path:
    sys.path.insert(0, str(MODEL_COMPRESSION))

from train_cedd_repair import (  # noqa: E402
    cap_examples_per_validation_group,
    load_repair_examples,
    planned_optimizer_steps,
    select_generation_validation_examples,
    split_grouped_validation,
    validation_group_ids,
)


def example(group: str, index: int) -> dict[str, object]:
    return {
        "dataset_key": "humaneval",
        "sample_id": f"synthetic/{group}/{index}",
        "validation_group_id": group,
        "messages": [{"role": "user", "content": f"task {group}"}],
        "answer": f"return {index}",
    }


class CeddTrainingSelectionTests(unittest.TestCase):
    def test_group_cap_balances_accepted_variants(self) -> None:
        rows = [example("large", index) for index in range(12)]
        rows.extend(example("small", index) for index in range(3))

        selected = cap_examples_per_validation_group(rows, 4, 42)

        counts = {
            group: sum(row["validation_group_id"] == group for row in selected)
            for group in ("large", "small")
        }
        self.assertEqual(counts, {"large": 4, "small": 3})

    def test_grouped_split_has_no_family_overlap(self) -> None:
        rows = [example(f"group-{group}", index) for group in range(20) for index in range(3)]

        train_rows, validation_rows = split_grouped_validation(rows, 0.25, 20260715)

        self.assertTrue(train_rows)
        self.assertTrue(validation_rows)
        self.assertFalse(validation_group_ids(train_rows) & validation_group_ids(validation_rows))

    def test_generation_selection_balances_multiple_examples_per_group(self) -> None:
        rows = [example(group, index) for group in ("a", "b", "c") for index in range(4)]

        selected = select_generation_validation_examples(rows, 6, 42, examples_per_group=2)

        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["validation_group_id"] for row in selected[:3]}), 3)
        self.assertEqual(
            {group: sum(row["validation_group_id"] == group for row in selected) for group in ("a", "b", "c")},
            {"a": 2, "b": 2, "c": 2},
        )

    def test_planned_optimizer_steps_matches_four_gpu_schedule(self) -> None:
        self.assertEqual(planned_optimizer_steps(588, 4, 2, 2, 6, 320), 222)

    def test_disabled_repair_source_does_not_require_a_file(self) -> None:
        rows = load_repair_examples(ROOT / "does-not-exist.jsonl", None, 0, 42, 0)

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
