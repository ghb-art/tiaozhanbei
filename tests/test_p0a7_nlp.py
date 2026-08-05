from __future__ import annotations

import unittest

from model_compression import build_p0a7_mmlu_specialist_data as builder


def verified_row(training: bool = True) -> dict:
    return {
        "sample_id": "p0a7/nlp/arc_easy/example",
        "validation_group_id": "mmlu-aux/example",
        "origin": "mmlu_auxiliary_train",
        "messages": [
            {"role": "system", "content": "source"},
            {
                "role": "user",
                "content": (
                    "以下是单项选择题。\n\n题目: Which?\n"
                    "A. 甲\nB. 乙\nC. 丙\nD. 丁\n\n"
                    "请给出一条简短理由，最后一行严格写成 FINAL: X。"
                ),
            },
        ],
        "answer": "理由：乙符合题意。\nFINAL: B",
        "teacher_verification": {"model_id": "teacher"},
        "used_for_training": training,
        "used_for_validation": not training,
        "used_for_final_test": False,
    }


class P0A7NlpDataTests(unittest.TestCase):
    def test_training_row_matches_evaluator_prompt_and_label(self) -> None:
        row = builder.convert_row(verified_row(True), training=True)
        self.assertEqual(row["dataset_key"], "mmlu_aux_chinese")
        self.assertEqual(row["messages"][0]["content"], builder.SYSTEM_PROMPT)
        self.assertTrue(row["messages"][1]["content"].startswith("问题："))
        self.assertEqual(row["answer_letter"], "B")
        self.assertTrue(row["answer"].endswith("最终答案：B"))
        self.assertEqual(row["kl_weight"], 0.05)

    def test_validation_row_is_not_training_schema(self) -> None:
        row = builder.convert_row(verified_row(False), training=False)
        self.assertEqual(row["split_role"], "p0a7_internal_validation")
        self.assertEqual(row["reference"], "B")
        self.assertNotIn("answer", row)

    def test_rejects_non_auxiliary_origin(self) -> None:
        row = verified_row(True)
        row["origin"] = "mmlu_test"
        with self.assertRaises(builder.DataError):
            builder.convert_row(row, training=True)


if __name__ == "__main__":
    unittest.main()
