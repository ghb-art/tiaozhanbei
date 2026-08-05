from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_compression import generate_p0a6_mcq_rationales as generator


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ceval_row(index: int, label: str) -> dict:
    return {
        "sample_id": f"ceval/val/fixture/{index}",
        "dataset_key": "ceval",
        "source": "C-Eval-labelled",
        "split_role": "train",
        "task_id": "nlp",
        "messages": [
            {"role": "system", "content": "Answer the question."},
            {
                "role": "user",
                "content": (
                    f"问题：fixture-{index} 的正确选项是什么？\n"
                    "A. 甲\nB. 乙\nC. 丙\nD. 丁"
                ),
            },
        ],
        "answer": f"最终答案：{label}",
        "answer_token_weight": 2.0,
        "training_weight": 1.0,
        "quality_weight": 1.0,
        "kl_weight": 0.2,
        "metadata": {
            "ceval_split": "val",
            "human_labelled": True,
            "reference_answer": label,
            "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
        },
    }


def fixture_args(root: Path, *, resume: bool = False, max_attempts: int = 3):
    argv = [
        "--source",
        str(root / "train.jsonl"),
        "--output",
        str(root / "rationale_train.jsonl"),
        "--trace",
        str(root / "trace.jsonl"),
        "--audit",
        str(root / "audit.json"),
        "--expected-rows",
        "2",
        "--workers",
        "1",
        "--max-attempts",
        str(max_attempts),
    ]
    if resume:
        argv.append("--resume")
    return generator.parse_args(argv)


class P0A6McqRationaleTests(unittest.TestCase):
    def test_response_parser_locks_final_human_label(self) -> None:
        rationale, answer = generator.parse_and_validate_response(
            "边际成本等于新增一单位产量带来的成本变化。\n最终答案：B", "B"
        )
        self.assertIn("边际成本", rationale)
        self.assertTrue(answer.endswith("最终答案：B"))

        with self.assertRaisesRegex(
            generator.RationaleGenerationError, "teacher_label_mismatch"
        ):
            generator.parse_and_validate_response(
                "这个选项符合题意并且概念定义完全一致。\n最终答案：A", "B"
            )
        inline_rationale, inline_answer = generator.parse_and_validate_response(
            "这个选项符合题意并且概念定义完全一致。最终答案：B。", "B"
        )
        self.assertIn("概念定义", inline_rationale)
        self.assertTrue(inline_answer.endswith("最终答案：B"))

        five_sentence_rationale, _ = generator.parse_and_validate_response(
            "第一句说明定义。第二句排除干扰。第三句对应条件。"
            "第四句核对题干。第五句得出结论。\n**最终答案：B**",
            "B",
        )
        self.assertIn("第五句", five_sentence_rationale)

        terminology_rationale, _ = generator.parse_and_validate_response(
            "该概念描述的是测量不确定性，这与题干条件一致。\n最终答案：B",
            "B",
        )
        self.assertIn("不确定性", terminology_rationale)

        with self.assertRaisesRegex(
            generator.RationaleGenerationError, "missing_strict_final_answer"
        ):
            generator.parse_and_validate_response(
                "这是被截断的解释，没有最终的人工标签行", "B"
            )

    def test_human_label_sources_must_agree(self) -> None:
        row = ceval_row(1, "C")
        self.assertEqual(generator.extract_human_label(row), "C")
        row["metadata"]["reference_answer"] = "D"
        with self.assertRaisesRegex(
            generator.RationaleGenerationError, "Human-label disagreement"
        ):
            generator.extract_human_label(row)

    def test_retries_use_distinct_concise_prompts(self) -> None:
        row = ceval_row(1, "C")
        first = generator.build_teacher_messages(row, "C", 1)[-1]["content"]
        second = generator.build_teacher_messages(row, "C", 2)[-1]["content"]
        third = generator.build_teacher_messages(row, "C", 3)[-1]["content"]
        self.assertNotIn("120个汉字", first)
        self.assertIn("120个汉字", second)
        self.assertIn("80个汉字", third)
        self.assertNotEqual(second, third)

    def test_retry_then_builds_trainer_compatible_rows_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-rationale-test-") as directory:
            root = Path(directory)
            write_jsonl(root / "train.jsonl", [ceval_row(1, "B"), ceval_row(2, "D")])
            calls = 0

            def teacher(messages, max_tokens, timeout):
                nonlocal calls
                calls += 1
                prompt = messages[-1]["content"]
                label = "B" if "fixture-1" in prompt else "D"
                if calls == 1:
                    return "概念之间的关系能够直接排除其他选项。\n最终答案：A"
                return f"该选项符合题干所考查概念的定义与适用条件。\n最终答案：{label}"

            audit = generator.run_generation(
                fixture_args(root),
                teacher=teacher,
                served_model_id=generator.DEFAULT_MODEL_ID,
            )
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["output_rows"], 2)
            self.assertEqual(audit["retry_count"], 1)
            output = generator.read_jsonl(root / "rationale_train.jsonl")
            self.assertEqual(
                {row["dataset_key"] for row in output}, {"ceval_rationale_train"}
            )
            self.assertTrue(all(row["domain"] == "nlp" for row in output))
            self.assertTrue(all(row["answer_token_weight"] == 2.0 for row in output))
            self.assertTrue(all(row["training_weight"] == 1.0 for row in output))
            self.assertTrue(all(row["kl_weight"] == 0.2 for row in output))
            self.assertTrue(output[0]["answer"].endswith("最终答案：B"))
            self.assertTrue(output[1]["answer"].endswith("最终答案：D"))

            def must_not_call(*_args):
                raise AssertionError("A completed accepted trace must be reused")

            resumed = generator.run_generation(
                fixture_args(root, resume=True),
                teacher=must_not_call,
                served_model_id=generator.DEFAULT_MODEL_ID,
            )
            self.assertEqual(resumed["status"], "passed")
            self.assertEqual(resumed["resumed_accepted_rows"], 2)
            self.assertEqual(resumed["newly_processed_rows"], 0)

    def test_rejected_wrong_labels_are_retried_on_resume_not_used_as_targets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-rationale-reject-") as directory:
            root = Path(directory)
            rows = [ceval_row(1, "B"), ceval_row(2, "D")]
            write_jsonl(root / "train.jsonl", rows)

            def wrong_teacher(messages, max_tokens, timeout):
                return "这个结论由定义直接推导，因此无需附加条件。\n最终答案：A"

            failed = generator.run_generation(
                fixture_args(root, max_attempts=2),
                teacher=wrong_teacher,
                served_model_id=generator.DEFAULT_MODEL_ID,
            )
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(generator.read_jsonl(root / "rationale_train.jsonl"), [])
            self.assertEqual(
                {row["status"] for row in generator.read_jsonl(root / "trace.jsonl")},
                {"rejected"},
            )

            def correct_teacher(messages, max_tokens, timeout):
                prompt = messages[-1]["content"]
                label = "B" if "fixture-1" in prompt else "D"
                return f"该选项与题干限定条件和相应定义完全一致。\n最终答案：{label}"

            passed = generator.run_generation(
                fixture_args(root, resume=True),
                teacher=correct_teacher,
                served_model_id=generator.DEFAULT_MODEL_ID,
            )
            self.assertEqual(passed["status"], "passed")
            self.assertEqual(passed["resumed_accepted_rows"], 0)
            self.assertEqual(passed["output_rows"], 2)


if __name__ == "__main__":
    unittest.main()
