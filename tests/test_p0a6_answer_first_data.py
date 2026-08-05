from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model_compression import build_p0a6_answer_first_data as builder


def ceval_row(index: int, label: str) -> dict:
    return {
        "sample_id": f"ceval_rationale_train/val/fixture/{index}",
        "dataset_key": "ceval_rationale_train",
        "domain": "nlp",
        "split_role": "train",
        "messages": [
            {"role": "system", "content": "old prompt"},
            {
                "role": "user",
                "content": f"问题：fixture-{index}\nA. 甲\nB. 乙\nC. 丙\nD. 丁",
            },
        ],
        "answer": (
            "简短理由：第一句给出知识依据。第二句对应题干条件。第三句不应进入短理由。"
            f"\n最终答案：{label}"
        ),
        "metadata": {"human_labelled": True, "reference_answer": label},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_cmmlu(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "Question", "A", "B", "C", "D", "Answer"])
        writer.writerow([0, "不同于C-Eval的题目一", "甲", "乙", "丙", "丁", "B"])
        writer.writerow([1, "不同于C-Eval的题目二", "春", "夏", "秋", "冬", "D"])


class P0A6AnswerFirstDataTests(unittest.TestCase):
    def test_compact_rationale_keeps_two_sentences_and_locked_label(self) -> None:
        rationale, label = builder.compact_rationale(
            "简短理由：第一句说明定义。第二句联系条件。第三句应被删除。\n最终答案：C"
        )
        self.assertEqual(label, "C")
        self.assertIn("第一句", rationale)
        self.assertIn("第二句", rationale)
        self.assertNotIn("第三句", rationale)

    def test_build_combines_only_labelled_ceval_and_cmmlu_dev(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-answer-first-") as directory:
            root = Path(directory)
            ceval = root / "rationales.jsonl"
            dev = root / "cmmlu" / "dev"
            output = root / "answer_first.jsonl"
            audit = root / "audit.json"
            write_jsonl(ceval, [ceval_row(1, "A"), ceval_row(2, "C")])
            write_cmmlu(dev / "fixture.csv")
            args = builder.parse_args(
                [
                    "--ceval-rationales", str(ceval),
                    "--cmmlu-dev", str(dev),
                    "--output", str(output),
                    "--audit", str(audit),
                    "--expected-ceval-rows", "2",
                    "--expected-cmmlu-rows", "2",
                    "--expected-cmmlu-subjects", "1",
                ]
            )
            report = builder.run(args)
            rows = builder.read_jsonl(output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["cmmlu_test_files_opened"], 0)
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["dataset_key"] for row in rows},
                {"ceval_answer_first_train", "cmmlu_dev_answer_first_train"},
            )
            self.assertTrue(all(row["answer_token_position"] == "first" for row in rows))
            self.assertTrue(all(row["messages"][0]["content"] == builder.NLP_SYSTEM_PROMPT for row in rows))
            self.assertTrue(all(row["answer"].splitlines()[0].startswith("答案：") for row in rows))
            self.assertTrue(all(row["answer"].splitlines()[-1].startswith("最终答案：") for row in rows))

    def test_cmmlu_source_must_be_explicit_dev_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-answer-first-guard-") as directory:
            root = Path(directory)
            test_root = root / "cmmlu" / "test"
            write_cmmlu(test_root / "fixture.csv")
            with self.assertRaisesRegex(builder.AnswerFirstDataError, "explicit dev"):
                builder.build_cmmlu_dev_rows(test_root, 2, 1)


if __name__ == "__main__":
    unittest.main()
