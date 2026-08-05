from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model_compression import build_p0a6_data as data_builder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_ceval(path: Path, question: str, answer: str = "B") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("id", "question", "A", "B", "C", "D", "answer", "explanation"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": 0,
                "question": question,
                "A": "甲",
                "B": "乙",
                "C": "丙",
                "D": "丁",
                "answer": answer,
                "explanation": "乙符合题意。",
            }
        )


def training_row(sample_id: str, dataset_key: str, prompt: str, answer: str, metadata=None):
    return {
        "sample_id": sample_id,
        "dataset_key": dataset_key,
        "source": "fixture",
        "split_role": "train",
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": prompt},
        ],
        "answer": answer,
        "metadata": metadata or {},
        "distill_validation": "teacher_verified",
    }


def code_row(sample_id: str, *, two_functions: bool = False):
    answer = "def add_one(x):\n    return x + 1"
    if two_functions:
        answer += "\n\ndef helper(x):\n    return x"
    return training_row(
        sample_id,
        "opencodeinstruct",
        "Implement `add_one(x)` as one Python function.",
        answer,
        {
            "unit_tests": [
                "assert add_one(0) == 1",
                "assert add_one(2) == 3",
                "assert add_one(-2) == -1",
            ],
            "independent_execution": "passed",
        },
    )


def fixture_args(tmp_path: Path):
    return data_builder.parse_args(
        [
            "--train-source",
            str(tmp_path / "distill_train.jsonl"),
            "--validation-source",
            str(tmp_path / "internal_validation.jsonl"),
            "--ceval-root",
            str(tmp_path / "ceval"),
            "--output-dir",
            str(tmp_path / "output"),
            "--audit",
            str(tmp_path / "audit.json"),
            "--quick-per-domain",
            "1",
            "--min-math-train",
            "1",
            "--min-code-train",
            "1",
            "--min-nlp-train",
            "1",
            "--min-code-validation",
            "1",
        ]
    )


class P0A6DataTests(unittest.TestCase):
    def test_builds_weighted_train_and_endpoint_manifests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-data-test-") as directory:
            tmp_path = Path(directory)
            train_rows = [
                training_row(
                    "gsm8k/train/1",
                    "gsm8k",
                    "What is 1+1?",
                    "1+1=2\n#### 2",
                    {"reference_answer": "2"},
                ),
                code_row("code/train/1"),
                code_row("code/train/rejected", two_functions=True),
                training_row("coig/train/1", "cmmlu", "中国的首都是哪里？", "北京"),
            ]
            validation_rows = [
                training_row(
                    "gsm8k/validation/1",
                    "gsm8k",
                    "What is 2+2?",
                    "2+2=4\n#### 4",
                    {"reference_answer": "4"},
                ),
                code_row("code/validation/1"),
            ]
            write_jsonl(tmp_path / "distill_train.jsonl", train_rows)
            write_jsonl(tmp_path / "internal_validation.jsonl", validation_rows)
            write_ceval(tmp_path / "ceval/val/math_val.csv", "训练选择题")
            write_ceval(tmp_path / "ceval/dev/math_dev.csv", "验证选择题")
            # A malformed file proves that C-Eval test is never discovered/read.
            (tmp_path / "ceval/test").mkdir(parents=True)
            (tmp_path / "ceval/test/broken.json").write_text(
                "not-json", encoding="utf-8"
            )

            audit = data_builder.build_outputs(fixture_args(tmp_path))

            self.assertEqual(audit["status"], "passed")
            self.assertEqual(
                audit["counts"]["train_by_task"],
                {"code": 1, "math": 1, "nlp": 2},
            )
            self.assertEqual(
                audit["counts"]["quick_validation"],
                {"code": 1, "math": 1, "nlp": 1},
            )
            self.assertEqual(
                audit["counts"]["full_validation"],
                {"code": 1, "math": 1, "nlp": 1},
            )
            for task, expected in {"math": 0.35, "code": 0.30, "nlp": 0.35}.items():
                self.assertAlmostEqual(audit["effective_task_mass"][task], expected)
            self.assertEqual(
                audit["rejections"]["code_not_one_sync_function"], 1
            )

            train = data_builder.read_jsonl(tmp_path / "output/train.jsonl")
            self.assertEqual(
                {row["task_id"] for row in train}, {"math", "code", "nlp"}
            )
            self.assertTrue(
                all(
                    "kl_weight" in row and "answer_token_weight" in row
                    for row in train
                )
            )
            ceval = next(row for row in train if row["dataset_key"] == "ceval")
            self.assertTrue(ceval["answer"].endswith("最终答案：B"))
            self.assertEqual(ceval["answer_token_weight"], 2.0)

            quick = data_builder.read_jsonl(
                tmp_path / "output/quick_validation.jsonl"
            )
            required = {"domain", "prompt", "reference", "unit_tests", "validator"}
            self.assertTrue(all(required <= set(row) for row in quick))
            code = next(row for row in quick if row["domain"] == "code")
            self.assertEqual(len(code["unit_tests"]), 3)
            self.assertEqual(
                next(row for row in quick if row["domain"] == "nlp")["reference"],
                "B",
            )
            manifest = json.loads(
                (tmp_path / "output/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["train"]["rows"], 4)
            self.assertEqual(
                manifest["validation"]["quick"]["expected_counts"],
                {"code": 1, "math": 1, "nlp": 1},
            )
            self.assertEqual(
                manifest["validation"]["full"]["expected_counts"],
                {"code": 1, "math": 1, "nlp": 1},
            )
            self.assertEqual(
                audit["outputs"]["manifest"]["path"],
                (tmp_path / "output/manifest.json").resolve().as_posix(),
            )

    def test_missing_ceval_has_actionable_download_message(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0a6-data-test-") as directory:
            args = fixture_args(Path(directory))
            with self.assertRaises(data_builder.DataBuildError) as caught:
                data_builder.build_outputs(args)
            message = str(caught.exception)
            self.assertIn("Missing C-Eval root", message)
            self.assertIn("https://github.com/hkust-nlp/ceval.git", message)

    def test_humaneval_filter_rejects_io_and_multiple_functions(self) -> None:
        kwargs = {
            "min_unit_tests": 3,
            "max_prompt_chars": 4000,
            "max_answer_chars": 4000,
        }
        self.assertEqual(
            data_builder.humaneval_style_code_reason(code_row("valid"), **kwargs),
            "",
        )
        self.assertEqual(
            data_builder.humaneval_style_code_reason(
                code_row("two", two_functions=True), **kwargs
            ),
            "not_one_sync_function",
        )
        io_row = code_row("io")
        io_row["answer"] = "def add_one(x):\n    print(x)\n    return x + 1"
        self.assertEqual(
            data_builder.humaneval_style_code_reason(io_row, **kwargs),
            "io_or_dynamic_execution",
        )


if __name__ == "__main__":
    unittest.main()
