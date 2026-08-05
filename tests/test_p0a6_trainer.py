from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    path = ROOT / "model_compression/train_p0a6_student.py"
    spec = importlib.util.spec_from_file_location("p0a6_student_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()


class FakeTokenizer:
    pad_token_id = 0
    all_special_ids = [1]

    @staticmethod
    def apply_chat_template(messages, *, add_generation_prompt, **kwargs):
        del kwargs
        prompt = "|".join(
            item["content"] for item in messages if item["role"] != "assistant"
        ) + "<assistant>"
        if add_generation_prompt:
            text = prompt
        else:
            answer = next(
                item["content"] for item in messages if item["role"] == "assistant"
            )
            text = prompt + answer
        return [ord(character) + 10 for character in text] + ([1] if not add_generation_prompt else [])

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        del skip_special_tokens
        return "".join(chr(token_id - 10) for token_id in token_ids if token_id != 1)

    @staticmethod
    def encode(text, *, add_special_tokens):
        del add_special_tokens
        return [ord(character) + 10 for character in text]


def valid_row(sample_id: str, domain: str) -> dict:
    answer = "简短理由。\n最终答案：C" if domain == "nlp" else "answer"
    return {
        "sample_id": sample_id,
        "domain": domain,
        "dataset_key": {"math": "gsm8k", "code": "opencodeinstruct", "nlp": "ceval"}[
            domain
        ],
        "source": "train-only",
        "split_role": "train",
        "messages": [{"role": "user", "content": "question"}],
        "answer": answer,
        "kl_weight": {"math": 0.6, "code": 0.15, "nlp": 0.2}[domain],
        "answer_token_weight": 2.0 if domain == "nlp" else 1.0,
        "training_weight": 1.0,
    }


class P0A6TrainerTests(unittest.TestCase):
    def test_fixed_training_protocol(self) -> None:
        self.assertEqual(trainer.LORA_RANK, 16)
        self.assertEqual(trainer.LORA_ALPHA, 32)
        self.assertEqual(trainer.LORA_DROPOUT, 0.05)
        self.assertEqual(trainer.LEARNING_RATE, 2e-6)
        self.assertEqual(trainer.KL_TEMPERATURE, 2.0)
        self.assertEqual(trainer.MAX_SEQUENCE_LENGTH, 1536)
        self.assertEqual(trainer.DEFAULT_MAX_STEPS, 200)
        self.assertEqual(trainer.CHECKPOINT_STEPS, 100)
        self.assertEqual(trainer.MAX_GRAD_NORM, 1.0)
        self.assertEqual(trainer.validate_batch_layout(4, 1, 8), 32)
        with self.assertRaises(trainer.TrainingError):
            trainer.validate_batch_layout(1, 1, 8)

    def test_choice_answer_token_gets_extra_weight(self) -> None:
        row = valid_row("nlp/train/1", "nlp")
        rendered = trainer.render(FakeTokenizer(), row, 1536)
        weighted = [
            index
            for index, weight in enumerate(rendered["token_weights"])
            if weight == 2.0
        ]
        self.assertEqual(len(weighted), 1)
        self.assertEqual(
            FakeTokenizer.decode(
                [rendered["input_ids"][weighted[0]]], skip_special_tokens=True
            ),
            "C",
        )
        self.assertNotEqual(rendered["labels"][weighted[0]], -100)

    def test_math_final_value_tokens_get_extra_weight(self) -> None:
        row = valid_row("gsm8k/train/weighted", "math")
        row["answer"] = "简短计算。\n#### 42"
        row["answer_token_weight"] = 4.0
        row["answer_value"] = "42"
        rendered = trainer.render(FakeTokenizer(), row, 1536)
        weighted = [
            index
            for index, weight in enumerate(rendered["token_weights"])
            if weight == 4.0
        ]
        self.assertEqual(
            FakeTokenizer.decode(
                [rendered["input_ids"][index] for index in weighted],
                skip_special_tokens=True,
            ).strip(),
            "42",
        )
        self.assertTrue(all(rendered["labels"][index] != -100 for index in weighted))

    def test_answer_first_choice_token_gets_weight_not_repeated_final_token(self) -> None:
        row = valid_row("answer-first/train/1", "nlp")
        row["dataset_key"] = "ceval_answer_first_train"
        row["answer"] = "答案：C\n简短理由：该选项符合题干条件。\n最终答案：C"
        row["answer_token_position"] = "first"
        rendered = trainer.render(FakeTokenizer(), row, 1536)
        weighted = [
            index
            for index, weight in enumerate(rendered["token_weights"])
            if weight == 2.0
        ]
        self.assertEqual(len(weighted), 1)
        answer_text = FakeTokenizer.decode(
            rendered["input_ids"][weighted[0] :], skip_special_tokens=True
        )
        self.assertTrue(answer_text.startswith("C\n简短理由"))

    def test_nlp_focus_normalizes_weights_and_multiplies_only_mcq_rows(self) -> None:
        math_row = valid_row("gsm8k/train/1", "math")
        code_row = valid_row("opencode/train/1", "code")
        nlp_mcq = valid_row("ceval/train/1", "nlp")
        nlp_open = valid_row("coig/train/1", "nlp")
        nlp_mcq["training_weight"] = 2.0
        nlp_open["training_weight"] = 4.0
        nlp_open["answer_token_weight"] = 1.0
        selected, stats = trainer.prepare_training_rows(
            [math_row, code_row, nlp_mcq, nlp_open],
            "nlp",
            2.0,
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["domain"] == "nlp" for row in selected))
        self.assertAlmostEqual(
            sum(row["training_weight"] for row in selected) / len(selected), 1.0
        )
        self.assertAlmostEqual(selected[0]["training_weight"], 2.0 / 3.0)
        self.assertAlmostEqual(selected[1]["training_weight"], 4.0 / 3.0)
        self.assertEqual(selected[0]["answer_token_weight"], 4.0)
        self.assertEqual(selected[1]["answer_token_weight"], 1.0)
        self.assertEqual(stats["source_rows"], 4)
        self.assertEqual(stats["selected_rows"], 2)
        self.assertEqual(stats["mcq_answer_token_weight_multiplied_rows"], 1)
        # The source rows remain frozen for another focus view or audit.
        self.assertEqual(nlp_mcq["training_weight"], 2.0)
        self.assertEqual(nlp_mcq["answer_token_weight"], 2.0)

    def test_nlp_mcq_focus_excludes_open_qa_and_normalizes_weights(self) -> None:
        math_row = valid_row("gsm8k/train/1", "math")
        code_row = valid_row("opencode/train/1", "code")
        nlp_mcq = valid_row("ceval/train/1", "nlp")
        nlp_open = valid_row("coig/train/1", "nlp")
        nlp_mcq["training_weight"] = 3.0
        nlp_open["dataset_key"] = "coig_cqia"
        nlp_open["answer_token_weight"] = 1.0
        selected, stats = trainer.prepare_training_rows(
            [math_row, code_row, nlp_mcq, nlp_open],
            "nlp_mcq",
            4.0,
        )
        self.assertEqual([row["sample_id"] for row in selected], ["ceval/train/1"])
        self.assertEqual(selected[0]["training_weight"], 1.0)
        self.assertEqual(selected[0]["answer_token_weight"], 8.0)
        self.assertEqual(stats["selected_dataset_counts"], {"ceval": 1})
        self.assertEqual(stats["mcq_answer_token_weight_multiplied_rows"], 1)

    def test_all_focus_preserves_training_weights(self) -> None:
        rows = [
            valid_row("gsm8k/train/1", "math"),
            valid_row("opencode/train/1", "code"),
            valid_row("ceval/train/1", "nlp"),
        ]
        rows[0]["training_weight"] = 3.0
        selected, stats = trainer.prepare_training_rows(rows, "all", 1.0)
        self.assertEqual([row["training_weight"] for row in selected], [3.0, 1.0, 1.0])
        self.assertEqual(stats["training_weight_normalization_divisor"], 1.0)
        with self.assertRaises(trainer.TrainingError):
            trainer.prepare_training_rows(rows, "nlp", 0.5)

    def test_train_reader_requires_all_domains_and_per_row_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            rows = [
                valid_row("gsm8k/train/1", "math"),
                valid_row("opencode/train/1", "code"),
                valid_row("ceval/train/1", "nlp"),
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = trainer.read_rows(path)
            self.assertEqual([row["domain"] for row in loaded], ["math", "code", "nlp"])
            rows[0].pop("kl_weight")
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(trainer.TrainingError, "Missing kl_weight"):
                trainer.read_rows(path)

    def test_focused_reader_accepts_one_matching_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "code_train.jsonl"
            row = valid_row("opencode/train/1", "code")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = trainer.read_rows(path, "code")
            self.assertEqual([item["domain"] for item in loaded], ["code"])
            with self.assertRaisesRegex(trainer.TrainingError, "All three domains"):
                trainer.read_rows(path)

    def test_nlp_rationale_reader_accepts_only_the_audited_nlp_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nlp_rationale_train.jsonl"
            row = valid_row("ceval-rationale/train/1", "nlp")
            row["dataset_key"] = "ceval_rationale_train"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = trainer.read_rows(path, "nlp_rationale")
            self.assertEqual(len(loaded), 1)
            row["dataset_key"] = "ceval"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(trainer.TrainingError, "unapproved dataset_key"):
                trainer.read_rows(path, "nlp_rationale")

    def test_nlp_answer_first_reader_requires_two_locked_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nlp_answer_first_train.jsonl"
            row = valid_row("ceval-answer-first/train/1", "nlp")
            row["dataset_key"] = "ceval_answer_first_train"
            row["answer"] = "答案：B\n简短理由：该选项符合题干条件。\n最终答案：B"
            row["answer_token_position"] = "first"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = trainer.read_rows(path, "nlp_answer_first")
            selected, stats = trainer.prepare_training_rows(
                loaded, "nlp_answer_first", 4.0
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["answer_token_weight"], 8.0)
            self.assertEqual(stats["selected_dataset_counts"], {"ceval_answer_first_train": 1})
            row["answer"] = "答案：A\n简短理由：标签冲突。\n最终答案：B"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(trainer.TrainingError, "repeat one locked label"):
                trainer.read_rows(path, "nlp_answer_first")

    def test_reader_rejects_formal_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            rows = [
                valid_row("gsm8k/test/1", "math"),
                valid_row("opencode/train/1", "code"),
                valid_row("ceval/train/1", "nlp"),
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(trainer.TrainingError, "Formal-test"):
                trainer.read_rows(path)

    def test_resume_auto_selects_latest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (100, 200):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            incomplete = output / "checkpoint-300"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text("{}", encoding="utf-8")
            self.assertEqual(trainer.latest_checkpoint(output), output / "checkpoint-200")
            self.assertEqual(trainer.resolve_resume("auto", output), output / "checkpoint-200")


if __name__ == "__main__":
    unittest.main()
