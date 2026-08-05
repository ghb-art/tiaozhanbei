from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("p0a5_builder", "model_compression/build_p0a5_data.py")
gate = load_module("p0a5_gate", "scripts/p0a5_gate.py")
trainer = load_module("p0a5_trainer", "model_compression/train_p0a5_lora.py")
distiller = load_module(
    "p0a5_distiller", "model_compression/generate_p0a5_distill.py"
)
gate_evaluator = load_module(
    "p0a5_gate_evaluator", "scripts/evaluate_p0a5_gate.py"
)
teacher_selection = load_module(
    "p0a5_teacher_selection", "scripts/select_p0a5_teacher.py"
)


class P0A5ProtocolTests(unittest.TestCase):
    def test_gate_parsers_accept_valid_common_answer_forms(self) -> None:
        self.assertEqual(
            gate_evaluator.extract_number("work: 16\n#### <number>"),
            "16",
        )
        self.assertEqual(gate_evaluator.extract_number("#### 54"), "54")
        self.assertEqual(gate_evaluator.extract_choice("正确答案是D。"), "D")
        self.assertEqual(gate_evaluator.extract_choice("因此故选B。"), "B")

    def test_gate_prompts_do_not_request_literal_placeholders(self) -> None:
        math_messages = gate_evaluator.build_messages(
            {"domain": "math", "prompt": "test"}
        )
        nlp_messages = gate_evaluator.build_messages(
            {"domain": "nlp", "prompt": "test"}
        )
        self.assertNotIn("<number>", math_messages[0]["content"])
        self.assertNotIn("最终答案：X", nlp_messages[0]["content"])

    def test_code_normalization_strips_empty_think_envelope_and_dedents(self) -> None:
        self.assertEqual(
            gate_evaluator.normalize_code_response(
                "<think>\n\n</think>\n\n    def f(x):\n        return x"
            ),
            "def f(x):\n    return x",
        )
        self.assertEqual(
            gate_evaluator.normalize_code_response("plain body"), "plain body"
        )

    def test_code_scoring_recovers_envelope_wrapped_complete_function(self) -> None:
        wrapped = (
            "<think>\n\n</think>\n\n"
            "    def add(a, b):\n        return a + b"
        )
        tests = ["assert add(1, 2) == 3"]
        passed, detail = gate_evaluator.score_code(wrapped, tests, 5)
        self.assertTrue(passed, detail)
        # Without protocol v3 normalization the same response is invalid python.
        raw_source = builder.extract_code(wrapped) + "\n\n" + tests[0] + "\n"
        self.assertFalse(builder.safe_python(raw_source))

    def test_teacher_uses_reentrant_gradient_checkpointing_for_zero3(self) -> None:
        self.assertEqual(
            trainer.gradient_checkpointing_config("teacher"),
            {"enabled": True, "use_reentrant": True},
        )
        self.assertEqual(
            trainer.gradient_checkpointing_config("student"),
            {"enabled": True, "use_reentrant": False},
        )
        self.assertFalse(trainer.checkpoint_due(49, 50))
        self.assertTrue(trainer.checkpoint_due(50, 50))
        self.assertTrue(trainer.checkpoint_due(100, 50))

    def test_ddp_early_stop_uses_max_decision_across_ranks(self) -> None:
        class FakeTensor:
            def __init__(self, value: int):
                self.value = value

            def item(self) -> int:
                return self.value

        class FakeDistributed:
            class ReduceOp:
                MAX = "max"

            @staticmethod
            def is_initialized() -> bool:
                return True

            @staticmethod
            def all_reduce(value, op) -> None:
                self.assertEqual(op, "max")
                value.value = 1

        class FakeTorch:
            distributed = FakeDistributed
            int32 = "int32"

            @staticmethod
            def tensor(value, device, dtype):
                self.assertEqual(device, "cuda:0")
                self.assertEqual(dtype, "int32")
                return FakeTensor(value)

        self.assertTrue(
            trainer.synchronize_distributed_stop(False, FakeTorch, "cuda:0")
        )

    def test_config_has_one_gate_and_no_retired_sources(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a5_capability.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["gate300"]["expected_counts"],
            {"math": 100, "code": 100, "nlp": 100},
        )
        self.assertEqual(config["gate300"]["initial_ratio"], 0.78)
        self.assertEqual(config["gate300"]["recommended_full_ratio"], 0.82)
        self.assertEqual(
            sum(config["datasets"]["nlp"]["category_train_quotas"].values()),
            config["datasets"]["nlp"]["train_rows"],
        )
        self.assertEqual(config["datasets"]["nlp"]["train_rows"], 9500)
        self.assertEqual(config["teacher_training"]["checkpoint_steps"], 50)
        self.assertEqual(config["student_training"]["checkpoint_steps"], 50)
        self.assertEqual(
            trainer.early_stopping_config(config["teacher_training"]),
            {
                "enabled": True,
                "metric": "eval_loss",
                "eval_steps": 100,
                "patience": 2,
                "threshold": 0.001,
            },
        )
        serialized = json.dumps(config, ensure_ascii=False).casefold()
        for retired in ("mbpp", "mmlu_aux", "smoke96", "selection170", "code_contests"):
            self.assertNotIn(retired, serialized)

    def test_nlp_category_mapping_is_multidomain(self) -> None:
        exam = {
            "task_type": {"major": ["试题"]},
            "domain": ["历史"],
        }
        self.assertIn("exam", builder.nlp_eligible_categories(exam))
        self.assertIn(
            "humanities_social_science", builder.nlp_eligible_categories(exam)
        )
        logic = {
            "task_type": {"major": ["自然语言推理"]},
            "domain": ["通用"],
        }
        self.assertIn("language_reasoning", builder.nlp_eligible_categories(logic))

    def test_training_token_budget_requires_complete_sequence(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                del kwargs
                return list(range(sum(len(item["content"]) for item in messages)))

        tokenizer = FakeTokenizer()
        self.assertTrue(
            builder.within_training_token_budget(tokenizer, "short", "answer", 200)
        )
        self.assertFalse(
            builder.within_training_token_budget(tokenizer, "x" * 200, "answer", 200)
        )
        self.assertTrue(
            builder.within_all_training_token_budgets(
                [tokenizer, tokenizer], "short", "answer", 200
            )
        )

    def test_distill_answer_falls_back_when_it_exceeds_student_budget(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                del kwargs
                return list(range(sum(len(item["content"]) for item in messages)))

        source = {
            "sample_id": "nlp/long",
            "messages": [{"role": "user", "content": "question"}],
            "answer": "verified",
        }
        answer, validation, before, after, repaired = (
            distiller.enforce_training_token_budget(
                FakeTokenizer(),
                source,
                "rationale-" * 20 + "verified",
                "teacher_rationale_human_reference",
                100,
            )
        )
        self.assertTrue(repaired)
        self.assertGreater(before, 100)
        self.assertLessEqual(after, 100)
        self.assertEqual(answer, "verified")
        self.assertEqual(validation, "source_verified_token_budget_fallback")

    def test_repeated_nlp_units_are_removed_without_rewriting_content(self) -> None:
        self.assertEqual(
            builder.deduplicate_repeated_units("第一条信息。\n第二条信息。\n第一条信息。\n"),
            "第一条信息。\n第二条信息。\n第一条信息。",
        )
        repeated = "这是一条长度足够的重复信息。"
        self.assertEqual(
            builder.deduplicate_repeated_units(f"{repeated}\n保留内容。\n{repeated}"),
            f"{repeated}\n保留内容。",
        )

    def test_task_weights_produce_requested_loss_mass(self) -> None:
        rows = [
            *[
                {"dataset_key": "gsm8k", "sample_id": f"m{index}"}
                for index in range(3)
            ],
            *[
                {"dataset_key": "opencodeinstruct", "sample_id": f"c{index}"}
                for index in range(7)
            ],
            *[
                {"dataset_key": "cmmlu", "sample_id": f"n{index}"}
                for index in range(5)
            ],
        ]
        target = {"gsm8k": 0.3, "opencodeinstruct": 0.35, "cmmlu": 0.35}
        builder.add_task_weights(rows, target)
        mass: Counter[str] = Counter()
        for row in rows:
            mass[row["dataset_key"]] += row["training_weight"]
        total = sum(mass.values())
        for task, expected in target.items():
            self.assertAlmostEqual(mass[task] / total, expected)
        self.assertTrue(all(row["preserve_math"] for row in rows[:3]))
        self.assertFalse(any(row["preserve_math"] for row in rows[3:]))

    def test_gate_trace_loader_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            row = {
                "sample_id": "x",
                "domain": "math",
                "prompt_hash": "h",
                "correct": True,
                "generation_error": "",
            }
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(gate.GateError):
                gate.read_trace(path, "test")

    def test_teacher_selection_uses_lowest_validation_loss(self) -> None:
        shared = {
            "config_hash": "config",
            "validation_data_hash": "validation",
            "evaluated_rows": 2200,
        }
        checkpoint_600 = {
            **shared,
            "checkpoint_step": 600,
            "evaluation_metrics": {"eval_loss": 1.02},
        }
        checkpoint_800 = {
            **shared,
            "checkpoint_step": 800,
            "evaluation_metrics": {"eval_loss": 0.98},
        }
        self.assertIs(
            teacher_selection.select_best([checkpoint_600, checkpoint_800]),
            checkpoint_800,
        )

    def test_runbook_exposes_cpu_stop_point(self) -> None:
        launcher = (ROOT / "scripts/run_p0a5.sh").read_text(encoding="utf-8")
        self.assertIn("CPU preflight complete. No GPU command was started.", launcher)
        self.assertIn("teacher-train", launcher)
        self.assertIn("--resume-from-checkpoint", launcher)
        self.assertIn("checkpoint-[0-9]*", launcher)
        self.assertIn('trainer_state.json"', launcher)
        self.assertIn('adapter_model.safetensors"', launcher)
        self.assertIn("teacher-eval [600|800]", launcher)
        self.assertIn("teacher-select", launcher)
        self.assertIn("student-quantize [1|2]", launcher)
        self.assertIn("gate300-pipeline [1|2]", launcher)
        self.assertTrue(
            (ROOT / "scripts/prepare_p0a5_quantized_student.py").is_file()
        )
        self.assertTrue(
            (ROOT / "scripts/run_p0a5_gate300_pipeline.sh").is_file()
        )
        self.assertNotIn("smoke96", launcher)
        self.assertNotIn("selection170", launcher)


if __name__ == "__main__":
    unittest.main()
