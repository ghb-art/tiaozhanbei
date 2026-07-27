from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contest = load_module(
    "p0a4r3_code_contests_utils",
    "model_compression/code_contests_utils.py",
)
nlp = load_module(
    "p0a4r3_nlp_data",
    "model_compression/generate_p0a4r3_nlp_data.py",
)


class P0A4R3SharedTests(unittest.TestCase):
    def test_contest_solution_must_pass_stdio_tests(self) -> None:
        source = (
            "import sys\n"
            "values = [int(value) for value in sys.stdin.read().split()]\n"
            "print(sum(values))\n"
        )
        result = contest.run_contest_tests(
            source,
            [
                {"input": "1 2\n", "output": "3\n"},
                {"input": "10 -3 5\n", "output": "12\n"},
            ],
            2.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.passed_tests, 2)

    def test_contest_solution_rejects_system_access(self) -> None:
        safe, reason = contest.safe_contest_source(
            "import os\nos.system('echo unsafe')\n"
        )
        self.assertFalse(safe)
        self.assertIn(reason, {"forbidden_import", "forbidden_attribute"})

    def test_nlp_teacher_output_requires_matching_train_label(self) -> None:
        request = {
            "request_id": "nlp-1",
            "validation_group_id": "mmlu-aux/science/1",
            "domain": "science",
            "expected_label": "B",
            "source_question_hash": "abc",
            "split_role": "train",
        }
        valid_response = json.dumps(
            {
                "question_zh": "下列哪一种情况最可能导致人体出现发热症状？",
                "options_zh": {
                    "A": "运动后腿部肌肉放松",
                    "B": "血液中出现大量细菌",
                    "C": "皮肤表面存在少量病毒颗粒",
                    "D": "胃中正在消化碳水化合物"
                },
                "reason_zh": "血液中的细菌感染会激活免疫反应并使体温升高。",
                "final": "B"
            },
            ensure_ascii=False,
        )
        row, reason = nlp.validate_translation(request, valid_response)
        self.assertEqual(reason, "")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertTrue(row["used_for_training"])
        mismatch = json.loads(valid_response)
        mismatch["final"] = "A"
        rejected, reason = nlp.validate_translation(
            request, json.dumps(mismatch, ensure_ascii=False)
        )
        self.assertIsNone(rejected)
        self.assertEqual(reason, "label_verification_failed")

    def test_nlp_verified_selection_keeps_all_domains_without_duplication(self) -> None:
        rows = []
        for domain_index in range(8):
            count = 8 if domain_index == 0 else 12
            for row_index in range(count):
                rows.append(
                    {
                        "domain": f"domain-{domain_index}",
                        "sample_id": f"{domain_index}-{row_index}",
                    }
                )
        selected = nlp.select_balanced_verified(
            rows,
            target=80,
            seed=1,
            split_name="test",
            minimum_equal_quota_ratio=0.8,
        )
        self.assertEqual(len(selected), 80)
        self.assertEqual(len({row["sample_id"] for row in selected}), 80)
        counts = {}
        for row in selected:
            counts[row["domain"]] = counts.get(row["domain"], 0) + 1
        self.assertEqual(len(counts), 8)
        self.assertGreaterEqual(min(counts.values()), 8)

    def test_protocol_freezes_math_and_only_registers_two_shared_candidates(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a4r3_shared_distillation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["policy"]["math_status"], "frozen_replay_only")
        self.assertFalse(config["policy"]["adapter_training_allowed"])
        self.assertEqual(config["policy"]["max_preregistered_candidates"], 2)
        self.assertEqual(
            config["models"]["teacher"]["request_model_id"],
            "distill-teacher-v1",
        )
        self.assertEqual(
            config["models"]["teacher"]["tensor_parallel_size"],
            4,
        )
        self.assertEqual(
            config["models"]["teacher"]["fallback_request_model_id"],
            "auto_base_from_endpoint",
        )
        self.assertEqual(config["training"]["student_shared"]["lora_rank"], 8)
        self.assertEqual(
            config["training"]["student_shared"]["candidate_overrides"]["2"][
                "lora_rank"
            ],
            16,
        )
        self.assertEqual(config["quantization"]["type"], "Q4_K_M")
        self.assertEqual(config["quantization"]["kv_cache_type"], "q8_0")
        protected = set(config["data"]["old_validation_sets"])
        self.assertIn("data/distill/p0a4_smoke96.jsonl", protected)
        self.assertIn(
            "data/distill/p0a2_recovery_validation.jsonl",
            protected,
        )

    def test_launcher_has_no_adapter_training_route(self) -> None:
        source = (ROOT / "scripts/run_p0a4r3.sh").read_text(encoding="utf-8")
        self.assertNotIn("student_expert", source)
        self.assertNotIn("train-adapter", source)
        self.assertIn("--role student_shared", source)
        self.assertIn("run_with_memory_guard.sh", source)


if __name__ == "__main__":
    unittest.main()
