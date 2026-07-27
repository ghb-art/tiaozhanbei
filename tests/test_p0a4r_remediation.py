from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nlp = load_module(
    "model_compression/generate_p0a4r_nlp_rationales.py",
    "p0a4r_nlp_rationales",
)
code_data = load_module(
    "model_compression/build_p0a4r_code_data.py",
    "p0a4r_code_data",
)
selector = load_module(
    "scripts/select_p0a4r_checkpoint.py",
    "p0a4r_checkpoint_selector",
)


class P0A4RRemediationTests(unittest.TestCase):
    def test_nlp_rationale_prompt_removes_label_only_instruction(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "Answer exactly."},
                {
                    "role": "user",
                    "content": (
                        "以下是单项选择题。只输出一个大写字母 A、B、C 或 D。\n\n"
                        "题目: 2+2等于多少？\nA. 3\nB. 4\nC. 5\nD. 6"
                    ),
                },
            ]
        }
        messages = nlp.rationale_messages(row, 0)
        self.assertNotIn("只输出一个大写字母", messages[-1]["content"])
        self.assertIn("FINAL: X", messages[-1]["content"])

    def test_nlp_rationale_requires_correct_final_and_nontrivial_reason(self) -> None:
        accepted, _, normalized = nlp.parse_verified_rationale(
            "因为二加二等于四，所以选择第二项。\nFINAL: B",
            "B",
            12,
            320,
        )
        self.assertTrue(accepted)
        self.assertTrue(normalized.endswith("FINAL: B"))
        wrong, reason, _ = nlp.parse_verified_rationale(
            "因为二加二等于四，所以选择第二项。\nFINAL: C",
            "B",
            12,
            320,
        )
        self.assertFalse(wrong)
        self.assertIn("choice_mismatch", reason)
        short, reason, _ = nlp.parse_verified_rationale("显然。\nFINAL: B", "B", 12, 320)
        self.assertFalse(short)
        self.assertEqual(reason, "rationale_too_short")

    def test_nlp_rationale_accepts_correct_inline_final_marker(self) -> None:
        accepted, reason, normalized = nlp.parse_verified_rationale(
            "根据题干中的关键条件，第二项与定义一致。 FINAL: B",
            "B",
            12,
            320,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "choice_and_rationale_verified")
        self.assertTrue(normalized.endswith("FINAL: B"))

    def test_nlp_resume_retries_only_rejected_in_incomplete_groups(self) -> None:
        rows = [
            {"validation_group_id": "g1"},
            {"validation_group_id": "g2"},
        ]
        existing = {
            "g1::rationale-0": {
                "validation_group_id": "g1",
                "accepted_for_training": True,
                "generation_attempt": 0,
            },
            "g1::rationale-1": {
                "validation_group_id": "g1",
                "accepted_for_training": False,
                "generation_attempt": 0,
            },
            "g2::rationale-0": {
                "validation_group_id": "g2",
                "accepted_for_training": False,
                "generation_attempt": 0,
            },
            "g2::rationale-1": {
                "validation_group_id": "g2",
                "accepted_for_training": False,
                "generation_attempt": 2,
            },
        }
        tasks = nlp.select_generation_tasks(rows, 2, existing, True, 1, 2)
        self.assertEqual([(nlp.group_id(row), variant) for row, variant in tasks], [("g2", 0)])

    def test_nlp_second_retry_repairs_with_train_label_but_student_prompt_is_label_free(self) -> None:
        row = {
            "answer": "B",
            "messages": [
                {
                    "role": "user",
                    "content": "题目: 2+2等于多少？\nA. 3\nB. 4\nC. 5\nD. 6",
                }
            ],
        }
        student_messages = nlp.rationale_messages(row, 0, 0)
        repair_messages = nlp.rationale_messages(row, 0, 2)
        self.assertNotIn("目标选项为 B", student_messages[-1]["content"])
        self.assertIn("目标选项为 B", repair_messages[-1]["content"])
        self.assertIn("FINAL: B", repair_messages[-1]["content"])

    def test_code_data_uses_one_executable_row_per_group(self) -> None:
        row = {
            "dataset_key": "humaneval",
            "sample_id": "mbpp/train/0001",
            "validation_group_id": "mbpp/task/0001",
            "source": "mbpp_official_train",
            "used_for_training": True,
            "messages": [
                {"role": "user", "content": 'def add(a, b):\n    """Return the sum."""\n'}
            ],
            "answer": "    return a + b",
            "code_eval": {
                "kind": "mbpp_assert_tests_v1",
                "entry_point": "add",
                "prompt_source": 'def add(a, b):\n    """Return the sum."""\n',
                "setup_code": "",
                "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps({**row, "sample_id": "duplicate"}) + "\n",
                encoding="utf-8",
            )
            unique = code_data.unique_code_rows(path, require_train=True)
            self.assertEqual(len(unique), 1)
            verified, rejected = code_data.verify_rows(unique, timeout_sec=2, workers=1)
            self.assertEqual(len(verified), 1)
            self.assertFalse(rejected)

    def test_code_data_rejects_protected_evaluation_identity(self) -> None:
        row = {
            "sample_id": "selection170/code/1",
            "validation_group_id": "train/group/1",
            "source": "train",
            "used_for_training": True,
        }
        with self.assertRaisesRegex(code_data.CodeDataError, "Forbidden evaluation identity"):
            code_data.require_train_identity(row, Path("train.jsonl"))

    def test_checkpoint_selector_uses_internal_task_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "p0a4r_code_internal_validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            base_trace = root / "base.jsonl"
            candidate_trace = root / "candidate.jsonl"
            base_rows = [
                {
                    "sample_id": f"p0a4r/internal/{index}",
                    "validation_group_id": f"mbpp/task/{index}",
                    "dataset_key": "humaneval",
                    "correct": index < 2,
                    "generation_error": "",
                }
                for index in range(4)
            ]
            candidate_rows = [
                {**row, "correct": index < 3}
                for index, row in enumerate(base_rows)
            ]
            for path, rows in ((base_trace, base_rows), (candidate_trace, candidate_rows)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            checkpoint = root / "checkpoint-3"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

            def write_audit(path: Path, trace: Path, adapter: dict) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "formal_test_labels_used": False,
                            "dataset_counts": {"humaneval": 4},
                            "generation_error_count": 0,
                            "validation_data": str(validation),
                            "validation_data_sha256": "same-validation",
                            "output_trace": str(trace),
                            "output_trace_sha256": selector.sha256_file(trace),
                            "adapter": adapter,
                        }
                    ),
                    encoding="utf-8",
                )

            baseline_audit = root / "baseline.json"
            candidate_audit = root / "candidate.json"
            write_audit(baseline_audit, base_trace, {})
            write_audit(candidate_audit, candidate_trace, {"path": str(checkpoint)})
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "policy": {
                            "feedback_source": "train_only_internal_validation",
                            "smoke96_item_feedback_used": False,
                            "selection170_feedback_used": False,
                            "formal_full_feedback_used": False,
                        },
                        "selection": {
                            "require_net_improvement": 1,
                            "max_regressions": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "selected"
            output_audit = root / "selection.json"
            argv = [
                "select",
                "--config",
                str(config),
                "--task",
                "humaneval",
                "--baseline-audit",
                str(baseline_audit),
                "--candidate-audit",
                str(candidate_audit),
                "--output-dir",
                str(output_dir),
                "--audit",
                str(output_audit),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(selector.main(), 0)
            report = json.loads(output_audit.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["selected_checkpoint_step"], 3)
            self.assertTrue((output_dir / "adapter_model.safetensors").is_file())

    def test_checkpoint_selector_requires_integral_scaled_alpha(self) -> None:
        self.assertEqual(selector.scaled_lora_alpha(16, 0.25), 4)
        self.assertEqual(selector.scaled_lora_alpha(16, 0.5), 8)
        with self.assertRaises(selector.SelectionError):
            selector.scaled_lora_alpha(8, 0.3)

    def test_launcher_exposes_only_train_internal_remediation_flow(self) -> None:
        launcher = (ROOT / "scripts/run_p0a4r.sh").read_text(encoding="utf-8")
        self.assertIn("train-code-pilot", launcher)
        self.assertIn("code-source-rebuild", launcher)
        self.assertIn("eval-code-pilot", launcher)
        self.assertIn("select-code-pilot", launcher)
        self.assertIn("require_code_data promotion", launcher)
        self.assertIn("require_router_lineage", launcher)
        self.assertIn("selected checkpoint is not in the matching training audit", launcher)
        self.assertIn('CODE_PILOT_OUTPUT="models/checkpoints/p0a4r/code-pilot"', launcher)
        self.assertIn("--external-checkpoint-selection", launcher)
        self.assertIn("select_p0a4r_checkpoint.py", launcher)
        self.assertIn("P0A4R_EXTRA_CODE_SOURCES", launcher)
        self.assertNotIn("p0a4_edge_student_full.jsonl", launcher)
        trainer = (ROOT / "model_compression/train_p0a4_lora.py").read_text(encoding="utf-8")
        self.assertIn("--external-checkpoint-selection", trainer)
        evaluator = (ROOT / "scripts/evaluate_edge_candidate_dev.py").read_text(encoding="utf-8")
        self.assertIn("--adapter-dir", evaluator)


if __name__ == "__main__":
    unittest.main()
