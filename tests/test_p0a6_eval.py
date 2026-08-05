from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/evaluate_p0a6_internal.py"
    spec = importlib.util.spec_from_file_location("p0a6_internal_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_module()


class P0A6EvaluatorTests(unittest.TestCase):
    def test_math_parser_reports_canonical_final_line(self) -> None:
        self.assertEqual(evaluator.extract_number("work\n#### 1,200.0"), ("1200", True))
        self.assertEqual(evaluator.extract_number("the answer is -2.50"), ("-2.5", False))
        self.assertEqual(evaluator.extract_number("work=16\n#### <number>"), ("16", False))

    def test_nlp_parser_strictly_prioritizes_final_line(self) -> None:
        response = "先考虑A，但排除它。\n正确答案可能是B。\n最终答案：D"
        self.assertEqual(evaluator.extract_choice(response), ("D", True))
        self.assertEqual(evaluator.extract_choice("brief reason\nFINAL: C"), ("C", True))
        self.assertEqual(evaluator.extract_choice("分析后故选C。"), ("C", False))
        self.assertEqual(evaluator.extract_choice("A和B都需要讨论。"), ("", False))

    def test_code_runs_in_isolated_interpreter(self) -> None:
        passed, detail, canonical = evaluator.score_code(
            "```python\ndef add(a, b):\n    return a + b\n```",
            ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
            2,
        )
        self.assertTrue(passed, detail)
        self.assertTrue(canonical)
        passed, detail, _ = evaluator.score_code(
            "```python\nimport os\ndef add(a, b): return a + b\n```",
            ["assert add(2, 3) == 5"],
            2,
        )
        self.assertFalse(passed)
        self.assertEqual(detail, "unsafe_or_invalid_python")

    def test_sidecar_supplies_expected_counts_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "quick_validation.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            digest = evaluator.sha256_file(manifest)
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "validation": {
                            "quick": {
                                "path": "quick_validation.jsonl",
                                "hash": digest,
                                "expected_counts": {
                                    "math": 2,
                                    "code": 3,
                                    "nlp": 4,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            counts, source, sidecar, declared = evaluator.expected_counts_from_manifest(
                manifest, Counter({"math": 2, "code": 3, "nlp": 4})
            )
            self.assertEqual(counts, {"math": 2, "code": 3, "nlp": 4})
            self.assertIn("validation.quick", source)
            self.assertEqual(sidecar, directory / "manifest.json")
            self.assertEqual(declared, digest)

    def test_validation_rejects_formal_or_retired_samples(self) -> None:
        rows = [
            {
                "sample_id": "gsm8k/train/1",
                "domain": "math",
                "prompt": "q",
                "reference": "1",
                "split_role": "quick_validation",
            },
            {
                "sample_id": "code/train/1",
                "domain": "code",
                "prompt": "q",
                "reference": "",
                "unit_tests": ["assert True"],
                "split_role": "quick_validation",
            },
            {
                "sample_id": "cmmlu/test/1",
                "domain": "nlp",
                "prompt": "q",
                "reference": "A",
                "split_role": "quick_validation",
            },
        ]
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.validate_rows(rows, "quick_validation.jsonl")

    def test_outputs_are_confined_to_p0a6_audit_directory(self) -> None:
        evaluator.validate_output_path(
            ROOT / "reports/audit/p0a6/candidate/trace.jsonl", ".jsonl"
        )
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.validate_output_path(ROOT / "data/eval/trace.jsonl", ".jsonl")
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.validate_output_path(
                ROOT / "reports/sealed/p0a6/trace.jsonl", ".jsonl"
            )

    def test_domain_model_ids_override_and_fall_back_to_default(self) -> None:
        with patch.object(
            evaluator,
            "discover_model",
            side_effect=lambda endpoint, requested, timeout: requested,
        ) as discover:
            result = evaluator.discover_models_by_domain(
                "http://127.0.0.1:8000",
                "shared-base",
                {"math": "", "code": "", "nlp": "nlp-specialist"},
                30,
            )
        self.assertEqual(
            result,
            {
                "math": "shared-base",
                "code": "shared-base",
                "nlp": "nlp-specialist",
            },
        )
        self.assertEqual(discover.call_count, 3)

    def test_empty_domain_models_preserve_single_model_behavior(self) -> None:
        with patch.object(evaluator, "discover_model", return_value="served"):
            result = evaluator.discover_models_by_domain(
                "http://127.0.0.1:8000",
                "",
                {domain: "" for domain in evaluator.DOMAINS},
                30,
            )
        self.assertEqual(result, {domain: "served" for domain in evaluator.DOMAINS})


if __name__ == "__main__":
    unittest.main()
