from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retention = load_module(
    "scripts/summarize_edge_candidate_dev.py", "summarize_edge_candidate_dev"
)
evaluator = load_module("scripts/evaluate_edge_candidate_dev.py", "edge_candidate_evaluator")


class FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "prompt"


class P0A3ReselectionTests(unittest.TestCase):
    def test_matched_retention_requires_every_task(self) -> None:
        teacher = []
        candidate = []
        for dataset in ("gsm8k", "humaneval", "cmmlu"):
            for index in range(10):
                sample_id = f"{dataset}/dev/{index}"
                teacher.append(
                    {"sample_id": sample_id, "dataset_key": dataset, "correct": True}
                )
                candidate.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": dataset,
                        "correct": index < (7 if dataset == "humaneval" else 8),
                        "generation_error": "",
                    }
                )
        result = retention.compare_traces(teacher, candidate, 0.8)
        self.assertFalse(result["passed"])
        self.assertIn("code_ratio", result["ratio_failures"])
        self.assertNotIn("math_ratio", result["ratio_failures"])

    def test_matched_retention_rejects_sample_mismatch(self) -> None:
        teacher = [
            {"sample_id": "gsm8k/dev/0", "dataset_key": "gsm8k", "correct": True}
        ]
        candidate = [
            {
                "sample_id": "gsm8k/dev/1",
                "dataset_key": "gsm8k",
                "correct": True,
                "generation_error": "",
            }
        ]
        result = retention.compare_traces(teacher, candidate, 0.8)
        self.assertFalse(result["matched_sample_ids"])
        self.assertFalse(result["passed"])

    def test_matched_retention_rejects_incomplete_or_teacher_errors(self) -> None:
        teacher = [
            {
                "sample_id": "gsm8k/dev/0",
                "dataset_key": "gsm8k",
                "correct": True,
                "generation_error": "TimeoutError",
            }
        ]
        candidate = [
            {
                "sample_id": "gsm8k/dev/0",
                "dataset_key": "gsm8k",
                "correct": True,
                "generation_error": "",
            }
        ]
        result = retention.compare_traces(teacher, candidate, 0.8)
        self.assertFalse(result["complete_frozen_dev"])
        self.assertEqual(result["teacher_generation_error_count"], 1)
        self.assertFalse(result["passed"])

    def test_qwen3_template_disables_thinking(self) -> None:
        tokenizer = FakeTokenizer()
        evaluator.render_generation_prompt(
            tokenizer,
            [{"role": "user", "content": "answer"}],
            close_reasoning_prefix=False,
            disable_thinking=True,
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_endpoint_urls_accept_base_or_v1(self) -> None:
        self.assertEqual(
            evaluator.chat_completions_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            evaluator.endpoint_health_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/health",
        )
        self.assertEqual(
            evaluator.endpoint_models_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/models",
        )

    def test_endpoint_model_id_is_discovered_and_validated(self) -> None:
        served = ["/absolute/path/to/qwen14"]
        self.assertEqual(evaluator.select_served_model_id(served, "auto"), served[0])
        with self.assertRaises(evaluator.RecoveryEvalError):
            evaluator.select_served_model_id(served, "relative/path/to/qwen14")

    def test_reselection_config_orders_q3_candidates(self) -> None:
        config = json.loads(
            (ROOT / "configs/p0a3_reselection.json").read_text(encoding="utf-8")
        )
        candidates = config["candidates"]
        self.assertEqual(candidates[0]["model_id"], "Qwen/Qwen3-1.7B")
        self.assertEqual(candidates[0]["quant_type"], "Q3_K_M")
        self.assertEqual(candidates[1]["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertFalse(config["decision_policy"]["train_before_hf_dev_pass"])

        runtime = json.loads(
            (ROOT / "configs/g0_capmem_candidates.json").read_text(encoding="utf-8")
        )["common"]
        self.assertEqual(runtime["cache_type_k"], "q8_0")
        self.assertEqual(runtime["cache_type_v"], "q8_0")

    def test_final_dev_config_uses_p0a3_total_memory_margin(self) -> None:
        config = (ROOT / "configs/final_config_dev.yaml").read_text(encoding="utf-8")
        self.assertIn("g1_min_ratio: 0.80", config)
        self.assertIn("g3_max_total_memory_mb_decimal: 1400", config)
        self.assertNotIn("g3_max_rss_mb_decimal", config)

    def test_launcher_has_no_training_command_and_guards_formal_gate(self) -> None:
        launcher = (ROOT / "scripts/run_p0a3.sh").read_text(encoding="utf-8")
        self.assertIn("teacher-dev", launcher)
        self.assertIn("qwen3-hf", launcher)
        self.assertIn("qwen3-q3-dev", launcher)
        self.assertIn("qwen3-f16-control", launcher)
        self.assertIn("qwen3-1p7b-f16-gguf-f16kv-control", launcher)
        self.assertIn("q3_q8kv_retention.json", launcher)
        self.assertNotIn(
            'Path("reports/audit/gate_p0a3_qwen3_1p7b_q3_retention.json")',
            launcher,
        )
        self.assertIn('--cache-type-k "$cache_type"', launcher)
        self.assertIn('--kv-cache-type "$cache_type"', launcher)
        self.assertIn("ensure_memory_margin", launcher)
        self.assertIn("resolve_eval_gpu", launcher)
        self.assertGreaterEqual(
            launcher.count('ensure_audit_status "$TEACHER_AUDIT" passed'), 2
        )
        self.assertRegex(launcher, r"download_fallback\(\) \{\s+fallback_guard")
        self.assertIn('value.get("complete_frozen_dev") is True', launcher)
        self.assertIn('value.get("matched_sample_ids") is True', launcher)
        self.assertIn('value.get("generation_error_count") == 0', launcher)
        self.assertIn("--require-feasible", launcher)
        self.assertNotIn("train_cedd_repair.py", launcher)
        self.assertNotIn("HumanEval.jsonl", launcher)

    def test_closed_training_and_adapter_code_is_removed(self) -> None:
        obsolete = (
            "model_compression/train_cedd_repair.py",
            "model_compression/train_cedd_structured.py",
            "model_compression/export_merged_hf.py",
            "model_compression/lora_utils.py",
            "scripts/summarize_chapter2_capability.py",
        )
        self.assertTrue(all(not (ROOT / path).exists() for path in obsolete))
        for relative in (
            "scripts/run_g0_capmem.py",
            "scripts/verify_gate_g3_gguf.py",
            "scripts/evaluate_chapter2_capability.py",
        ):
            active_code = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("lora", active_code)
            self.assertNotIn("adapter-map", active_code)

    def test_llama_cpp_utf8_parser_patch_is_reproducible(self) -> None:
        patch = (
            ROOT / "patches/llama_cpp_chat_utf8_sanitize.patch"
        ).read_text(encoding="utf-8")
        setup = (ROOT / "scripts/setup_llama_cpp.sh").read_text(encoding="utf-8")
        self.assertIn("common_chat_sanitize_utf8", patch)
        self.assertIn("llama_cpp_chat_utf8_sanitize.patch", setup)
        self.assertIn("2d973636e292ee6f75fadcf08d29cb33511f509f", setup)


if __name__ == "__main__":
    unittest.main()
