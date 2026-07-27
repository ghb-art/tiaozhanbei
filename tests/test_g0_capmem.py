from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g0 = load_module("scripts/run_g0_capmem.py", "run_g0_capmem")
g3 = load_module("scripts/verify_gate_g3_gguf.py", "verify_gate_g3_gguf")
capability = load_module("scripts/evaluate_chapter2_capability.py", "evaluate_chapter2_capability")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class G0CapmemTests(unittest.TestCase):
    def test_matched_smoke_ratios_use_identical_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            datasets = ["gsm8k", "humaneval", "cmmlu"]
            edge_rows = []
            cloud_rows = []
            for dataset in datasets:
                for index in range(10):
                    sample_id = f"{dataset}/test/{index}"
                    cloud_rows.append({"sample_id": sample_id, "dataset_key": dataset, "correct": True})
                    edge_rows.append({"sample_id": sample_id, "dataset_key": dataset, "correct": index < 8})
            edge = tmp_path / "edge.jsonl"
            cloud = tmp_path / "cloud.jsonl"
            write_jsonl(edge, edge_rows)
            write_jsonl(cloud, cloud_rows)

            result = g0.ratios_from_trace(edge, cloud, 0.8)

            self.assertTrue(result["passed"])
            self.assertEqual(result["sample_count"], 30)
            self.assertEqual(result["ratios"], {"math_ratio": 0.8, "code_ratio": 0.8, "nlp_ratio": 0.8})

    def test_missing_cloud_sample_rejects_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            edge = tmp_path / "edge.jsonl"
            cloud = tmp_path / "cloud.jsonl"
            write_jsonl(edge, [{"sample_id": "gsm8k/test/0", "dataset_key": "gsm8k", "correct": True}])
            write_jsonl(cloud, [])

            result = g0.ratios_from_trace(edge, cloud, 0.8)

            self.assertFalse(result["passed"])
            self.assertEqual(result["missing_cloud_sample_count"], 1)

    def test_recommendation_prioritizes_missing_artifacts(self) -> None:
        results = [
            {
                "joint_feasible": False,
                "artifact_exists": False,
                "memory": {"passed": None},
                "capability": {"passed": False},
            }
        ]
        self.assertTrue(g0.recommended_action(results).startswith("prepare_quantized_artifacts"))

    def test_pruned_candidate_does_not_remain_pending(self) -> None:
        result = g0.candidate_result(
            {"name": "dominated", "decision_status": "pruned", "prune_reason": "larger"},
            {},
        )
        self.assertEqual(result["status"], "pruned")
        self.assertFalse(result["joint_feasible"])
        self.assertEqual(result["prune_reason"], "larger")

    def test_memory_sampler_records_process_total(self) -> None:
        sampler = g3.MemorySampler(psutil.Process(os.getpid()), interval_ms=50, include_gpu=False)
        sampler.sample_once()

        self.assertFalse(sampler.errors)
        self.assertEqual(len(sampler.samples), 1)
        sample = sampler.samples[0]
        self.assertGreater(float(sample["rss_mb_decimal"]), 0)
        self.assertEqual(sample["gpu_memory_mb_decimal"], 0.0)
        self.assertEqual(sample["total_memory_mb_decimal"], sample["rss_mb_decimal"])

    def test_percentile_handles_interpolation(self) -> None:
        self.assertEqual(g3.percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_memory_gate_disables_host_prompt_cache(self) -> None:
        self.assertEqual(
            g3.prompt_cache_server_args(0, False),
            ["--cache-ram", "0", "--no-cache-idle-slots"],
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            g3.prompt_cache_server_args(0, True)
        with self.assertRaisesRegex(ValueError, "reserved"):
            g3.validate_server_extra_args(["--cache-ram=8192"])

    def test_max_new_tokens_map_parses_positive_limits(self) -> None:
        result = capability.parse_max_new_tokens_map(["cmmlu=16,gsm8k=160", "humaneval=256"])
        self.assertEqual(result, {"cmmlu": 16, "gsm8k": 160, "humaneval": 256})

        with self.assertRaises(capability.CapabilityEvalError):
            capability.parse_max_new_tokens_map(["cmmlu=0"])

    def test_fail_fast_accuracy_map_validates_range(self) -> None:
        result = capability.parse_min_accuracy_map(["cmmlu=0.5,gsm8k=0.75"])
        self.assertEqual(result, {"cmmlu": 0.5, "gsm8k": 0.75})

        with self.assertRaises(capability.CapabilityEvalError):
            capability.parse_min_accuracy_map(["humaneval=1.01"])

    def test_chapter2_endpoint_explicitly_disables_qwen3_thinking(self) -> None:
        captured = {}
        original = capability.request_json

        def fake_request(url, payload, timeout_sec):
            captured.update(payload)
            return {"choices": [{"message": {"content": "ok"}}]}

        capability.request_json = fake_request
        try:
            text, _ = capability.generate_text_endpoint(
                "http://127.0.0.1:1",
                "student",
                [{"role": "user", "content": "x"}],
                1.0,
                8,
                True,
                {"lora": {"id": 0, "scale": 1.0}},
            )
            self.assertEqual(text, "ok")
            self.assertEqual(captured["chat_template_kwargs"], {"enable_thinking": False})
            with self.assertRaises(capability.CapabilityEvalError):
                capability.generate_text_endpoint(
                    "http://127.0.0.1:1",
                    "student",
                    [{"role": "user", "content": "x"}],
                    1.0,
                    8,
                    True,
                    {"chat_template_kwargs": {"enable_thinking": True}},
                )
        finally:
            capability.request_json = original

    def test_candidate_artifact_is_the_quantized_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.gguf"
            base.write_bytes(b"b" * 10)
            result = g0.candidate_result(
                {
                    "name": "edge-base",
                    "gguf": str(base),
                },
                {},
            )
            self.assertTrue(result["artifact_exists"])
            self.assertEqual(result["artifact_bytes"], 10)


if __name__ == "__main__":
    unittest.main()
