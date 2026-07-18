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

    def test_endpoint_lora_map_parses_dataset_ids(self) -> None:
        result = capability.parse_endpoint_lora_map(["gsm8k=0,humaneval=1", "cmmlu=2"])
        self.assertEqual(result, {"gsm8k": 0, "humaneval": 1, "cmmlu": 2})

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

    def test_candidate_artifact_includes_resident_loras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.gguf"
            lora_a = root / "a.gguf"
            lora_b = root / "b.gguf"
            base.write_bytes(b"b" * 10)
            lora_a.write_bytes(b"a" * 3)
            lora_b.write_bytes(b"c" * 4)
            result = g0.candidate_result(
                {
                    "name": "router",
                    "gguf": str(base),
                    "lora_adapters": [str(lora_a), str(lora_b)],
                },
                {},
            )
            self.assertTrue(result["artifact_exists"])
            self.assertEqual(result["base_gguf_bytes"], 10)
            self.assertEqual(result["resident_lora_bytes"], 7)
            self.assertEqual(result["artifact_bytes"], 17)


if __name__ == "__main__":
    unittest.main()
