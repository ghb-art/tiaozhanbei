from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "serve_vllm_teachers", ROOT / "scripts/serve_vllm_teachers.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launcher = load_launcher()


class VllmTeacherLauncherTests(unittest.TestCase):
    def test_four_gpus_form_one_tensor_parallel_group(self) -> None:
        groups = launcher.parse_gpu_groups(["0,1,2,3"], [])
        self.assertEqual(groups, [("0", "1", "2", "3")])
        self.assertEqual(launcher.resolve_tensor_parallel_size("auto", 4), 4)

    def test_tensor_parallel_size_must_use_complete_group(self) -> None:
        with self.assertRaises(ValueError):
            launcher.resolve_tensor_parallel_size("1", 4)

    def test_command_uses_four_way_tensor_parallelism(self) -> None:
        args = argparse.Namespace(
            vllm_bin=".venv/bin/vllm",
            model_dir="models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ",
            quantization="awq",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            host="0.0.0.0",
            disable_log_requests=True,
        )
        teacher = launcher.TeacherSpec(
            gpus=("0", "1", "2", "3"), port=8000, tensor_parallel_size=4
        )
        command = launcher.build_command(args, teacher)
        tp_index = command.index("--tensor-parallel-size")
        self.assertEqual(command[tp_index + 1], "4")

    def test_command_exposes_stable_served_model_name(self) -> None:
        args = argparse.Namespace(
            vllm_bin=".venv/bin/vllm",
            model_dir="models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ",
            quantization="awq",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            host="0.0.0.0",
            disable_log_requests=True,
            served_model_name="baseline-14b-awq",
        )
        teacher = launcher.TeacherSpec(
            gpus=("0", "1", "2", "3"), port=8001, tensor_parallel_size=4
        )
        command = launcher.build_command(args, teacher)
        name_index = command.index("--served-model-name")
        self.assertEqual(command[name_index + 1], "baseline-14b-awq")

    def test_active_entrypoint_passes_one_gpu_group(self) -> None:
        script = (ROOT / "scripts/run_p0a.sh").read_text(encoding="utf-8")
        self.assertIn('--gpu-group "$P0A_GPUS"', script)
        self.assertIn("--tensor-parallel-size auto", script)
        self.assertRegex(script, r"teacher_serve\(\) \{\s+gpu_preflight")
        self.assertIn("teacher-plan", script)
        self.assertIn("teacher-stop", script)


if __name__ == "__main__":
    unittest.main()
