from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = load_module("scripts/p0a4_protocol.py", "p0a4_protocol")
retention = load_module("scripts/p0a4_retention_gate.py", "p0a4_retention_gate")
trials = load_module("scripts/p0a4_trials.py", "p0a4_trials")
trainer = load_module("model_compression/train_p0a4_lora.py", "p0a4_trainer")
launcher = load_module("scripts/serve_vllm_teachers.py", "p0a4_vllm_launcher")
edge_eval = load_module("scripts/evaluate_edge_candidate_dev.py", "p0a4_edge_eval")
imatrix = load_module("scripts/build_imatrix_calibration.py", "p0a4_imatrix")
v2_route = load_module("scripts/p0a4_select_v2_route.py", "p0a4_v2_route")


class P0A4PipelineTests(unittest.TestCase):
    def test_official_full_manifest_is_complete_and_separate(self) -> None:
        manifest = json.loads(
            (ROOT / "data/splits/p0a4_official_full/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["dataset_counts"],
            {"gsm8k": 1319, "humaneval": 164, "cmmlu": 11582},
        )
        self.assertEqual(manifest["total_count"], 13065)
        self.assertEqual(
            len((ROOT / "data/splits/gsm8k_test.txt").read_text().splitlines()), 500
        )

    def test_p0a4_holdouts_and_selection_are_group_disjoint(self) -> None:
        paths = {
            "train": ROOT / "data/distill/p0a4_train.jsonl",
            "teacher": ROOT / "data/distill/p0a4_teacher_validation.jsonl",
            "smoke": ROOT / "data/distill/p0a4_smoke96.jsonl",
            "selection": ROOT / "data/distill/p0a2_recovery_validation.jsonl",
        }
        groups = {}
        counts = {}
        for name, path in paths.items():
            rows = protocol.read_jsonl(path)
            groups[name] = {protocol.group_id(row) for row in rows}
            counts[name] = {
                task: sum(row.get("dataset_key") == task for row in rows)
                for task in protocol.TASKS
            }
        self.assertEqual(counts["teacher"], {task: 32 for task in protocol.TASKS})
        self.assertEqual(counts["smoke"], {task: 32 for task in protocol.TASKS})
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertFalse(groups[left] & groups[right], f"{left}/{right}")

    def test_p0a4_nlp_training_pool_is_unique_and_nonselection(self) -> None:
        train = protocol.read_jsonl(ROOT / "data/distill/p0a4_train.jsonl")
        selection = protocol.read_jsonl(
            ROOT / "data/distill/p0a2_recovery_validation.jsonl"
        )
        nlp = [row for row in train if row.get("dataset_key") == "cmmlu"]
        prompt_hashes = {
            protocol.sha256_text(
                json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
            )
            for row in nlp
        }
        selection_groups = {
            protocol.group_id(row)
            for row in selection
            if row.get("dataset_key") == "cmmlu"
        }
        self.assertEqual(len(nlp), 512)
        self.assertEqual(len(prompt_hashes), 512)
        self.assertFalse({protocol.group_id(row) for row in nlp} & selection_groups)
        self.assertTrue(
            {row["source"] for row in nlp}
            <= {
                "cmmlu_official_dev_nonselection_train",
                "mmlu_auxiliary_nonformal_train",
            }
        )
        self.assertFalse(any("cmmlu/test/" in protocol.group_id(row) for row in nlp))

    def test_imatrix_sampling_is_balanced_by_task(self) -> None:
        rows = [
            {"dataset_key": task, "sample_id": f"{task}/{index}"}
            for task, count in (("gsm8k", 20), ("humaneval", 8), ("cmmlu", 12))
            for index in range(count)
        ]
        selected, counts = imatrix.select_rows(
            rows,
            random.Random(202606),
            rows_per_source=256,
            stratify_key="dataset_key",
            strata=("gsm8k", "humaneval", "cmmlu"),
            rows_per_stratum=8,
        )
        self.assertEqual(counts, {"gsm8k": 8, "humaneval": 8, "cmmlu": 8})
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({row["sample_id"] for row in selected}), 24)

        with self.assertRaisesRegex(ValueError, "requires 9"):
            imatrix.select_rows(
                rows,
                random.Random(202606),
                rows_per_source=256,
                stratify_key="dataset_key",
                strata=("humaneval",),
                rows_per_stratum=9,
            )

    def test_student_v2_sampling_is_explicit_and_bounded(self) -> None:
        rows = [
            {
                "dataset_key": task,
                "sample_id": f"{task}/{index}",
                "messages": [{"role": "user", "content": str(index)}],
                "answer": str(index),
            }
            for task, count in (("gsm8k", 10), ("humaneval", 4), ("cmmlu", 8))
            for index in range(count)
        ]
        balanced = trainer.balance_rows(
            rows,
            202608,
            max_upsample_factor=4.0,
            target_rows_by_task={"gsm8k": 6, "humaneval": 16, "cmmlu": 12},
        )
        self.assertEqual(
            {task: sum(row["dataset_key"] == task for row in balanced) for task in trainer.TASKS},
            {"gsm8k": 6, "humaneval": 16, "cmmlu": 12},
        )
        with self.assertRaisesRegex(trainer.TrainingError, "exceeds max_upsample_factor"):
            trainer.balance_rows(
                rows,
                202608,
                max_upsample_factor=2.0,
                target_rows_by_task={"gsm8k": 6, "humaneval": 9, "cmmlu": 12},
            )

    def test_peft_adapter_scale_updates_each_lora_branch(self) -> None:
        class Layer:
            def __init__(self, scaling=None):
                if scaling is not None:
                    self.scaling = scaling

        class Model:
            def __init__(self):
                self.layers = [Layer({"default": 2.0}), Layer(), Layer({"default": 1.5})]

            def modules(self):
                return iter(self.layers)

        model = Model()
        self.assertEqual(edge_eval.apply_peft_adapter_scale(model, 0.25), 2)
        self.assertEqual(model.layers[0].scaling["default"], 0.5)
        self.assertEqual(model.layers[2].scaling["default"], 0.375)

    def test_retention_requires_each_task_macro_identity_and_no_errors(self) -> None:
        baseline = []
        candidate = []
        for task in retention.DATASET_TO_RATIO:
            for index in range(4):
                sample_id = f"{task}/{index}"
                baseline.append(
                    {"sample_id": sample_id, "dataset_key": task, "correct": True, "generation_error": ""}
                )
                candidate.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": task,
                        "correct": not (task == "humaneval" and index == 0),
                        "generation_error": "",
                    }
                )
        result = retention.compare(
            baseline,
            candidate,
            {task: 4 for task in retention.DATASET_TO_RATIO},
            0.8,
            0.8,
            0,
        )
        self.assertFalse(result["passed"])
        self.assertIn("code_ratio", result["ratio_failures"])

    def test_trial_ledger_limits_selection_and_formal_attempts(self) -> None:
        config = {
            "gates": {
                "selection170": {"max_student_versions": 2},
                "official_full": {"student_attempts": 1},
            }
        }
        ledger = {"trials": []}
        trials.reserve(ledger, config, "selection170", "v1")
        trials.reserve(ledger, config, "selection170", "v2")
        with self.assertRaises(trials.TrialError):
            trials.reserve(ledger, config, "selection170", "v3")
        formal = {"trials": []}
        trials.reserve(formal, config, "official_full_student", "v1")
        trials.reserve(formal, config, "official_full_student", "v1", resume_reserved=True)
        self.assertIn("last_resume_ts", formal["trials"][0])
        with self.assertRaises(trials.TrialError):
            trials.reserve(formal, config, "official_full_student", "v2")

    def test_trainer_rejects_formal_identity_even_outside_sealed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "gsm8k/test/0",
                        "dataset_key": "gsm8k",
                        "messages": [{"role": "user", "content": "x"}],
                        "answer": "1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(trainer.TrainingError):
                trainer.read_jsonl(path)

    def test_teacher_zero3_keeps_parameters_and_optimizer_on_gpu(self) -> None:
        path = ROOT / "configs/deepspeed_p0a4_zero3.json"
        config = trainer.validate_gpu_zero3_config(path)
        zero = config["zero_optimization"]
        self.assertEqual(zero["stage"], 3)
        self.assertNotIn("offload_optimizer", zero)
        self.assertNotIn("offload_param", zero)

        source = (ROOT / "model_compression/train_p0a4_lora.py").read_text(encoding="utf-8")
        self.assertIn('optim="adamw_torch"', source)
        self.assertIn('label_names=["labels"]', source)
        self.assertIn('ddp_find_unused_parameters=False', source)
        self.assertIn('gradient_checkpointing_kwargs={"use_reentrant": False}', source)
        self.assertLess(
            source.index("training_args = TrainingArguments"),
            source.index("model = AutoModelForCausalLM.from_pretrained"),
        )
        self.assertIn("is_deepspeed_zero3_enabled", source)

        launcher_source = (ROOT / "scripts/run_p0a4.sh").read_text(encoding="utf-8")
        self.assertIn("require_student_training_ready", launcher_source)
        self.assertIn("require_student_merge_ready", launcher_source)
        self.assertIn("require_quantized_student_ready", launcher_source)
        self.assertIn("require_student_eval_runtime", launcher_source)
        self.assertIn("--stratify-key dataset_key", launcher_source)
        self.assertIn("--rows-per-stratum 128", launcher_source)
        self.assertIn("--disable-thinking --kv-cache-type q8_0", launcher_source)
        self.assertIn('p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96_retention.json', launcher_source)

    def test_teacher_zero3_rejects_cpu_offload_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deepspeed.json"
            path.write_text(
                json.dumps(
                    {
                        "zero_optimization": {
                            "stage": 3,
                            "offload_optimizer": {"device": "cpu"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(trainer.TrainingError, "forbids CPU offload"):
                trainer.validate_gpu_zero3_config(path)

    def test_best_student_adapter_is_published_without_ddp_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-1"
            output = root / "published"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

            published = trainer.publish_best_adapter_checkpoint(checkpoint, output)

            self.assertEqual(
                published, ["adapter_config.json", "adapter_model.safetensors"]
            )
            self.assertEqual((output / "adapter_model.safetensors").read_bytes(), b"adapter")

        source = (ROOT / "model_compression/train_p0a4_lora.py").read_text(encoding="utf-8")
        self.assertIn('load_best_model_in_trainer = args.role == "teacher"', source)

    def test_teacher_v2_limits_code_repetition_and_uses_conservative_lora(self) -> None:
        rows = [
            {"dataset_key": task, "sample_id": f"{task}/{index}"}
            for task, count in (("gsm8k", 8), ("humaneval", 2), ("cmmlu", 6))
            for index in range(count)
        ]
        balanced = trainer.balance_rows(rows, seed=202606, max_upsample_factor=2.0)
        counts = {
            task: sum(row["dataset_key"] == task for row in balanced)
            for task in trainer.TASKS
        }
        self.assertEqual(counts, {task: 4 for task in trainer.TASKS})

        config = json.loads(
            (ROOT / "configs/p0a4_distillation.json").read_text(encoding="utf-8")
        )
        v2 = config["training"]["teacher"]["candidate_overrides"]["2"]
        self.assertEqual(v2["epochs"], 1)
        self.assertLess(v2["learning_rate"], config["training"]["teacher"]["learning_rate"])
        self.assertEqual(v2["max_upsample_factor"], 2.0)

    def test_evaluator_supports_explicit_task_model_routing(self) -> None:
        parsed = edge_eval.parse_model_id_map(
            ["gsm8k=base,humaneval=base", "cmmlu=teacher-v1"]
        )
        self.assertEqual(
            parsed,
            {"gsm8k": "base", "humaneval": "base", "cmmlu": "teacher-v1"},
        )
        self.assertEqual(edge_eval.select_served_model_id(["base", "teacher-v1"], "base"), "base")
        with self.assertRaises(edge_eval.RecoveryEvalError):
            edge_eval.select_served_model_id(["base", "teacher-v1"], "auto")

    def test_evaluator_fingerprints_adapter_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            first = edge_eval.artifact_fingerprint(adapter)["key_metadata_hash"]
            weights.write_bytes(b"other")
            second = edge_eval.artifact_fingerprint(adapter)["key_metadata_hash"]
            self.assertNotEqual(first, second)

    def test_vllm_bf16_teacher_enables_four_gpu_lora(self) -> None:
        args = argparse.Namespace(
            vllm_bin=".venv/bin/vllm",
            model_dir="models/pretrained/Qwen--Qwen2.5-14B-Instruct",
            quantization="none",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            host="0.0.0.0",
            disable_log_requests=True,
            lora_module=["teacher=models/checkpoints/p0a4/teacher-v1"],
        )
        spec = launcher.TeacherSpec(
            gpus=("0", "1", "2", "3"), port=8000, tensor_parallel_size=4
        )
        command = launcher.build_command(args, spec)
        self.assertNotIn("--quantization", command)
        self.assertIn("--enable-lora", command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")

    def test_launcher_freezes_q4_q8_and_all_gates(self) -> None:
        script = (ROOT / "scripts/run_p0a4.sh").read_text(encoding="utf-8")
        for command in (
            "baseline-full",
            "build-llama-cuda",
            "teacher-train",
            "teacher-distill",
            "student-smoke96",
            "student-170-check",
            "student-170",
            "student-memory-precheck-q4",
            "student-memory",
            "student-full",
            "student-full-direct",
            "student-full-direct-gpu",
            "student-v2-select-route",
            "student-v2-170",
            "student-v2-memory",
            "student-adapter-memory-precheck",
        ):
            self.assertIn(command, script)
        self.assertIn('EDGE_QUANT_TYPE="Q4_K_M"', script)
        self.assertIn('P0A4_EDGE_GPUS="${P0A4_EDGE_GPUS:-0}"', script)
        self.assertIn('P0A4_EDGE_PARALLEL="${P0A4_EDGE_PARALLEL:-4}"', script)
        self.assertIn("-DGGML_CUDA=ON", script)
        self.assertIn('"gpu_backend":True', script)
        self.assertIn('--quant-type "$EDGE_QUANT_TYPE"', script)
        self.assertIn('EDGE_QUANT_AUDIT="reports/audit/gate_p0a4_student_q4_prepare.json"', script)
        self.assertIn("require_edge_smoke96_ready", script)
        self.assertIn("require_baseline_selection170_ready", script)
        self.assertIn("require_edge_memory_precheck_ready", script)
        self.assertIn("require_edge_service_ready", script)
        self.assertIn('local trial_version="v${version}-${EDGE_QUANT_TAG}"', script)
        self.assertIn("READY: Q4_K_M Student", script)
        self.assertIn('P0A4_ALLOW_UNGATED_FULL=1 student_full', script)
        self.assertIn("trap stop_edge_server EXIT INT TERM", script)
        self.assertIn('STUDENT_MERGED_DIR="models/checkpoints/p0a4/student-shared-v${P0A4_STUDENT_VERSION}-merged"', script)
        self.assertIn("Direct v2 shared 170 is locked", script)
        self.assertIn("require_v2_route_ready", script)
        self.assertIn('"waived_prerequisites":["selection170","memory_20_plus_100"]', script)
        self.assertIn('"$P0A4_EDGE_URL" "$EDGE_GGUF"', script)
        self.assertIn("--cache-type-k q8_0", script)
        self.assertNotIn("--cache-type-k q4", script.lower())
        self.assertIn("--cache-ram 0 --no-cache-idle-slots", script)
        self.assertIn("--host-prompt-cache-mib 0 --no-cache-idle-slots", script)
        self.assertIn("--nproc_per_node=4", script)
        self.assertIn("--endpoint-model-id-map", script)
        self.assertIn("--teacher-model-id-map", script)
        self.assertIn("require_distill_ready", script)
        self.assertIn("--min-accepted-count-map", script)

        config = json.loads((ROOT / "configs/p0a4_distillation.json").read_text(encoding="utf-8"))
        student = config["models"]["student"]
        self.assertEqual(student["quant_type"], "Q4_K_M")
        self.assertEqual(
            student["quantized_gguf"],
            "models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf",
        )
        self.assertEqual(
            config["artifacts"]["student_selection170_retention"],
            "reports/audit/gate_p0a4_edge_student_v1_q4_k_m_selection170_retention.json",
        )
        v2 = config["training"]["student_shared"]["candidate_overrides"]["2"]
        self.assertEqual(
            v2["target_rows_by_task"],
            {"gsm8k": 466, "humaneval": 932, "cmmlu": 880},
        )
        self.assertEqual(config["gates"]["v2_promotion"]["min_ratio_per_task"], 0.85)
        self.assertFalse(config["gates"]["v2_promotion"]["full_test_feedback_used"])

    def test_v2_route_verification_uses_aggregate_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            trace.write_text('{"sample_id":"aggregate-only"}\n', encoding="utf-8")
            trace_hash = v2_route.sha256_file(trace)
            evaluation = root / "eval.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "dataset_counts": {"cmmlu": 32, "gsm8k": 32, "humaneval": 32},
                        "disable_thinking": True,
                        "kv_cache_type": "q8_0",
                        "model": {"sha256": "model-hash"},
                        "output_trace_sha256": trace_hash,
                    }
                ),
                encoding="utf-8",
            )
            retention_path = root / "retention.json"
            retention_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "candidate_trace_hash": trace_hash,
                        "generation_error_count": 0,
                        "ratios": {"math_ratio": 0.9, "code_ratio": 0.86, "nlp_ratio": 0.88},
                        "capped_macro_ratio": 0.88,
                    }
                ),
                encoding="utf-8",
            )
            verified = v2_route.verify_route(
                "shared", evaluation, retention_path, "model-hash"
            )
            self.assertEqual(verified["worst_task_ratio"], 0.86)
            self.assertEqual(verified["capped_macro_ratio"], 0.88)
            self.assertEqual(verified["retention_gate_status"], "passed")

            # A completed route below the ordinary smoke threshold remains valid
            # comparison evidence; the stricter v2 promotion filter rejects it.
            failed_retention = json.loads(retention_path.read_text(encoding="utf-8"))
            failed_retention["status"] = "failed"
            retention_path.write_text(json.dumps(failed_retention), encoding="utf-8")
            failed_verified = v2_route.verify_route(
                "adapter_top1", evaluation, retention_path, "model-hash"
            )
            self.assertEqual(failed_verified["retention_gate_status"], "failed")


if __name__ == "__main__":
    unittest.main()
