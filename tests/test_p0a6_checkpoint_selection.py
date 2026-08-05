from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/select_p0a6_checkpoint.py"
    spec = importlib.util.spec_from_file_location("p0a6_checkpoint_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load_module()


def evaluation_audit(
    manifest: Path,
    accuracy: dict[str, float],
    candidate_name: str,
    manifest_hash: str = "",
) -> dict[str, object]:
    if not manifest_hash:
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return {
        "gate": "P0-A6-INTERNAL-EVAL",
        "created_by": "scripts/evaluate_p0a6_internal.py",
        "status": "passed",
        "candidate_name": candidate_name,
        "served_model_id": candidate_name,
        "manifest": str(manifest),
        "manifest_hash": manifest_hash,
        "expected_counts": {"math": 100, "code": 100, "nlp": 100},
        "actual_counts": {"math": 100, "code": 100, "nlp": 100},
        "accuracy_by_domain": accuracy,
        "macro_accuracy": sum(accuracy.values()) / 3,
        "generation_error_count": 0,
    }


class P0A6CheckpointSelectionTests(unittest.TestCase):
    def run_selection(
        self,
        directory: Path,
        candidates: list[tuple[int, dict[str, float]]],
    ) -> tuple[int, dict[str, object]]:
        audit_root = directory / "reports/audit/p0a6"
        audit_root.mkdir(parents=True)
        manifest = directory / "data/p0a6/quick_validation.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        base_path = audit_root / "base.json"
        base_path.write_text(
            json.dumps(
                evaluation_audit(
                    manifest,
                    {"math": 0.80, "code": 0.50, "nlp": 0.60},
                    "base",
                )
            ),
            encoding="utf-8",
        )
        arguments = [
            "select_p0a6_checkpoint.py",
            "--base-audit",
            str(base_path),
        ]
        for step, accuracy in candidates:
            path = audit_root / f"candidate-{step}.json"
            path.write_text(
                json.dumps(evaluation_audit(manifest, accuracy, f"candidate-{step}")),
                encoding="utf-8",
            )
            arguments.extend(["--candidate", f"{step}={path}"])
        output = audit_root / "selection.json"
        arguments.extend(["--output", str(output)])
        with (
            patch.object(selector, "AUDIT_ROOT", audit_root.resolve()),
            patch.object(selector, "QUICK_MANIFEST", manifest.resolve()),
            patch.object(sys, "argv", arguments),
        ):
            result = selector.main()
        return result, json.loads(output.read_text(encoding="utf-8"))

    def test_qualifying_candidate_uses_macro_then_earliest_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report = self.run_selection(
                Path(temp),
                [
                    (200, {"math": 0.79, "code": 0.54, "nlp": 0.64}),
                    (100, {"math": 0.79, "code": 0.54, "nlp": 0.64}),
                    (300, {"math": 0.81, "code": 0.52, "nlp": 0.70}),
                ],
            )
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_step"], 100)
        self.assertEqual(
            report["selected_gain_by_domain"],
            {"math": -0.010000000000000009, "code": 0.040000000000000036, "nlp": 0.040000000000000036},
        )

    def test_no_qualifying_candidate_writes_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report = self.run_selection(
                Path(temp),
                [(100, {"math": 0.78, "code": 0.54, "nlp": 0.64})],
            )
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["selected_step"])
        self.assertEqual(report["eligible_candidate_count"], 0)

    def test_gain_thresholds_are_domain_specific(self) -> None:
        eligible, failures = selector.qualifies(
            {"math": -0.01, "code": 0.03, "nlp": 0.03}
        )
        self.assertTrue(eligible, failures)
        eligible, failures = selector.qualifies(
            {"math": -0.011, "code": 0.20, "nlp": 0.20}
        )
        self.assertFalse(eligible)
        self.assertTrue(any(item.startswith("math_gain") for item in failures))

    def test_candidate_parser_rejects_invalid_step(self) -> None:
        with self.assertRaises(selector.SelectionError):
            selector.parse_candidate("zero=/tmp/audit.json")
        with self.assertRaises(selector.SelectionError):
            selector.parse_candidate("0=/tmp/audit.json")

    def test_full_manifest_can_be_selected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            audit_root = directory / "reports/audit/p0a6"
            audit_root.mkdir(parents=True)
            manifest = directory / "data/p0a6/full_validation.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")
            base_path = audit_root / "base-full.json"
            candidate_path = audit_root / "candidate-full.json"
            base_path.write_text(
                json.dumps(
                    evaluation_audit(
                        manifest,
                        {"math": 0.80, "code": 0.50, "nlp": 0.60},
                        "base",
                    )
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    evaluation_audit(
                        manifest,
                        {"math": 0.79, "code": 0.53, "nlp": 0.63},
                        "candidate",
                    )
                ),
                encoding="utf-8",
            )
            output = audit_root / "selection-full.json"
            arguments = [
                "select_p0a6_checkpoint.py",
                "--base-audit",
                str(base_path),
                "--validation-manifest",
                str(manifest),
                "--candidate",
                f"100={candidate_path}",
                "--output",
                str(output),
            ]
            with (
                patch.object(selector, "AUDIT_ROOT", audit_root.resolve()),
                patch.object(selector, "FULL_MANIFEST", manifest.resolve()),
                patch.object(sys, "argv", arguments),
            ):
                result = selector.main()
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(report["validation_scope"], "full")


if __name__ == "__main__":
    unittest.main()
