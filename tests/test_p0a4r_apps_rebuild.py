from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "model_compression" / "rebuild_p0a4r_apps_data.py"
SPEC = importlib.util.spec_from_file_location("p0a4r_apps_rebuild", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
apps = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = apps
SPEC.loader.exec_module(apps)


class P0A4RAppsRebuildTests(unittest.TestCase):
    def test_archive_member_accepts_train_only(self) -> None:
        self.assertTrue(apps.safe_archive_member("APPS/train/0001/question.txt"))
        self.assertFalse(apps.safe_archive_member("APPS/test/0001/question.txt"))
        self.assertFalse(apps.safe_archive_member("../../etc/passwd"))

    def test_dangerous_solution_is_rejected(self) -> None:
        self.assertTrue(apps.safe_source("def add(a, b):\n    return a + b\n")[0])
        safe, reason = apps.safe_source(
            "import os\ndef add(a, b):\n    os.system('id')\n    return a+b\n"
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "forbidden_import")

    def test_explicit_group_exclusions_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "excluded.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"validation_group_id": "apps/task/1"}),
                        json.dumps({"validation_group_id": "apps/task/2"}),
                        json.dumps({"validation_group_id": "apps/task/1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                apps.load_excluded_groups([path]),
                {"apps/task/1", "apps/task/2"},
            )

    def test_call_based_task_builds_and_executes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary) / "0001"
            task.mkdir()
            (task / "metadata.json").write_text(
                json.dumps({"difficulty": "introductory", "url": "train-only"}),
                encoding="utf-8",
            )
            (task / "input_output.json").write_text(
                json.dumps(
                    {
                        "fn_name": "add",
                        "inputs": [[1, 2], [-1, 3]],
                        "outputs": [3, 2],
                    }
                ),
                encoding="utf-8",
            )
            (task / "solutions.json").write_text(
                json.dumps(["def add(a, b):\n    return a + b\n"]),
                encoding="utf-8",
            )
            (task / "question.txt").write_text("Return the sum.", encoding="utf-8")
            self.assertEqual(apps.train_task_count(task.parent), 1)
            self.assertEqual(apps.complete_task_count(task.parent), 1)
            candidate = apps.apps_descriptor(task, 10000)
            self.assertIsNotNone(candidate)
            assert candidate is not None
            row, reason = apps.build_apps_row(candidate, 3, 1, 1000, 1000, 2000)
            self.assertEqual(reason, "")
            assert row is not None
            self.assertEqual(row["validation_group_id"], "apps/task/0001")
            self.assertTrue(row["used_for_training"])
            self.assertEqual(row["code_eval"]["kind"], "apps_call_tests_v1")
            self.assertTrue(apps.code_fingerprint(row))

    def test_launcher_exposes_rebuild_source(self) -> None:
        launcher = (ROOT / "scripts" / "run_p0a4r.sh").read_text(encoding="utf-8")
        self.assertIn("code-source-rebuild", launcher)
        config = json.loads(
            (ROOT / "configs" / "p0a4_remediation.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "data/distill/p0a4r_apps_verified_train.jsonl",
            config["data"]["code_optional_sources"],
        )


if __name__ == "__main__":
    unittest.main()
