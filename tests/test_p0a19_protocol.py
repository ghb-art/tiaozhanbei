import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p0a19_has_only_two_registered_checkpoints() -> None:
    config = json.loads(
        (ROOT / "configs/p0a19_code_mixed_distill.json").read_text(encoding="utf-8")
    )
    assert config["training"]["checkpoints"] == [128, 256]
    assert config["validation"]["maximum_candidate_evaluations"] == 2
    assert config["validation"]["minimum_absolute_gain"] == 0.03


def test_p0a19_freezes_math_and_nlp() -> None:
    config = json.loads(
        (ROOT / "configs/p0a19_code_mixed_distill.json").read_text(encoding="utf-8")
    )
    assert set(config["frozen_domains"]) == {"math", "nlp"}
    run = (ROOT / "scripts/run_p0a19.sh").read_text(encoding="utf-8")
    assert "--focus-domain code" in run
    assert "run_with_memory_guard.sh" in run


def test_p0a19_validation_cannot_flow_into_training() -> None:
    builder = (ROOT / "model_compression/build_p0a19_data.py").read_text(
        encoding="utf-8"
    )
    assert "sanitized_descriptions" in builder
    assert "train_validation_prompt_overlap" in builder
    assert '"reference_code_written_to_manifest": False' in builder
