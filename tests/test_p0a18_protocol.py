from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p0a18_protocol_is_frozen_to_mbpp_validation() -> None:
    text = (ROOT / "configs/p0a18_code_transfer.json").read_text(encoding="utf-8")
    assert '"split": "validation"' in text
    assert '"rows": 90' in text
    assert '"minimum_absolute_gain": 0.03' in text
    assert '"maximum_candidate_evaluations": 2' in text


def test_p0a18_does_not_train_or_open_formal_data() -> None:
    run = (ROOT / "scripts/run_p0a18.sh").read_text(encoding="utf-8")
    assert "train_p0" not in run
    assert "formal_full_opened':False" in run
    assert "run_with_memory_guard.sh" in run


def test_p0a18_evaluator_has_isolated_protocol() -> None:
    evaluator = (ROOT / "scripts/evaluate_p0a11_domain.py").read_text(encoding="utf-8")
    assert '"p0a18"' in evaluator
    assert '"p0a18_external_validation"' in evaluator
    assert 'expected_tests = 3 if protocol == "p0a18" else 10' in evaluator
    assert 'if protocol == "p0a19"' in evaluator
