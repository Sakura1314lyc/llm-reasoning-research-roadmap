import pytest

from compare_methods import compare_predictions, exact_mcnemar_pvalue


def record(sample_id: str, output: str, gold: str = "1") -> dict:
    return {"sample_id": sample_id, "raw_output": output, "gold_answer": gold}


def test_pairwise_transitions() -> None:
    before = [record("a", "#### 0"), record("b", "#### 1"), record("c", "#### 1")]
    after = [record("a", "#### 1"), record("b", "#### 0"), record("c", "#### 1")]
    result = compare_predictions(before, after)
    assert result["transitions"] == {
        "wrong_to_wrong": 0,
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "correct_to_correct": 1,
    }
    assert result["mcnemar_exact_pvalue"] == 1.0


def test_mcnemar_detects_one_sided_changes() -> None:
    assert exact_mcnemar_pvalue(0, 6) == pytest.approx(0.03125)


def test_rejects_mismatched_sample_sets() -> None:
    with pytest.raises(ValueError, match="样本集合不一致"):
        compare_predictions([record("a", "#### 1")], [record("b", "#### 1")])
