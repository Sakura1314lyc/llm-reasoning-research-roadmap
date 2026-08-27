from evaluate_math import evaluate_record, summarize


def test_strict_correct_and_equivalent_fraction() -> None:
    result = evaluate_record({"raw_output": "work... #### 0.5", "gold_answer": "1/2", "generated_tokens": 8})
    assert result["numeric_correct"] and result["strict_correct"]


def test_loose_answer_is_not_strict() -> None:
    result = evaluate_record({"raw_output": "answer is 6 but I continue", "gold_answer": "6", "generated_tokens": 10})
    assert result["numeric_correct"] and not result["strict_correct"]


def test_summary_tracks_truncation_and_length() -> None:
    metrics = summarize([
        {"raw_output": "#### 5", "gold_answer": "5", "generated_tokens": 10},
        {"raw_output": "long 4", "gold_answer": "5", "generated_tokens": 20, "reached_max_tokens": True},
    ])
    assert metrics["numeric_accuracy"] == 0.5
    assert metrics["truncation_rate"] == 0.5
    assert metrics["average_generated_tokens"] == 15
