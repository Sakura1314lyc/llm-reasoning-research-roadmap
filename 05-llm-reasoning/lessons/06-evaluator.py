"""从逐样本输出汇总准确率、有效率和平均生成长度。"""

from reasoning_utils import exact_match, extract_answer


def evaluate(records: list[dict]) -> dict:
    evaluated = []
    for record in records:
        prediction = extract_answer(record["raw_output"])
        evaluated.append({**record, "extracted_answer": prediction, "correct": exact_match(prediction, record["gold_answer"])})
    count = len(evaluated)
    return {
        "accuracy": sum(item["correct"] for item in evaluated) / count,
        "valid_answer_rate": sum(item["extracted_answer"] is not None for item in evaluated) / count,
        "average_output_characters": sum(len(item["raw_output"]) for item in evaluated) / count,
        "predictions": evaluated,
    }


def main() -> None:
    metrics = evaluate([
        {"id": "1", "gold_answer": "6", "raw_output": "12-6=6. #### 6"},
        {"id": "2", "gold_answer": "5", "raw_output": "I cannot solve it."},
    ])
    print(metrics)
    assert metrics["accuracy"] == 0.5


if __name__ == "__main__":
    main()
