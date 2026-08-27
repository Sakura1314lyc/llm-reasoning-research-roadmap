"""按人工标注的错误类型汇总预测记录。"""

from collections import Counter


def summarize_errors(records: list[dict[str, str | bool]]) -> Counter[str]:
    """只统计错误样本；真实项目应由人工复核 error_type。"""
    return Counter(str(row.get("error_type", "unclassified")) for row in records if not row["correct"])


if __name__ == "__main__":
    predictions = [
        {"sample_id": "1", "correct": True, "error_type": "none"},
        {"sample_id": "2", "correct": False, "error_type": "calculation"},
        {"sample_id": "3", "correct": False, "error_type": "format"},
        {"sample_id": "4", "correct": False, "error_type": "calculation"},
    ]
    summary = summarize_errors(predictions)
    assert summary == Counter({"calculation": 2, "format": 1})
    print(summary)
