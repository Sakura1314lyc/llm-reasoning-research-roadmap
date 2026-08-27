"""保存多模态逐样本结果，并按错误类型汇总。"""

from collections import Counter


VALID_ERROR_TYPES = {"correct", "perception", "ocr", "grounding", "reasoning", "calculation", "format"}


def summarize(records: list[dict]) -> dict:
    for record in records:
        if record["error_type"] not in VALID_ERROR_TYPES:
            raise ValueError(f"未知错误类型：{record['error_type']}")
    counts = Counter(record["error_type"] for record in records)
    correct = counts["correct"]
    return {"samples": len(records), "accuracy": correct / len(records) if records else 0.0, "error_counts": dict(counts)}


def main() -> None:
    records = [
        {"id": "caption-001", "image": "a.png", "question": "图中有什么？", "prediction": "猫", "answer": "猫", "error_type": "correct"},
        {"id": "ocr-001", "image": "b.png", "question": "数字是多少？", "prediction": "18", "answer": "16", "error_type": "ocr"},
        {"id": "math-001", "image": "c.png", "question": "合计多少？", "prediction": "7", "answer": "8", "error_type": "calculation"},
    ]
    metrics = summarize(records)
    print(metrics)
    assert metrics["accuracy"] == 1 / 3


if __name__ == "__main__":
    main()
