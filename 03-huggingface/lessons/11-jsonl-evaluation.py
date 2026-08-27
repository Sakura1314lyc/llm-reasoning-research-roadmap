"""把逐样本预测写成 JSONL，再计算准确率。"""

import argparse
import json
from pathlib import Path


def evaluate(records: list[dict[str, str]]) -> dict[str, float | int]:
    if not records:
        return {"count": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(row["prediction"].strip() == row["reference"].strip() for row in records)
    return {"count": len(records), "correct": correct, "accuracy": correct / len(records)}


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("03-huggingface/outputs/generation-baseline/predictions.jsonl"),
    )
    args = parser.parse_args()

    examples = [
        {"sample_id": "math-001", "prediction": "2", "reference": "2", "raw_output": "答案是 2。"},
        {"sample_id": "math-002", "prediction": "6", "reference": "5", "raw_output": "答案是 6。"},
    ]
    write_jsonl(examples, args.output)
    print(json.dumps(evaluate(examples), ensure_ascii=False, indent=2))
    print("predictions:", args.output)
