"""对 JSONL 数学预测计算数值、严格格式、截断和长度指标。"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re


STRICT = re.compile(r"####\s*([^\n]+?)\s*$")
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def normalize(value: str) -> Fraction | None:
    matches = NUMBER.findall(value.replace(",", ""))
    if not matches:
        return None
    try:
        return Fraction(matches[-1])
    except (ValueError, ZeroDivisionError):
        return None


def evaluate_record(record: dict) -> dict:
    output = record["raw_output"]
    strict_match = STRICT.search(output.strip())
    strict_answer = normalize(strict_match.group(1)) if strict_match else None
    loose_answer = strict_answer if strict_answer is not None else normalize(output)
    target = normalize(str(record["gold_answer"]))
    return {
        **record,
        "extracted_answer": str(loose_answer) if loose_answer is not None else None,
        "numeric_correct": loose_answer is not None and loose_answer == target,
        "strict_correct": strict_answer is not None and strict_answer == target,
        "format_compliant": strict_answer is not None,
        "truncated": bool(record.get("reached_max_tokens", False)),
        "generated_tokens": int(record.get("generated_tokens", 0)),
    }


def summarize(records: list[dict]) -> dict:
    evaluated = [evaluate_record(record) for record in records]
    count = len(evaluated)
    if count == 0:
        raise ValueError("没有预测记录")
    return {
        "samples": count,
        "numeric_accuracy": sum(r["numeric_correct"] for r in evaluated) / count,
        "strict_accuracy": sum(r["strict_correct"] for r in evaluated) / count,
        "format_compliance": sum(r["format_compliant"] for r in evaluated) / count,
        "truncation_rate": sum(r["truncated"] for r in evaluated) / count,
        "average_generated_tokens": sum(r["generated_tokens"] for r in evaluated) / count,
        "predictions": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = summarize(records)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
