"""对两个方法的逐题预测做配对 transition 与精确 McNemar 检验。"""

import argparse
from math import comb
import json
from pathlib import Path

from evaluate_math import evaluate_record


def index_records(records: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("每条记录都必须包含 sample_id")
        if sample_id in indexed:
            raise ValueError(f"sample_id 重复：{sample_id}")
        indexed[sample_id] = evaluate_record(record)
    return indexed


def exact_mcnemar_pvalue(correct_to_wrong: int, wrong_to_correct: int) -> float:
    """双侧精确二项检验；只使用发生变化的配对样本。"""
    discordant = correct_to_wrong + wrong_to_correct
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(correct_to_wrong, wrong_to_correct) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_predictions(before: list[dict], after: list[dict]) -> dict:
    if not before or not after:
        raise ValueError("配对比较至少需要一条预测记录")
    before_by_id = index_records(before)
    after_by_id = index_records(after)
    if before_by_id.keys() != after_by_id.keys():
        missing_after = sorted(before_by_id.keys() - after_by_id.keys())
        missing_before = sorted(after_by_id.keys() - before_by_id.keys())
        raise ValueError(f"样本集合不一致：after 缺少 {missing_after}；before 缺少 {missing_before}")

    transitions = {"wrong_to_wrong": 0, "wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0}
    rows = []
    for sample_id in sorted(before_by_id):
        before_correct = bool(before_by_id[sample_id]["numeric_correct"])
        after_correct = bool(after_by_id[sample_id]["numeric_correct"])
        transition = f"{'correct' if before_correct else 'wrong'}_to_{'correct' if after_correct else 'wrong'}"
        transitions[transition] += 1
        rows.append({"sample_id": sample_id, "transition": transition})

    count = len(rows)
    return {
        "samples": count,
        "before_accuracy": (transitions["correct_to_wrong"] + transitions["correct_to_correct"]) / count,
        "after_accuracy": (transitions["wrong_to_correct"] + transitions["correct_to_correct"]) / count,
        "transitions": transitions,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(
            transitions["correct_to_wrong"], transitions["wrong_to_correct"]
        ),
        "paired_records": rows,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_predictions(read_jsonl(args.before), read_jsonl(args.after))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
