"""拿几条内存样本练习 Hugging Face Dataset 的切分与 map。"""

from datasets import Dataset


def add_prompt(example: dict[str, str]) -> dict[str, str]:
    return {"prompt": f"请回答：{example['question']}"}


if __name__ == "__main__":
    dataset = Dataset.from_dict(
        {
            "sample_id": ["math-001", "math-002", "math-003", "math-004"],
            "question": ["1+1=?", "2+3=?", "6/2=?", "3*4=?"],
            "answer": ["2", "5", "3", "12"],
        }
    )
    split = dataset.train_test_split(test_size=0.25, seed=42)
    prepared = split.map(add_prompt)

    assert len(prepared["train"]) == 3
    assert len(prepared["test"]) == 1
    assert {"sample_id", "question", "answer", "prompt"}.issubset(prepared["train"].column_names)
    print(prepared)
    print(prepared["train"][0])
