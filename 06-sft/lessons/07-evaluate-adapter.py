"""按相同样本汇总 Base 与 SFT 的配对得失分。"""


def compare(base: list[bool], sft: list[bool]) -> dict:
    if len(base) != len(sft) or not base:
        raise ValueError("两组必须针对同一批非空样本")
    gained = sum(not before and after for before, after in zip(base, sft))
    lost = sum(before and not after for before, after in zip(base, sft))
    return {"base_accuracy": sum(base) / len(base), "sft_accuracy": sum(sft) / len(sft), "gained": gained, "lost": lost}


def main() -> None:
    metrics = compare([True, False, True, False], [True, True, False, True])
    print(metrics)
    assert metrics["gained"] == 2 and metrics["lost"] == 1


if __name__ == "__main__":
    main()
