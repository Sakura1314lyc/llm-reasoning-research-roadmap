"""把 OPD 的任务质量、输出长度与吞吐放在一起比较。"""


def scorecard(name: str, correct: int, samples: int, invalid: int, tokens: int, seconds: float) -> dict:
    return {"method": name, "accuracy": correct / samples, "invalid_rate": invalid / samples, "tokens_per_sample": tokens / samples, "samples_per_second": samples / seconds}


def main() -> None:
    sft = scorecard("SFT", 80, 100, 2, 10000, 50)
    opd = scorecard("OPD", 82, 100, 6, 18000, 120)
    print(sft)
    print(opd)
    assert opd["accuracy"] > sft["accuracy"] and opd["samples_per_second"] < sft["samples_per_second"]


if __name__ == "__main__":
    main()
