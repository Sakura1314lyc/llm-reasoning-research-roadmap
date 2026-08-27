"""从多次采样结果中抽取答案并做多数投票。"""

from reasoning_utils import majority_vote


def main() -> None:
    outputs = [
        "12 - 6 = 6. #### 6",
        "She gives 6 away, leaving 6. #### 6",
        "I made an arithmetic mistake. #### 5",
        "The result is 6.",
    ]
    winner, counts = majority_vote(outputs)
    print("计票：", counts)
    print("最终答案：", winner)
    assert winner == "6"


if __name__ == "__main__":
    main()
