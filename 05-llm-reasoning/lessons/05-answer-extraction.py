"""抽取整数、负数、小数、分数和严格结尾答案。"""

from reasoning_utils import extract_answer


def main() -> None:
    cases = {
        "Reasoning... #### 1,200": "1200",
        "Final: -2.5": "-5/2",
        "The fraction is 3/4": "3/4",
        "no numeric answer": None,
    }
    for text, expected in cases.items():
        actual = extract_answer(text)
        print(repr(text), "->", actual)
        assert actual == expected


if __name__ == "__main__":
    main()
