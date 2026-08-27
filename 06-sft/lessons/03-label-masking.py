"""屏蔽 Prompt 和 Padding，只让回答部分参与 Loss。"""


def build_labels(input_ids: list[int], prompt_length: int, padded_length: int) -> tuple[list[int], list[int], list[int]]:
    if not 0 <= prompt_length <= len(input_ids) <= padded_length:
        raise ValueError("长度关系必须满足 0 <= prompt <= input <= padded")
    padding = padded_length - len(input_ids)
    padded_ids = input_ids + [0] * padding
    attention_mask = [1] * len(input_ids) + [0] * padding
    labels = input_ids.copy()
    labels[:prompt_length] = [-100] * prompt_length
    labels += [-100] * padding
    return padded_ids, attention_mask, labels


def main() -> None:
    input_ids, mask, labels = build_labels([10, 11, 12, 13, 14], prompt_length=3, padded_length=8)
    print("input_ids：", input_ids)
    print("mask：", mask)
    print("labels：", labels)
    assert labels == [-100, -100, -100, 13, 14, -100, -100, -100]


if __name__ == "__main__":
    main()
