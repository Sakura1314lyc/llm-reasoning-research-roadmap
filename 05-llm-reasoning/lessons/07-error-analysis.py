"""把推理失败记录归到固定错误类型中。"""

from collections import Counter


ERROR_TYPES = {"format_error", "arithmetic_error", "reasoning_error", "instruction_error", "context_error", "truncation", "evaluation_error"}


def summarize(labels: list[str]) -> Counter:
    unknown = set(labels) - ERROR_TYPES
    if unknown:
        raise ValueError(f"未知错误类型：{sorted(unknown)}")
    return Counter(labels)


def main() -> None:
    counts = summarize(["arithmetic_error", "reasoning_error", "arithmetic_error", "format_error"])
    print(counts)
    assert counts["arithmetic_error"] == 2


if __name__ == "__main__":
    main()
