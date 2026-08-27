"""固定示例数量与顺序，构造可比较的 Few-shot Prompt。"""


EXAMPLES = [
    ("A box has 4 red and 3 blue balls. Total?", "4 + 3 = 7. #### 7"),
    ("18 cookies are shared by 3 children. Each gets?", "18 / 3 = 6. #### 6"),
]


def build_prompt(question: str, examples: list[tuple[str, str]]) -> str:
    blocks = [f"Question: {q}\nAnswer: {a}" for q, a in examples]
    return "\n\n".join([*blocks, f"Question: {question}\nAnswer:"])


def main() -> None:
    prompt = build_prompt("A shop has 20 pens and sells 8. How many remain?", EXAMPLES)
    print(prompt)
    assert prompt.count("Question:") == 3


if __name__ == "__main__":
    main()
