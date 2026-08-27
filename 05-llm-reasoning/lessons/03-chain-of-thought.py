"""只改提示方式，对照直接回答和 Chain-of-Thought。"""


def build_prompts(question: str) -> dict[str, str]:
    return {
        "direct": f"Answer the question. End with #### <answer>.\n{question}",
        "zero_shot_cot": f"Solve step by step, verify the arithmetic, then end with #### <answer>.\n{question}",
    }


def main() -> None:
    prompts = build_prompts("Janet has 12 apples and gives away 6. How many remain?")
    for name, prompt in prompts.items():
        print(f"[{name}]\n{prompt}\n")
    assert set(prompts) == {"direct", "zero_shot_cot"}


if __name__ == "__main__":
    main()
