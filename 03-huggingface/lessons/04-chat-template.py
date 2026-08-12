"""Hugging Face 课程 04：Chat Template 与 SFT Label Mask。"""

from transformers import AutoTokenizer

from sft_utils import MODEL_NAME, build_sft_example


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_messages = [
        {"role": "user", "content": "解释一下梯度下降。"}
    ]
    full_messages = [
        *prompt_messages,
        {
            "role": "assistant",
            "content": "梯度下降通过负梯度方向更新参数来降低损失。"
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    sample = build_sft_example(
        {
            "question": "解释一下梯度下降。",
            "answer": "梯度下降通过负梯度方向更新参数来降低损失。"
        },
        tokenizer
    )

    labels = sample["labels"]
    first_supervised_position = next(
        index
        for index, label in enumerate(labels)
        if label != -100
    )
    supervised_ids = [
        label
        for label in labels
        if label != -100
    ]

    print("推理 Prompt 模板：\n", prompt_text)
    print("\n完整 SFT 模板：\n", full_text)
    print("input_ids 长度：", len(sample["input_ids"]))
    print("首个监督位置：", first_supervised_position)
    print(
        "被监督的 assistant 文本：",
        tokenizer.decode(supervised_ids, skip_special_tokens=True)
    )

    assert labels[:first_supervised_position] == [-100] * first_supervised_position
    assert first_supervised_position < len(labels)


if __name__ == "__main__":
    main()
