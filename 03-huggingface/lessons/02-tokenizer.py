"""Hugging Face 课程 02：Tokenizer 编码、解码与批处理。

知识点：Token 切分、Token ID、特殊 Token、Padding、Truncation、
``input_ids``、``attention_mask`` 以及批量解码。
"""

import sys

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SINGLE_TEXT = "我喜欢学习大语言模型"
BATCH_TEXTS = [
    "你好",
    "我喜欢机器学习",
    "人工智能正在快速发展"
]


def main() -> None:
    # 部分字节级 Token 无法由 Windows GBK 控制台直接显示，
    # backslashreplace 会把它们安全地打印为转义序列。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --------------------------------------------------------
    # 1. 单条文本：Token -> ID -> 文本
    # --------------------------------------------------------
    tokens = tokenizer.tokenize(SINGLE_TEXT)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    encoded = tokenizer(
        SINGLE_TEXT,
        return_tensors="pt"
    )
    decoded = tokenizer.decode(
        encoded["input_ids"][0],
        skip_special_tokens=True
    )

    print("模型仓库：", MODEL_NAME)
    print("Tokenizer 类型：", type(tokenizer).__name__)
    print("词表大小：", tokenizer.vocab_size)
    print("\n原始文本：", SINGLE_TEXT)
    print("Tokens：", tokens)
    print("Token IDs：", token_ids)
    print("完整 input_ids：", encoded["input_ids"])
    print("解码结果：", decoded)

    # --------------------------------------------------------
    # 2. 批量文本：Padding 与 Attention Mask
    # --------------------------------------------------------
    batch = tokenizer(
        BATCH_TEXTS,
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt"
    )
    decoded_batch = tokenizer.batch_decode(
        batch["input_ids"],
        skip_special_tokens=True
    )

    print("\n批量 input_ids：\n", batch["input_ids"])
    print("\nattention_mask：\n", batch["attention_mask"])
    print("批量形状：", batch["input_ids"].shape)
    print("批量解码：", decoded_batch)

    print("\n特殊 Token：")
    print("pad_token / id：", tokenizer.pad_token, tokenizer.pad_token_id)
    print("bos_token / id：", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos_token / id：", tokenizer.eos_token, tokenizer.eos_token_id)
    print("padding_side：", tokenizer.padding_side)
    print("truncation_side：", tokenizer.truncation_side)

    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert len(decoded_batch) == len(BATCH_TEXTS)


if __name__ == "__main__":
    main()
