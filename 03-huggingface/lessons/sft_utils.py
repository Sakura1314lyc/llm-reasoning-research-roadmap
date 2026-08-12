"""课程 04–08 共用的 SFT 数据、预处理和动态 Padding 工具。"""

from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizerBase


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

RAW_DATA = [
    {
        "question": "什么是梯度下降？",
        "answer": "梯度下降是一种沿损失函数负梯度方向更新参数的优化方法。"
    },
    {
        "question": "2 + 3 等于多少？",
        "answer": "2 + 3 = 5。"
    },
    {
        "question": "什么是 Transformer？",
        "answer": "Transformer 是一种以注意力机制为核心的神经网络架构。"
    }
]


def _extract_input_ids(encoded: object) -> list[int]:
    """兼容 Transformers 不同版本的 Chat Template 返回类型。"""
    if isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("单样本预处理意外收到多个序列")
        input_ids = input_ids[0]

    return list(input_ids)


def build_sft_example(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256
) -> dict[str, list[int]]:
    """构造只监督 assistant 回答部分的单条 SFT 样本。"""
    question = example["question"]
    answer = example["answer"]

    prompt_messages = [
        {"role": "user", "content": question}
    ]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True
        )
    )
    input_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False
        )
    )

    # prompt_ids 可能比截断后的完整序列更长，因此要限制屏蔽范围。
    input_ids = input_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = input_ids.copy()
    labels[:prompt_length] = [-100] * prompt_length
    attention_mask = [1] * len(input_ids)

    if not any(label != -100 for label in labels):
        raise ValueError(
            "样本截断后没有保留 assistant 回答，请增大 max_length"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def make_preprocess_function(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256
) -> Callable[[dict[str, str]], dict[str, list[int]]]:
    """为 ``Dataset.map`` 创建可复用的预处理函数。"""

    def preprocess(example: dict[str, str]) -> dict[str, list[int]]:
        return build_sft_example(
            example=example,
            tokenizer=tokenizer,
            max_length=max_length
        )

    return preprocess


class SFTDataCollator:
    """把变长样本右侧补齐，并让 Padding 标签保持为 -100。"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = None
    ) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer 必须定义 pad_token_id")

        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        features: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features 不能为空")

        maximum_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            maximum_length = (
                (maximum_length + multiple - 1) // multiple * multiple
            )

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            padding_length = maximum_length - len(feature["input_ids"])

            batch_input_ids.append(
                feature["input_ids"]
                + [self.pad_token_id] * padding_length
            )
            batch_attention_mask.append(
                feature["attention_mask"]
                + [0] * padding_length
            )
            batch_labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long
            ),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }
