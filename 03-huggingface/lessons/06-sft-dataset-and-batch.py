"""Hugging Face 课程 06：SFT Dataset、动态 Padding 与 DataLoader。"""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sft_utils import (
    MODEL_NAME,
    RAW_DATA,
    SFTDataCollator,
    make_preprocess_function
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_list(RAW_DATA)
    tokenized_dataset = dataset.map(
        make_preprocess_function(tokenizer),
        remove_columns=dataset.column_names
    )

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=SFTDataCollator(tokenizer)
    )
    batch = next(iter(data_loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print("input_ids：", input_ids.shape)
    print("attention_mask：", attention_mask.shape)
    print("labels：", labels.shape)
    print("有效 token 数：", attention_mask.sum(dim=1).tolist())
    print("监督 token 数：", (labels != -100).sum(dim=1).tolist())

    padding_positions = attention_mask == 0

    assert input_ids.shape == attention_mask.shape == labels.shape
    assert torch.all(labels[padding_positions] == -100)
    assert torch.all(input_ids[padding_positions] == tokenizer.pad_token_id)


if __name__ == "__main__":
    main()
