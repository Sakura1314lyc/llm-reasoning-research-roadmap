"""Hugging Face 课程 05：把问答样本转换成 SFT 模型输入。"""

from datasets import Dataset
from transformers import AutoTokenizer

from sft_utils import MODEL_NAME, RAW_DATA, make_preprocess_function


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_list(RAW_DATA)
    tokenized_dataset = dataset.map(
        make_preprocess_function(tokenizer),
        remove_columns=dataset.column_names
    )

    sample = tokenized_dataset[0]
    supervised_ids = [
        token_id
        for token_id, label in zip(sample["input_ids"], sample["labels"])
        if label != -100
    ]

    print("原始字段：", dataset.column_names)
    print("处理后字段：", tokenized_dataset.column_names)
    print("input_ids 长度：", len(sample["input_ids"]))
    print("attention_mask 长度：", len(sample["attention_mask"]))
    print("labels 长度：", len(sample["labels"]))
    print(
        "监督文本：",
        tokenizer.decode(supervised_ids, skip_special_tokens=True)
    )

    assert len(sample["input_ids"]) == len(sample["attention_mask"])
    assert len(sample["input_ids"]) == len(sample["labels"])
    assert any(label == -100 for label in sample["labels"])
    assert any(label != -100 for label in sample["labels"])


if __name__ == "__main__":
    main()
