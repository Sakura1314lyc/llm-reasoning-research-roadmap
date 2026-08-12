"""Hugging Face 课程 07：Qwen 全参数 SFT 的最小训练循环。

全参数微调需要为所有模型参数保存梯度和优化器状态，显存/内存占用明显高于
LoRA。默认只训练 1 个 epoch、3 条教学样本，用于理解流程而非获得实用模型。
"""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_utils import (
    MODEL_NAME,
    RAW_DATA,
    SFTDataCollator,
    make_preprocess_function
)


EPOCHS = 1
LEARNING_RATE = 1e-5


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype
    ).to(device)

    dataset = Dataset.from_list(RAW_DATA)
    tokenized_dataset = dataset.map(
        make_preprocess_function(tokenizer),
        remove_columns=dataset.column_names
    )
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=SFTDataCollator(tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for batch in data_loader:
            batch = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f"epoch={epoch}/{EPOCHS} | loss={average_loss:.4f}")

    print("设备 / dtype：", device, model_dtype)
    print("最后一个 batch logits：", outputs.logits.shape)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    main()
