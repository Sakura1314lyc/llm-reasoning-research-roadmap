"""Hugging Face 课程 08：使用 PEFT LoRA 微调 Qwen。

LoRA 冻结基础模型，只训练插入注意力投影层的低秩矩阵。默认训练 3 个 epoch，
并将 adapter 写入 ``03-huggingface/outputs/qwen2.5-0.5b-lora``。
"""

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_utils import (
    MODEL_NAME,
    RAW_DATA,
    SFTDataCollator,
    make_preprocess_function
)


EPOCHS = 3
LEARNING_RATE = 1e-4
MODULE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = MODULE_ROOT / "outputs" / "qwen2.5-0.5b-lora"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("设备 / dtype：", device, model_dtype)
    print("可训练参数：", trainable_parameters)
    print("总参数：", total_parameters)
    print("可训练比例：", f"{trainable_parameters / total_parameters:.4%}")

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

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE
    )
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

        print(
            f"epoch={epoch}/{EPOCHS} | "
            f"loss={total_loss / len(data_loader):.4f}"
        )

    if device.type == "cuda":
        print(
            "峰值显存：",
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GB"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA adapter：", OUTPUT_DIR.resolve())

    assert trainable_parameters < total_parameters
    assert torch.isfinite(loss)


if __name__ == "__main__":
    main()
