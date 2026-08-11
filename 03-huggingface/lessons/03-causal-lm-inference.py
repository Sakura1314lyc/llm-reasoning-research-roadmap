"""Hugging Face 课程 03：Causal LM 前向传播与文本生成。

本课使用基础模型而不是 Instruct 模型，直接输入续写 Prompt。先手动执行一次
前向传播并检查 logits，再用 ``generate()`` 做确定性的 Greedy Decoding。
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "人工智能的发展将会"
MAX_NEW_TOKENS = 30


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype
    ).to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    inputs = {
        name: tensor.to(device)
        for name, tensor in encoded.items()
    }

    # --------------------------------------------------------
    # 1. 手动前向：每个输入位置都会输出整个词表的 logits
    # --------------------------------------------------------
    with torch.inference_mode():
        forward_output = model(**inputs)

    logits = forward_output.logits
    next_token_logits = logits[:, -1, :]
    top_values, top_indices = torch.topk(next_token_logits, k=5, dim=-1)

    print("模型：", MODEL_NAME)
    print("设备 / dtype：", device, model_dtype)
    print("input_ids 形状：", inputs["input_ids"].shape)
    print("logits 形状：", logits.shape)
    print("最后位置 Top-5 候选：")

    for score, token_id in zip(
        top_values[0].tolist(),
        top_indices[0].tolist()
    ):
        token_text = tokenizer.decode([token_id])
        print(f"  token_id={token_id:6d} | logit={score:8.3f} | {token_text!r}")

    # --------------------------------------------------------
    # 2. Greedy Decoding：每一步选择概率最大的下一个 token
    # --------------------------------------------------------
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].size(1)
    new_token_ids = generated[0, prompt_length:]
    continuation = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True
    )

    print("\nPrompt：", PROMPT)
    print("新生成 token 数：", new_token_ids.numel())
    print("续写结果：", continuation)
    print("完整文本：", PROMPT + continuation)

    assert logits.shape[:2] == inputs["input_ids"].shape
    assert logits.size(-1) == model.config.vocab_size
    assert generated.size(1) >= prompt_length


if __name__ == "__main__":
    main()
