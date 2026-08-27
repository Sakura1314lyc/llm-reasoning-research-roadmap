"""构造 Qwen-VL 的图文对话消息，并保留可选生成入口。

默认只展示模板；传入 --run 与图片路径后才加载模型执行生成。
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_messages(image_path: Path, question: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image", "image": str(image_path.resolve())},
        {"type": "text", "text": question},
    ]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--question", default="请描述图片中的主要内容。")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    messages = build_messages(args.image, args.question)
    print("messages：", messages)
    if not args.run:
        print("未传入 --run：只检查多模态消息结构。")
        return

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[Image.open(args.image).convert("RGB")], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=64)
    new_ids = generated[:, inputs["input_ids"].size(1):]
    print(processor.batch_decode(new_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
