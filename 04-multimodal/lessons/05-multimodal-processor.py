"""用 AutoProcessor 把图像和文字整理成同一个 Batch。

首次运行需要下载模型 Processor，不会加载模型权重。
"""

import argparse
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="本地图片路径")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    batch = processor(images=[image], text=["Describe this image."], return_tensors="pt")

    print("Processor：", type(processor).__name__)
    for name, value in batch.items():
        print(name, getattr(value, "shape", type(value).__name__))

    assert "pixel_values" in batch
    assert "input_ids" in batch


if __name__ == "__main__":
    main()
