"""Hugging Face 快速入门：用 Pipeline 完成一次情感分类。"""

import torch
from transformers import pipeline


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
TEXT = "I really like learning machine learning."


def main() -> None:
    # Pipeline 封装了预处理、模型前向和后处理，适合先快速跑通任务。
    classifier = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )

    result = classifier(TEXT)

    print("模型：", MODEL_NAME)
    print("输入：", TEXT)
    print("输出：", result)

    assert len(result) == 1
    assert {"label", "score"} <= result[0].keys()


if __name__ == "__main__":
    main()
