"""Transformer 课程 09：检查 Decoder-only Causal Language Model。

核心模型位于同目录的 ``mini_gpt.py``，便于后续训练和生成章节复用。
本入口验证 logits、Causal LM Loss、反向传播、因果性与最小生成流程。
"""

from mini_gpt import main


if __name__ == "__main__":
    main()
