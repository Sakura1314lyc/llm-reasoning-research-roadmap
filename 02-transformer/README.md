# Transformer 原理与手写实现

本阶段从注意力机制开始，逐步实现一个可解释、可测试的简化 Transformer。

## 当前内容

| 序号 | 主题 | 核心内容 | 文件 |
| ---: | --- | --- | --- |
| 01 | 多头注意力 | 缩放点积注意力、多头拆分、因果掩码、Dropout | [01-multi-head-attention.py](01-multi-head-attention.py) |

## 计划顺序

- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention
- [x] Causal Mask
- [ ] Token Embedding
- [ ] Positional Encoding / RoPE
- [ ] Feed-Forward Network
- [ ] RMSNorm / LayerNorm
- [ ] Transformer Block
- [ ] 简化版 GPT

## 运行说明

脚本末尾包含一个最小形状检查，可以从仓库根目录直接运行：

```bash
python 02-transformer/01-multi-head-attention.py
```

后续课程会把可复用组件整理成正常的 Python 包。
