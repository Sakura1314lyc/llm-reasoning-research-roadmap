# 阶段 2：Transformer 原理与手写实现

这一阶段从注意力机制开始，逐步实现一个可解释、可测试的简化 Transformer，并为加载真实大语言模型打下基础。

建议用时：**7–10 天，每天 3–4 小时**。

## 前置知识

- 完成 PyTorch Tensor、Autograd、`nn.Module` 和训练流程
- 熟悉矩阵乘法、Softmax 和基本概率
- 能持续跟踪张量形状

## 完成后应该具备的能力

- 解释 Q、K、V 和缩放点积注意力
- 手写 Multi-Head Attention 与 Causal Mask
- 理解 Token Embedding、位置编码和 RoPE
- 实现 Feed-Forward Network 与归一化层
- 组合一个 Pre-Norm Transformer Block
- 实现简化版 Causal Language Model
- 为关键组件编写形状与因果性测试

## 推荐学习顺序

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | 多头注意力 | [01-multi-head-attention.py](01-multi-head-attention.py) | 已完成 Q/K/V、多头拆分合并和因果掩码 |
| 02 | Token Embedding | `02-token-embedding.py` | 输入 Token ID，输出 `[B, T, C]` |
| 03 | 位置编码 | `03-positional-encoding.py` | 实现正弦位置编码并验证不同位置表示不同 |
| 04 | RoPE | `04-rope.py` | 理解旋转维度，并验证形状保持不变 |
| 05 | 归一化 | `05-normalization.py` | 对比 LayerNorm 与 RMSNorm |
| 06 | Feed-Forward Network | `06-feed-forward.py` | 实现 Linear → 激活 → Linear |
| 07 | Transformer Block | `07-transformer-block.py` | 组合 Attention、FFN、残差与归一化 |
| 08 | Causal LM | `08-mini-gpt.py` | 输入 Token，输出词表 logits |
| 09 | 训练实验 | `09-train-mini-gpt.py` | 在小文本上 overfit，验证 loss 能下降 |
| 10 | 生成 | `10-autoregressive-generation.py` | 实现逐 Token 自回归生成 |

## 第一部分：注意力机制

当前脚本已经实现：

- Scaled Dot-Product Attention
- Multi-Head Attention
- Q/K/V 线性投影
- Causal Mask
- Attention/Residual Dropout
- 输入形状和维度校验

从仓库根目录运行：

```bash
python 02-transformer/01-multi-head-attention.py
```

运行后应看到输入输出都为 `[2, 8, 32]`。

## 学习时必须维护“形状账本”

```text
token_ids:       [B, T]
embedding:       [B, T, C]
q/k/v:           [B, H, T, D]
attention score: [B, H, T, T]
attention out:   [B, H, T, D]
merged heads:    [B, T, C]
lm logits:       [B, T, V]
```

每实现一个模块，都在代码注释中写清输入、变换和输出形状。

## 推荐测试

为每个模块至少测试：

- 输入输出形状
- dtype 与 device 是否保持一致
- 错误维度是否给出明确异常
- 因果注意力是否看不到未来 Token
- `eval()` 模式下 Dropout 是否关闭
- 梯度能否正常反向传播

因果性测试可以固定前半段 Token，只修改未来 Token，确认较早位置的输出不受影响。

## 阶段项目：Mini GPT

```text
projects/01-mini-gpt/
├── model.py
├── dataset.py
├── train.py
├── generate.py
├── config.yaml
├── tests/
└── README.md
```

先在很小的字符级或 Token 级数据上训练。目标不是效果，而是验证完整数据流：

1. 文本转换为 Token
2. 输入序列与右移标签
3. Causal LM logits
4. Cross-Entropy Loss
5. 反向传播与参数更新
6. 自回归生成

## 常见误区

- `dim` 必须能被 `n_heads` 整除
- Softmax 应在 key 序列维度上计算
- Mask 的布尔含义要统一，避免 True/False 反转
- `transpose` 后使用 `view` 前通常需要 `contiguous()`
- 位置编码与 Token Embedding 必须形状兼容
- 训练 Causal LM 时要正确右移标签
- 不要急着优化 KV Cache，先保证普通生成正确

## 完成清单

- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention
- [x] Causal Mask
- [ ] Token Embedding
- [ ] Positional Encoding
- [ ] RoPE
- [ ] LayerNorm / RMSNorm
- [ ] Feed-Forward Network
- [ ] Transformer Block
- [ ] Mini GPT 训练与生成
- [ ] 关键组件自动化测试

## 权威资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

完成后进入 [阶段 3：Hugging Face 与大语言模型](../03-huggingface/README.md)。
