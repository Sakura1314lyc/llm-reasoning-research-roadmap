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
| 02 | Feed-Forward Network | [02-feed-forward-network.py](02-feed-forward-network.py) | 已完成逐 Token FFN 和梯度验证 |
| 03 | 归一化 | [03-normalization.py](03-normalization.py) | 已对比 BatchNorm 与 LayerNorm 的统计维度 |
| 04 | Encoder Layer | [04-encoder-layer.py](04-encoder-layer.py) | 已组合自注意力、FFN、残差与 Pre-Norm |
| 05 | Decoder Layer | [05-decoder-layer.py](05-decoder-layer.py) | 已实现因果自注意力、交叉注意力与 Decoder-only 变体 |
| 06 | 正弦位置编码 | [06-positional-encoding.py](06-positional-encoding.py) | 已用 NumPy 实现并验证位置 0 的编码 |
| 07 | 完整 Transformer | [07-transformer.py](07-transformer.py) | 已串联 Embedding、位置编码、Encoder 和 Decoder |
| 08 | RoPE | [08-rope.py](08-rope.py) | 已旋转 Q/K，并验证形状、范数和梯度 |
| 09 | Causal LM | [09-mini-gpt.py](09-mini-gpt.py) | 已实现 Mini GPT 前向、loss、因果性检查和最小生成 |
| 10 | 训练实验 | `10-train-mini-gpt.py` | 在小文本上 overfit，验证 loss 能下降 |
| 11 | 自回归生成 | `11-autoregressive-generation.py` | 实现逐 Token 生成 |

## 第一部分：注意力机制

当前代码已经实现：

- Scaled Dot-Product Attention
- Multi-Head Attention
- Q/K/V 线性投影
- Causal Mask
- Attention/Residual Dropout
- 输入形状和维度校验
- 逐位置 Feed-Forward Network
- BatchNorm 与 LayerNorm 对比
- Pre-Norm Encoder/Decoder Layer
- 教学版 Encoder-Decoder Transformer
- Rotary Position Embedding
- Decoder-only Mini GPT 与 Causal LM Loss

从仓库根目录运行：

```bash
python 02-transformer/01-multi-head-attention.py
python 02-transformer/02-feed-forward-network.py
python 02-transformer/03-normalization.py
python 02-transformer/04-encoder-layer.py
python 02-transformer/05-decoder-layer.py
python 02-transformer/06-positional-encoding.py
python 02-transformer/07-transformer.py
python 02-transformer/08-rope.py
python 02-transformer/09-mini-gpt.py
```

每个脚本都带有最小示例和形状断言，可按顺序单独运行。

## 第 08 章：RoPE

RoPE 将每两个相邻特征看作二维坐标，并按照 token 的位置旋转 Query 和 Key：

```text
x_even' = x_even × cos(θ) - x_odd × sin(θ)
x_odd'  = x_even × sin(θ) + x_odd × cos(θ)
```

需要记住：

- RoPE 应用于注意力的 Q 和 K，而不是直接加到 token embedding
- 每个注意力头的维度必须是偶数，才能两两配对旋转
- 位置 0 的旋转角为 0，因此向量保持不变
- 旋转不会改变向量范数
- Q 与 K 的点积会自然包含相对位置差

## 第 09 章：Mini GPT

Decoder-only Causal LM 的主要数据流：

```text
token_ids [B, T]
  → Token Embedding [B, T, D]
  → N × Transformer Block
  → RMSNorm
  → LM Head
  → logits [B, T, V]
```

本章实现了：

- QKV 联合投影、多头拆分与因果掩码
- 在每层注意力中应用 RoPE
- RMSNorm、SwiGLU、Pre-Norm 与残差连接
- Token Embedding 与 LM Head 权重共享
- 输入和标签右移一位后的 Cross-Entropy Loss
- 修改未来 token 不影响过去 logits 的因果性测试
- 基于 temperature 与 top-k 的逐 token 采样

随机初始化模型只能用于验证数据流；生成有意义的文本需要在第 10 章加入训练。

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
- [x] Token Embedding
- [x] Positional Encoding
- [x] RoPE
- [x] LayerNorm
- [x] RMSNorm
- [x] Feed-Forward Network
- [x] Encoder / Decoder Block
- [x] Encoder-Decoder Transformer 前向传播
- [x] Mini GPT 前向、Loss 与最小生成
- [ ] Mini GPT 训练实验
- [ ] 关键组件自动化测试

## 权威资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

完成后进入 [阶段 3：Hugging Face 与大语言模型](../03-huggingface/README.md)。
