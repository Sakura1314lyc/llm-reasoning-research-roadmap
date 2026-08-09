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
| 10 | 训练实验 | [10-train-mini-gpt.py](10-train-mini-gpt.py) | 已实现字符数据、右移标签、AdamW、梯度裁剪和 checkpoint |
| 11 | 自回归生成 | [11-autoregressive-generation.py](11-autoregressive-generation.py) | 已实现 Greedy、Temperature、Top-k、Top-p 与重复惩罚 |

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
- 字符级语言模型训练与 checkpoint
- Greedy、Top-k、Top-p 自回归生成
- 关键组件自动化测试

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
python 02-transformer/10-train-mini-gpt.py
python 02-transformer/11-autoregressive-generation.py
pytest 02-transformer/tests -q
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

核心模型已提取到 [mini_gpt.py](mini_gpt.py)，字符 tokenizer 与采样数据工具位于 [mini_gpt_data.py](mini_gpt_data.py)，后续章节直接复用，避免复制模型代码。

## 第 10 章：训练 Mini GPT

训练脚本使用重复的小诗作为字符级语料，让小模型快速过拟合。完整训练数据流：

```text
原始文本
  → CharacterTokenizer
  → token_ids [N]
  → 随机截取长度 T+1 的片段
  → inputs = tokens[:-1]
  → targets = tokens[1:]
  → Mini GPT logits [B, T, V]
  → Cross-Entropy Loss
  → backward + gradient clipping + AdamW
```

关键知识点：

- Causal LM 的每个位置负责预测下一个 token，因此输入与标签相差一位
- `optimizer.zero_grad(set_to_none=True)` 在每次反向传播前清除旧梯度
- 梯度裁剪可以限制异常大的梯度范数
- AdamW 将权重衰减与梯度更新解耦
- 小数据过拟合是验证模型和训练链路是否正确的有效方法
- checkpoint 同时保存模型配置、参数、tokenizer 词表和训练信息

快速验证可以减少训练步数：

```bash
python 02-transformer/10-train-mini-gpt.py --steps 20 --no-save --no-plot
```

正式运行使用默认 200 步，checkpoint 和 loss 曲线会写入 `02-transformer/outputs/mini-gpt/`。模型权重以 `.pth` 结尾，受仓库 `.gitignore` 保护，不会提交到 GitHub。

## 第 11 章：自回归生成

自回归生成会不断重复“截取上下文 → 前向传播 → 读取最后位置 logits → 选择 token → 追加”的过程。

不同解码策略的作用：

| 策略 | 行为 | 主要特点 |
| --- | --- | --- |
| Greedy | 每次选择概率最大的 token | 稳定、可复现，但容易重复 |
| Temperature | 用温度缩放 logits | 越低越保守，越高越随机 |
| Top-k | 只从概率最高的 k 个 token 中采样 | 限制低质量长尾候选 |
| Top-p | 从累计概率达到 p 的最小集合中采样 | 候选数量可随分布变化 |
| Repetition Penalty | 降低已出现 token 的分数 | 缓解循环重复 |

推荐按这个顺序实验：

```bash
# 1. 先训练并保存 checkpoint
python 02-transformer/10-train-mini-gpt.py

# 2. 同时观察 Greedy 和 Sampling
python 02-transformer/11-autoregressive-generation.py --prompt "春眠"

# 3. 调整采样参数
python 02-transformer/11-autoregressive-generation.py --strategy sample --temperature 0.7 --top-k 8 --top-p 0.9
```

生成时只保留模型允许的最大上下文长度。普通实现会在每一步重复计算过去 token；KV Cache 属于后续性能优化，不影响当前对生成逻辑的理解。

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

## Mini GPT 文件结构

```text
02-transformer/
├── mini_gpt.py                       # 可复用模型
├── mini_gpt_data.py                  # tokenizer、语料和 batch 采样
├── 09-mini-gpt.py                    # 模型结构与因果性验证
├── 10-train-mini-gpt.py              # 训练与 checkpoint
├── 11-autoregressive-generation.py   # 解码与采样策略
├── tests/                            # 自动化测试
└── outputs/mini-gpt/                 # 本地训练产物
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
- 采样前必须只读取最后一个位置的 logits
- Temperature 不能为 0，Top-p 必须位于 `(0, 1]`
- 加载模型时必须恢复训练时使用的同一份 tokenizer 词表
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
- [x] Mini GPT 训练实验
- [x] Greedy / Temperature / Top-k / Top-p 生成
- [x] 关键组件自动化测试

## 阶段完成标准

完成本目录后，应该能够独立回答并演示：

- 为什么注意力分数要除以 `sqrt(head_dim)`
- Causal Mask 如何阻止信息从未来泄漏到过去
- LayerNorm、RMSNorm、残差连接和 Pre-Norm 分别解决什么问题
- 正弦位置编码与 RoPE 的使用位置有何不同
- Encoder-Decoder 与 Decoder-only 模型的结构差异
- Causal LM 的输入、右移标签、logits 与 loss 如何对齐
- 训练模式与生成模式的数据流有何不同
- Greedy、Top-k 和 Top-p 会怎样改变生成结果

## 权威资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

完成后进入 [阶段 3：Hugging Face 与大语言模型](../03-huggingface/README.md)。
