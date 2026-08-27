# BERT、GPT、LLaMA 与 Qwen 架构对照

前面手写的 Transformer，到了真实模型里会分成不同路线。这里先看结构和训练目标；层数、隐藏维度之类的配置会随版本变化，不必硬背。

| 模型族 | 主体结构 | 预训练目标 | 双向/因果 | 常见用途 |
| --- | --- | --- | --- | --- |
| BERT | Encoder-only | Masked Language Modeling | 双向 | 分类、抽取、编码表示 |
| GPT | Decoder-only | Next-token Prediction | 因果 | 生成、对话、推理 |
| LLaMA | Decoder-only | Next-token Prediction | 因果 | 开源通用基座与后训练 |
| Qwen | Decoder-only 为主 | Next-token Prediction + 后训练 | 因果 | 多语言、代码、数学、多模态 |

## Encoder-only：BERT

BERT 的每个 Token 能同时关注左右上下文，因此适合理解型任务。预训练时随机遮挡部分 Token 并预测它们。分类任务常读取 `[CLS]` 表示，Token 分类则读取每个位置的表示。它不是按自回归方式逐 Token 生成长文本的首选。

## Decoder-only：GPT、LLaMA、Qwen

Decoder-only 模型用 Causal Mask 保证位置 `t` 只能读取 `≤t` 的信息，再预测位置 `t+1`。训练和生成使用同一个 next-token prediction 目标，便于扩展到统一的文本接口。

LLaMA/Qwen 等现代模型常见改进包括：

- RoPE 位置编码；
- RMSNorm 与 Pre-Norm；
- SwiGLU 门控前馈层；
- Grouped-Query Attention（GQA）；
- KV Cache 加速生成；
- 在基座模型上继续进行 SFT、偏好优化或强化学习。

## Base 和 Instruct 别混用

- Base：完成大规模预训练，擅长续写，但未必遵循对话指令。
- Instruct/Chat：在 Base 上经过指令数据和偏好/强化学习后训练，配有特定 Chat Template。

实验里要写完整仓库名，例如 `Qwen/Qwen2.5-0.5B` 或 `Qwen/Qwen2.5-0.5B-Instruct`。两者的输入格式也要和各自的 Chat Template 对上。

## Encoder-Decoder

Encoder 先对完整输入编码，Decoder 再通过 Cross-Attention 条件生成输出。翻译、摘要和“输入到输出”任务常使用这一结构。与 Decoder-only 拼接输入输出相比，它显式区分 source 与 target。

## 合上笔记后试着回答

1. 为什么 BERT 可以同时利用一个 Token 左右两侧信息？
2. 为什么 GPT 训练需要右移标签和 Causal Mask？
3. Base 模型直接套 Instruct Prompt 可能有什么问题？
4. GQA 为什么能减少生成时的 KV Cache？
5. Encoder-Decoder 的 Cross-Attention 中 Q、K、V 分别来自哪里？
