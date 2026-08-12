# 阶段 3：Hugging Face 与大语言模型

这一阶段的目标是把阶段 2 的 Transformer 原理连接到真实开源模型：会加载、会生成、会处理数据、会评测，并能解释每一步的输入输出。

建议用时：**5–7 天，每天 3–4 小时**。

## 前置知识

- 完成 PyTorch 主线课程 01–13
- 理解 Token、Embedding、Attention 和 Causal Mask
- 能读懂 `[batch, sequence, hidden]` 等张量形状

## 完成后应该具备的能力

- 使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载模型
- 正确使用模型自带的 Chat Template
- 理解 `input_ids`、`attention_mask`、logits 和生成结果
- 使用 `datasets` 加载、切分、映射和保存数据
- 完成批量推理，并记录可复现的生成参数
- 在小型数据集上建立文本生成基线

## 推荐学习顺序

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 00 | Pipeline 快速入门 | [lessons/00-pipeline-quickstart.py](lessons/00-pipeline-quickstart.py) | 已用显式模型完成情感分类 |
| 01 | Hub、配置与缓存 | [lessons/01-model-and-config.py](lessons/01-model-and-config.py) | 已打印模型类型、词表大小、层数和隐藏维度 |
| 02 | Tokenizer | [lessons/02-tokenizer.py](lessons/02-tokenizer.py) | 已完成编码、批量解码、Padding、Truncation 和特殊 Token |
| 03 | Causal LM 推理 | [lessons/03-causal-lm-inference.py](lessons/03-causal-lm-inference.py) | 已检查 logits、Top-k 候选并完成 Greedy 续写 |
| 04 | Chat Template | [lessons/04-chat-template.py](lessons/04-chat-template.py) | 已区分推理 Prompt 和完整 SFT 对话，并构造 Label Mask |
| 05 | SFT 预处理 | [lessons/05-sft-preprocessing.py](lessons/05-sft-preprocessing.py) | 已用 Dataset.map 构造 input_ids、attention_mask 和 labels |
| 06 | Dataset 与 Batch | [lessons/06-sft-dataset-and-batch.py](lessons/06-sft-dataset-and-batch.py) | 已实现动态 Padding，并屏蔽 Prompt 与 Padding 标签 |
| 07 | 全参数 SFT | [lessons/07-full-parameter-sft.py](lessons/07-full-parameter-sft.py) | 已走通 Qwen 全参数训练循环并理解资源开销 |
| 08 | LoRA SFT | [lessons/08-lora-sft.py](lessons/08-lora-sft.py) | 已配置 q/v 投影 LoRA、统计可训练参数并保存 Adapter |

## 当前课程如何运行

```bash
python 03-huggingface/lessons/00-pipeline-quickstart.py
python 03-huggingface/lessons/01-model-and-config.py
python 03-huggingface/lessons/02-tokenizer.py
python 03-huggingface/lessons/03-causal-lm-inference.py
python 03-huggingface/lessons/04-chat-template.py
python 03-huggingface/lessons/05-sft-preprocessing.py
python 03-huggingface/lessons/06-sft-dataset-and-batch.py
python 03-huggingface/lessons/07-full-parameter-sft.py
python 03-huggingface/lessons/08-lora-sft.py
```

第一次运行会从 Hugging Face Hub 下载模型资源，之后会复用本地缓存。课程 01 只读取体积较小的配置文件；课程 00 会下载情感分类模型权重。

当前四个脚本依次回答：

- Pipeline 如何封装预处理、模型前向与后处理
- `config.json` 能告诉我们哪些模型结构信息
- Token、Token ID、`input_ids` 和 `attention_mask` 如何对应
- 单条 `decode()` 与批量 `batch_decode()` 应该如何使用
- Padding 和 Truncation 为什么只影响批处理后的序列布局
- Causal LM 的 `[B, T, V]` logits 分别代表什么
- 如何读取最后一个位置的候选 token，并只解码新生成部分
- CPU 与 GPU 为什么通常需要选择不同的模型 dtype
- Chat Template 如何把 role/content 消息转换为模型协议
- 为什么 SFT labels 要用 `-100` 屏蔽 Prompt 和 Padding
- 动态 Padding 如何把变长样本组成 `[B, T]` Batch
- 全参数微调与 LoRA 的可训练参数和资源占用差异

## 第 04–06 章：SFT 数据管线

监督微调的样本需要同时保留完整对话输入，并只让 assistant 回答参与损失：

```text
question + answer
  → apply_chat_template
  → input_ids / attention_mask
  → labels = input_ids.clone()
  → Prompt 部分 labels = -100
  → 动态 Padding
  → Padding 部分 labels = -100
```

`-100` 是 PyTorch Cross-Entropy 常用的 `ignore_index`，这些位置不会贡献 loss。公共实现位于 [sft_utils.py](lessons/sft_utils.py)，后续训练课程直接复用。

## 第 07–08 章：全参数 SFT 与 LoRA

| 对比项 | 全参数 SFT | LoRA |
| --- | --- | --- |
| 更新内容 | 基础模型全部参数 | 插入的低秩矩阵 |
| 优化器状态 | 为所有参数维护 | 只为 LoRA 参数维护 |
| 资源占用 | 高 | 较低 |
| 保存结果 | 完整模型权重 | 小型 Adapter |
| 适用目标 | 充分改变模型能力 | 低成本任务适配 |

两个脚本默认只使用 3 条教学样本，因此只能验证训练链路，不能代表真实微调效果。LoRA 输出保存在本地 `03-huggingface/outputs/qwen2.5-0.5b-lora/`，已被 Git 忽略。

## 建议每天怎么学

### Day 1：模型与 Tokenizer

1. 选一个 0.5B–0.6B 级别的小模型
2. 加载 tokenizer 和 config，暂时不要加载大模型
3. 比较普通文本与对话消息的编码结果
4. 记录 BOS、EOS、PAD 等特殊 Token

### Day 2：模型推理

1. 加载 `AutoModelForCausalLM`
2. 手动执行一次前向传播
3. 检查 logits 形状
4. 使用 `generate()` 完成文本生成

### Day 3：生成策略

固定同一个 Prompt，只修改一个变量：

- greedy decoding
- `temperature`
- `top_p`
- `max_new_tokens`
- 随机种子

把输出与参数一起保存，不要只看终端结果。

### Day 4：数据处理

1. 使用 `load_dataset()` 加载小型数据集
2. 检查字段、数据类型和 split
3. 用 `map()` 完成格式转换
4. 保留原始问题、标准答案和样本 ID

### Day 5–7：阶段项目

完成一个“**小模型批量推理与评测器**”：

```text
projects/01-text-generation-baseline/
├── config.yaml
├── run_inference.py
├── extract_answer.py
├── evaluate.py
├── predictions.jsonl
└── analysis.md
```

至少比较两种生成设置，并分析 10 个失败案例。

## 代码记录要求

每次推理至少保存：

- 模型完整名称与版本
- Prompt 或 Chat Template
- 数据集名称与 split
- 随机种子
- `max_new_tokens`、`temperature`、`top_p`
- 原始输出、抽取答案和标准答案
- 运行设备与耗时

## 常见误区

- 不要手写聊天格式替代模型自带 Chat Template
- 不要只保存最终准确率，必须保留逐样本输出
- 不要混淆 `max_length` 与 `max_new_tokens`
- 批量推理时注意左/右 Padding 与模型要求
- 模型权重、缓存和 Token 不进入 Git
- SFT 中不能让 Prompt 与 Padding Token 参与回答损失
- 全参数 SFT 即使模型只有 0.5B，在 CPU 上也会很慢并占用大量内存
- LoRA 的 `target_modules` 必须和目标模型中的真实模块名称匹配

## 完成清单

- [x] 能独立加载 Pipeline、config 和 tokenizer
- [x] 能解释编码、批量解码、Padding 和 Attention Mask
- [x] 能独立加载 causal LM
- [x] 能解释 `input_ids`、logits 与新生成 token
- [x] 能正确应用 Chat Template
- [x] 能构造 SFT Dataset、Label Mask 和动态 Padding Batch
- [x] 能解释全参数微调与 LoRA 的差异
- [x] 能配置并保存 LoRA Adapter
- [ ] 能比较至少三种生成设置
- [ ] 能处理 Hugging Face Dataset
- [ ] 能输出 JSONL 预测文件和准确率
- [ ] 完成阶段项目与错误分析

## 权威资料

- [Transformers Quicktour](https://huggingface.co/docs/transformers/quicktour)
- [Datasets Tutorials](https://huggingface.co/docs/datasets/tutorial)
- [Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Text Generation](https://huggingface.co/docs/transformers/generation_strategies)

完成后进入 [阶段 4：视觉与多模态模型](../04-multimodal/README.md)。
