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
| 01 | Hub、配置与缓存 | `lessons/01-model-and-config.py` | 能打印模型类型、词表大小、层数和隐藏维度 |
| 02 | Tokenizer | `lessons/02-tokenizer.py` | 能解释编码、解码、Padding、Truncation 和特殊 Token |
| 03 | Chat Template | `lessons/03-chat-template.py` | 能把 `messages` 正确转换为模型训练时使用的格式 |
| 04 | Causal LM 推理 | `lessons/04-causal-lm-inference.py` | 能从 logits 到生成文本走通完整流程 |
| 05 | 生成参数 | `lessons/05-generation-config.py` | 比较 greedy、sampling、temperature、top-p 的差异 |
| 06 | Datasets | `lessons/06-datasets.py` | 完成加载、查看、`map`、`filter` 和划分 |
| 07 | 批量推理 | `lessons/07-batch-inference.py` | 对一批 Prompt 推理并保存 JSONL 结果 |
| 08 | 基础评测 | `lessons/08-evaluation.py` | 实现答案抽取、Exact Match 和错误样本保存 |

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

## 完成清单

- [ ] 能独立加载 tokenizer、config 和 causal LM
- [ ] 能解释一次前向传播的主要张量
- [ ] 能正确应用 Chat Template
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
