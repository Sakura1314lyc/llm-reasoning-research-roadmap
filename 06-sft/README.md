# 阶段 6：SFT 监督微调

SFT（Supervised Fine-Tuning）直接用整理好的输入和回答训练模型。后面的蒸馏与强化学习是否有效，都得先和这条基线比较。

建议用时：**7 天，每天 3–4 小时**。

## 前置知识

- 完成 Hugging Face 模型、Tokenizer、Dataset 和评测流程
- 理解语言模型的 next-token prediction loss
- 能建立训练前基线，并保存逐样本结果

## 走完这一段要会什么

- 检查并构造对话式 SFT 数据
- 正确使用模型 Chat Template
- 理解输入 Token、标签和 Loss Mask
- 使用 TRL `SFTTrainer` 完成小规模训练
- 使用 LoRA/QLoRA 降低训练成本
- 比较 Base、Instruct 和 SFT 后模型

## 先处理数据，再碰 Trainer

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | SFT 数据格式 | [lessons/01-sft-dataset.py](lessons/01-sft-dataset.py) | 已检查 messages、角色、空回答和重复样本 |
| 02 | Chat Template | [lessons/02-format-and-tokenize.py](lessons/02-format-and-tokenize.py) | 已展示模板化文本、Token 和标签 |
| 03 | Loss Mask | [lessons/03-label-masking.py](lessons/03-label-masking.py) | 已屏蔽 Prompt 与 Padding Token |
| 04 | LoRA 原理 | [lessons/04-lora.py](lessons/04-lora.py) | 已手写低秩增量并统计可训练参数 |
| 05 | SFTTrainer | [lessons/05-sft-trainer.py](lessons/05-sft-trainer.py) | 已固定最小实验配置与有效 Batch |
| 06 | QLoRA | [lessons/06-qlora.py](lessons/06-qlora.py) | 已区分 4-bit 存储、计算 dtype 与 Adapter |
| 07 | 训练评测 | [lessons/07-evaluate-adapter.py](lessons/07-evaluate-adapter.py) | 已实现配对得失分比较 |

## 数据准备检查表

训练前先统计：

- 样本数量和数据来源
- Prompt/Response 长度分布
- 空值、重复和异常长样本
- train/validation/test 是否有重复题目
- 模板化后是否出现重复 BOS/EOS
- Assistant 回答是否真的参与 Loss

推荐保留原始字段，不要只留下拼接后的字符串。

## 第一轮实验别做太大

第一次先用这套规模检查训练链路：

- 模型：0.5B–0.6B 级别小模型
- 数据：500–2,000 条高质量样本
- 方法：LoRA
- Epoch：1–3
- 评测：固定的 100–300 条独立测试集

先跑通 20 条样本的 overfit test：如果模型无法在极小数据上降低损失，优先检查数据、模板和标签，而不是增加训练时间。

## 阶段项目：数学推理 SFT 基线

```text
experiments/experiment-002-sft/
├── config.yaml
├── prepare_dataset.py
├── train.py
├── evaluate.py
├── adapter/              # 本地保存，不进入 Git
├── metrics.json
└── analysis.md
```

必须比较：

| 模型 | 训练数据 | 评测设置 | 准确率 | 有效回答率 |
| --- | --- | --- | ---: | ---: |
| Base/Instruct | 无 | 固定 | 待测 | 待测 |
| SFT + LoRA | 固定 SFT 数据 | 相同 | 待测 | 待测 |

## 哪些参数不能漏记

- 基座模型完整名称与 revision
- Chat Template
- LoRA `r`、`alpha`、dropout、target modules
- 学习率、batch size、gradient accumulation
- 最大序列长度和截断比例
- 精度类型、量化配置和显存峰值
- 训练步数、耗时与最佳 checkpoint

## 训练跑起来也可能是错的

- Chat Template 与模型训练格式不匹配会静默降低效果
- 不要在测试集上挑 checkpoint 或调 Prompt
- 训练 loss 降低不代表推理能力提升
- LoRA Adapter 不是完整模型权重
- QLoRA 节省显存，但不会消除数据与评测问题

## 完成清单

- [x] 完成数据质量检查与去重规则
- [x] 展示模板化输入和 Loss Mask
- [x] 已在阶段 3 跑通极小数据 LoRA SFT
- [x] 完成 LoRA/QLoRA 原理与配置
- [x] 能独立保存与加载 Adapter
- [x] 已定义未微调/SFT 配对公平比较
- [x] 已提供训练与错误分析报告规范

## 权威资料

- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL Chat Templates](https://huggingface.co/docs/trl/chat_templates)
- [PEFT Quicktour](https://huggingface.co/docs/peft/quicktour)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)

完成后进入 [阶段 7：OPD 在线策略蒸馏](../07-opd/README.md)。
