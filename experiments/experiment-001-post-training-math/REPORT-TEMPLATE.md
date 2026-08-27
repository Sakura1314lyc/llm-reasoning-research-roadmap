# 数学推理后训练实验报告模板

## 1. 实验问题

- SFT 相对 Base 改变了哪些题目、格式和输出长度？
- OPD 相对同一个 SFT 起点是否产生稳定的净收益？
- 如果运行 GRPO，reward 的变化是否迁移到冻结评测集？

## 2. 别人复现实验需要的信息

| 项目 | 内容 |
| --- | --- |
| Git commit |  |
| 模型与 revision |  |
| tokenizer |  |
| 数据 manifest / SHA-256 |  |
| 训练 Seeds |  |
| 生成 Seeds |  |
| GPU / 软件版本 |  |
| Prompt / Chat Template |  |

## 3. 汇总结果

| 方法 | Seed | Numeric Acc. | Strict Acc. | Format | Truncation | Avg. Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base |  |  |  |  |  |  |
| SFT |  |  |  |  |  |  |
| OPD |  |  |  |  |  |  |
| GRPO（选做） |  |  |  |  |  |  |

多 Seed 结果要同时放均值、标准差和每个 Seed 的原值，不能只挑最好的一次。

## 4. 配对变化

| 对比 | wrong→correct | correct→wrong | 净变化 | McNemar p-value |
| --- | ---: | ---: | ---: | ---: |
| Base → SFT |  |  |  |  |
| SFT → OPD |  |  |  |  |
| SFT → GRPO |  |  |  |  |

## 5. 哪些题答错了

把失败分成无法抽取、格式错误、计算错误、推理错误、截断、重复和疑似数据污染。每一类都保留题目 ID、原始输出和人工判断依据。

## 6. 能下什么结论，还有哪些限制

结论里要回答几个问题：提升能不能跨 Seed 保持，是否主要来自格式变化，输出成本有没有增加。模型规模、样本数、算力和评测污染也要写清楚。没有真实运行的数据格就留空，不能拿参考仓库的数字补位。
