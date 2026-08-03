# 阶段 5：LLM Reasoning 与模型评测

这一阶段的重点不是“让模型多说几步”，而是建立公平、可复现、可分析的推理实验流程。

建议用时：**7 天，每天 3–4 小时**。

## 前置知识

- 完成阶段 3 的批量推理与答案抽取
- 能保存逐样本预测与生成参数
- 理解基本概率采样和随机种子

## 完成后应该具备的能力

- 建立 Zero-shot、Few-shot 和 CoT 基线
- 实现 Self-Consistency 多样本投票
- 为 GSM8K/MATH 类任务抽取与归一化答案
- 区分格式错误、计算错误和推理错误
- 生成公平的实验对比表
- 对结论给出样本级证据

## 推荐学习顺序

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | Zero-shot 基线 | `lessons/01-zero-shot.py` | 固定 Prompt 与生成参数，得到第一版准确率 |
| 02 | Few-shot Prompt | `lessons/02-few-shot.py` | 控制示例数量和顺序，只改变一个变量 |
| 03 | Chain-of-Thought | `lessons/03-chain-of-thought.py` | 比较直接回答和推理提示 |
| 04 | Self-Consistency | `lessons/04-self-consistency.py` | 多次采样、答案归一化和多数投票 |
| 05 | 答案抽取 | `lessons/05-answer-extraction.py` | 为数字、分数和选项实现稳定抽取 |
| 06 | 自动评测 | `lessons/06-evaluator.py` | 输出正确率、无效回答率和逐题结果 |
| 07 | 错误分类 | `lessons/07-error-analysis.py` | 为失败样本建立可复用分类规则 |

## 第一套标准实验

建议先使用 GSM8K 的小型固定子集，例如 100–300 题：

| 实验 | Prompt | 采样 | 目的 |
| --- | --- | --- | --- |
| A | 直接回答 | Greedy | 最小基线 |
| B | Zero-shot CoT | Greedy | 比较推理提示 |
| C | Few-shot CoT | Greedy | 比较示例作用 |
| D | Zero-shot CoT | 多次采样 | Self-Consistency |

除目标变量外，模型、数据、答案抽取、最大生成长度等设置必须相同。

## 推荐结果格式

```json
{
  "id": "gsm8k-test-0001",
  "question": "...",
  "gold_answer": "42",
  "prompt_method": "zero-shot-cot",
  "raw_output": "...",
  "extracted_answer": "42",
  "correct": true,
  "latency_seconds": 1.23
}
```

## 错误分类建议

- `format_error`：无法抽取答案
- `arithmetic_error`：算术步骤错误
- `reasoning_error`：方法或逻辑错误
- `instruction_error`：没有遵守回答格式
- `context_error`：遗漏或误读题目信息
- `truncation`：生成长度不足
- `evaluation_error`：评测器自身误判

先人工检查评测器，再分析模型。不要把答案抽取 Bug 当成模型错误。

## 阶段项目：推理方法对比报告

```text
experiments/experiment-001-prompting/
├── config.yaml
├── prompts/
├── run.py
├── evaluate.py
├── predictions/
├── metrics.json
└── analysis.md
```

报告至少回答：

1. 哪种方法准确率最高？
2. 提升是否来自更高有效回答率？
3. 哪些题型改善最多？
4. 哪些错误仍然存在？
5. 推理成本增加了多少？

## 完成清单

- [ ] 完成四组公平 Prompt 实验
- [ ] 答案抽取经过人工抽样验证
- [ ] 保存所有逐样本原始输出
- [ ] 实现 Self-Consistency
- [ ] 分析至少 30 个失败案例
- [ ] 形成完整实验报告

## 权威资料

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [GSM8K](https://arxiv.org/abs/2110.14168)
- [MATH Dataset](https://arxiv.org/abs/2103.03874)
- [MathVista](https://arxiv.org/abs/2310.02255)

完成后进入 [阶段 6：SFT 监督微调](../06-sft/README.md)。
