# 实验记录规范

这个目录只保存可以复现、可以比较、可以解释的正式实验。随手练习放在各阶段的 `practice/` 或 `lessons/`。

## 推荐结构

```text
experiments/
└── experiment-001-zero-shot/
    ├── README.md
    ├── config.yaml
    ├── run.py
    ├── evaluate.py
    ├── metrics.json
    ├── predictions.jsonl
    └── analysis.md
```

大型模型、Adapter、Checkpoint 和缓存不进入 Git。

## 实验 README 模板

```markdown
# Experiment XXX：实验名称

## 研究问题

## 假设

## 自变量

## 控制变量

## 模型与数据

## 运行命令

## 评测方法

## 主要结果

## 错误分析

## 局限性

## 结论与下一步
```

## `config.yaml` 至少包含

```yaml
experiment_id: experiment-001
seed: 42
model:
  name: full-model-name
  revision: null
dataset:
  name: dataset-name
  split: test
generation:
  max_new_tokens: 256
  temperature: 0.0
evaluation:
  metric: exact_match
```

训练实验还需要记录：

- 学习率、batch size、gradient accumulation
- Epoch/steps、warmup 和 scheduler
- 最大序列长度与截断比例
- LoRA/量化配置
- 精度类型与设备
- 显存峰值、训练耗时和 checkpoint 选择规则

## `predictions.jsonl` 推荐字段

```json
{
  "id": "sample-0001",
  "input": "...",
  "gold_answer": "...",
  "raw_output": "...",
  "prediction": "...",
  "correct": true,
  "metadata": {}
}
```

## 对比实验原则

- 一次只改变一个主要变量
- 所有方法使用相同测试集和评测器
- 不在测试集上调 Prompt、奖励或 checkpoint
- 报告多个随机种子或说明只运行一次的限制
- 同时报告质量、成本和失败案例
- 保留逐样本结果，不能只保存平均分

## 提交前检查

- [ ] 配置和随机种子完整
- [ ] 运行命令可以复制执行
- [ ] 数据 split 清晰且没有泄漏
- [ ] 指标可以从 predictions 重新计算
- [ ] 没有模型权重、Token 或隐私数据
- [ ] 结论与表格数字一致
- [ ] 至少分析 10 个失败案例

## 计划实验

| 编号 | 实验 | 状态 |
| --- | --- | --- |
| 001 | Zero-shot / Few-shot / CoT / Self-Consistency | 未开始 |
| 002 | Base vs SFT | 未开始 |
| 003 | SFT vs Off-policy KD vs OPD | 未开始 |
| 004 | SFT vs SFT+GRPO | 未开始 |
