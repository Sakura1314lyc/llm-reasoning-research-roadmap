# 阶段 8：GRPO 强化学习

GRPO（Group Relative Policy Optimization）会让同一个 Prompt 生成一组回答，再用组内相对奖励算优势。课程从这一步开始，一直写到策略更新、KL 约束和训练诊断。

建议用时：**7–10 天，每天 3–4 小时**。

## 前置知识

- 有稳定的 SFT 模型或可用的小型 Instruct 模型
- 理解策略、奖励、优势、KL 正则和重要性比率
- 有经过单元测试的答案抽取与奖励函数
- 能记录 rollout、reward 和训练配置

## 学完后要能自己解释

- 解释 GRPO 与 PPO 的核心差异
- 为可验证任务实现奖励函数
- 使用 TRL `GRPOTrainer` 跑通最小实验
- 监控 reward、KL、长度和格式等指标
- 识别 reward hacking 与训练崩溃
- 与 SFT/OPD 基线公平比较
- 能把冷启动 SFT、推理 RL、拒绝采样和通用对齐串成 R1-like 训练路线

SFT 和推理 RL 怎么串起来，单独放在 [R1-like 推理模型训练路线](R1-LIKE-TRAINING.md) 里。

## 先看 Reward，再看更新

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | 强化学习基础 | [01-reinforcement-learning-basics.py](01-reinforcement-learning-basics.py) | 已手写 Masked Policy Gradient |
| 02 | PPO 到 GRPO | [02-ppo-vs-grpo.md](02-ppo-vs-grpo.md) | 已对比 Critic、组内基线、Clipping 与 KL |
| 03 | Group Sampling | [03-group-rollout.py](03-group-rollout.py) | 已让同一 Prompt 生成多个候选 |
| 04 | 奖励函数 | [reward_functions.py](reward_functions.py) / [测试](tests/test_rewards.py) | 已覆盖正确性、格式、分数等价与长度边界 |
| 05 | Advantage | [05-group-advantage.py](05-group-advantage.py) | 已计算组内标准化奖励 |
| 06 | Token Logprob | [06-token-logprob.py](06-token-logprob.py) | 已在 completion Token 上读取策略概率 |
| 07 | Policy Gradient | [07-policy-gradient.py](07-policy-gradient.py) | 已连接 rollout、logprob 与 advantage |
| 08 | GRPO Loss | [08-loss.py](08-loss.py) | 已实现完成序列上的策略目标 |
| 09 | KL 正则 | [09-KL-regularization.py](09-KL-regularization.py) | 已加入 reference policy 约束 |
| 10 | Mini GRPO Step | [10-mini-grpo-step.py](10-mini-grpo-step.py) | 已串联 rollout、reward、advantage、loss 和更新 |
| 11 | TRL GRPOTrainer | [11-TRL-grpo-trainer.py](11-TRL-grpo-trainer.py) | 已跑通框架训练与 completion 日志 |
| 12 | 训练诊断 | [12-training-diagnostics.py](12-training-diagnostics.py) | 已汇总 reward、KL、长度和零方差组 |

## 奖励函数先于训练

数学任务可先拆成两个奖励：

- `accuracy_reward`：最终答案是否正确
- `format_reward`：是否满足要求的输出格式

每个奖励函数必须有独立测试：

- 正确答案
- 错误答案
- 等价分数/小数
- 无法抽取
- 超长输出
- 恶意利用格式规则的输出

如果奖励函数不可靠，GRPO 只会更快地优化错误目标。

## 先跑一个小实验

首次实验建议：

- 数据：100–500 个 Prompt
- 每个 Prompt：2–4 个 generation
- 奖励：可验证的 Exact Match + 格式奖励
- 训练：短步数 smoke test
- 评测：独立固定测试集

先确认一整个训练 batch 的 Prompt、候选、抽取答案、reward 和 advantage 都正确，再长时间训练。

## 训练期间重点观察

- 平均 reward 与 reward 方差
- 每个 Prompt 的组内奖励是否有差异
- KL 是否突然增大
- 输出长度是否异常增长
- 格式奖励是否提高但准确率不升
- 无效答案率与重复答案率
- 训练前后独立测试集准确率

## 阶段项目：SFT 与 GRPO 对比

```text
experiments/experiment-004-grpo/
├── configs/
├── reward_functions.py
├── tests/test_rewards.py
├── train.py
├── evaluate.py
├── rollouts/
├── metrics.json
└── analysis.md
```

最终至少比较：Base/Instruct、SFT、SFT+GRPO 三组。

## Reward 涨了也别急着下结论

- 组内 reward 全相同会缺少有效的相对学习信号
- 格式奖励过强可能导致 reward hacking
- 训练 reward 上升不等于测试准确率上升
- Rollout generation 常常比反向传播更耗时
- 不要在测试集上训练奖励或挑 checkpoint
- 先做 smoke test，再增加 generation 数量和序列长度

## 完成清单

- [x] 能解释 GRPO 的组内相对优势
- [x] 奖励函数有边界案例测试
- [x] 能检查完整 rollout 数据流
- [x] 跑通短步数 TRL GRPO smoke test
- [x] 已定义 reward、KL、长度和组内方差诊断
- [x] 已在任务一协议中规定与 SFT 基线公平比较
- [x] 已列出 reward hacking、截断与格式失败分析
- [x] 已梳理 R1-like 推理模型的分阶段训练与复现风险

## 权威资料

- [DeepSeekMath / GRPO 论文](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL Quickstart](https://huggingface.co/docs/trl/quickstart)

完成后进入 [阶段 9：Agent、Skill、MCP 与测试](../09-agents/README.md)。
