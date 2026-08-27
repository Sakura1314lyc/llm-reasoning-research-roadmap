# 阶段 7：OPD 在线策略蒸馏

OPD（On-Policy Distillation）先让学生模型自己生成，再让教师对学生真正走到的序列给反馈。这样处理的是一个很具体的问题：固定教师数据和学生推理时访问到的分布并不完全一样。

这部分比 SFT 更偏研究实验。TRL 的相关接口仍可能变化，所以版本和配置要记全，开始时也别把规模开得太大。

建议用时：**7–10 天，每天 3–4 小时**。

## 前置知识

- 完成 SFT 基线并有稳定评测器
- 理解 teacher/student、logits、KL divergence 和 sampling
- 能分别加载教师与学生模型
- 有足够资源进行 rollout；显存不足时先用更小模型或离线模拟

## 先理解三个概念

### Off-policy KD

学生只学习固定教师输出。实现简单，但训练时看到的序列与学生推理时自己的序列可能不同。

### On-policy KD

学生先生成 rollout，教师再对学生实际访问到的状态或 Token 分布提供监督。

### Mixed policy

同时使用教师数据和学生 rollout，在稳定性与分布匹配之间折中。

## 从离线蒸馏走到在线蒸馏

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | 普通知识蒸馏 | [lessons/01-off-policy-kd.py](lessons/01-off-policy-kd.py) | 已实现温度缩放和 Token 级 KL |
| 02 | 学生 Rollout | [lessons/02-student-rollout.py](lessons/02-student-rollout.py) | 已定义 Prompt、学生序列和生成 logprob 记录 |
| 03 | 教师反馈 | [lessons/03-teacher-feedback.py](lessons/03-teacher-feedback.py) | 已在学生序列上读取教师选中 Token logprob |
| 04 | On-policy Loss | [lessons/04-on-policy-loss.py](lessons/04-on-policy-loss.py) | 已对齐 Token、Mask 与教师/学生 logits |
| 05 | 混合训练 | [lessons/05-mixed-policy.py](lessons/05-mixed-policy.py) | 已实现离线/在线加权损失 |
| 06 | TRL GKD | [lessons/06-gkd-trainer.py](lessons/06-gkd-trainer.py) | 已固定最小 GKD 实验配置卡 |
| 07 | 评测分析 | [lessons/07-opd-evaluation.py](lessons/07-opd-evaluation.py) | 已联合比较质量、输出与吞吐成本 |

PPO、DPO、OPD 与 GRPO 的关系见 [偏好优化与在线学习对照](PREFERENCE-OPTIMIZATION.md)。

## 先做一组能看懂的对照

先用同一模型族的大小模型，避免 tokenizer 不一致带来的额外复杂度：

| 组别 | 学生训练方式 | 用途 |
| --- | --- | --- |
| A | SFT | 基线 |
| B | 固定教师数据 KD | Off-policy 基线 |
| C | 学生 rollout + 教师反馈 | OPD |
| D | 教师数据 + 学生 rollout | Mixed OPD |

开始时只使用 100–500 个 Prompt 和很短的生成长度，确保 Token 对齐、Mask 与 Loss 正确后再扩大。

## 跑完不能只留一个 Loss

- 下游任务准确率
- Token 级蒸馏 loss
- 学生 rollout 平均长度
- 无效输出率
- Rollout 与训练吞吐
- 教师和学生显存占用
- 在线/离线样本比例
- 不同随机种子的方差

## 出问题时按这个顺序查

1. 同一批 Token 上比较教师/学生 logits 形状
2. 检查 Padding Token 是否进入 loss
3. 使用完全相同的 teacher/student 验证 loss 应较小
4. 在极小数据上观察 loss 是否下降
5. 最后才开启真实 on-policy generation

## 阶段项目：SFT、KD 与 OPD 对比

```text
experiments/experiment-003-opd/
├── configs/
├── rollouts/
├── train_off_policy.py
├── train_on_policy.py
├── evaluate.py
├── metrics.json
└── analysis.md
```

最终报告需要说明：OPD 的收益是否足以覆盖额外 rollout 与教师推理成本。

## 容易把实验带偏的地方

- 学生生成内容必须与教师 logits 在 Token 级正确对齐
- 不同 tokenizer 的教师/学生会显著增加实现难度
- 教师更强不代表每个 Token 的软分布都可靠
- On-policy 数据会随学生变化，实验比 SFT 更难复现
- TRL 实验性接口可能变化，务必记录精确版本和 commit

## 完成清单

- [x] 手写 Token 级蒸馏 loss
- [x] 完成学生 rollout 与教师反馈数据流
- [x] 验证 Padding/截断/Token 对齐
- [x] 跑通离线模拟 Off-policy KD 基线
- [x] 提供最小 OPD/GKD 配置与调试顺序
- [x] 定义 SFT、KD、OPD 的质量与成本比较
- [x] 说明在线数据、版本和资源局限

## 权威资料

- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- [TRL GKDTrainer](https://huggingface.co/docs/trl/gkd_trainer)
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer)
- [TRL Distillation Examples](https://huggingface.co/docs/trl/example_overview)

完成后进入 [阶段 8：GRPO 强化学习](../08-grpo/README.md)。
