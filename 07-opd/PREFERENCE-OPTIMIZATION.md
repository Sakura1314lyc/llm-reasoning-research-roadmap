# PPO、DPO 与 OPD 的位置

OPD、PPO 和 DPO 都属于后训练，但学到这里很容易只记住缩写。先看它们各自使用什么信号、更新什么对象，再谈联系。

| 方法 | 主要数据/信号 | 核心对象 | 是否在线采样 |
| --- | --- | --- | --- |
| SFT | 标准回答 | Token Cross-Entropy | 否 |
| DPO | chosen/rejected 偏好对 | 相对 reference 的隐式偏好目标 | 通常否 |
| PPO | Reward Model/环境奖励 | clipped policy objective + value/advantage | 是 |
| OPD/GKD | 教师分布或教师反馈 | 教师—学生 Token 分布差异 | OPD 是 |
| GRPO | 同一 Prompt 的组内奖励 | 组相对优势 + policy/KL | 是 |

## PPO

PPO 使用 rollout、reward、advantage、value function 和 clipped importance ratio。LLM 场景通常还加入 reference-model KL，限制策略偏离。它流程完整但需要 Actor/Critic/Reward/Reference 等组件，资源和调参成本高。

## DPO

DPO 直接使用 `(prompt, chosen, rejected)`，比较策略相对 reference 对 chosen/rejected 的 log-ratio。它不需要显式训练 Reward Model，也不需要在线 rollout，但高度依赖偏好数据质量和 reference 选择。

## OPD

OPD 不靠“回答拿到高奖励”来训练。学生在自己访问到的序列状态上学习教师分布，用在线生成和教师打分的成本，换取对学生实际推理分布的覆盖。

先手算一遍每种 Loss，再去用 TRL、verl 之类的框架。正式实验里要留下版本、reference、采样温度、生成长度和数据来源，否则结果很难复现。
