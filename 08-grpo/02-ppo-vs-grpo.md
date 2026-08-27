# 从 PPO 到 GRPO

## PPO

PPO 使用旧策略与新策略的概率比 `r_t = πθ(a|s) / πold(a|s)`，再对它做 clipping，避免单次更新过大。优势通常依赖 Value/Critic：

```text
L_PPO = -min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
```

LLM 后训练还常加入 reference KL、value loss 与 entropy 等项。

## GRPO

GRPO 对同一个 Prompt 生成一组回答，用组内 reward 的均值/标准差构造相对优势：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
```

这样可以不训练独立 Critic，但必须为每个 Prompt 生成多个候选。若组内奖励完全一样，所有优势接近 0，缺少学习信号。

## 放在一起看

| 项目 | PPO | GRPO |
| --- | --- | --- |
| 在线 rollout | 需要 | 需要 |
| 优势基线 | Critic/GAE | 同 Prompt 组内奖励 |
| 独立 Value Model | 通常需要 | 不需要 |
| 多候选生成 | 非核心要求 | 核心要求 |
| KL Reference | 常见 | 常见 |
| 风险 | Critic 不稳定 | 组内方差不足、Reward Hacking |

两种方法都不能只盯着训练 reward。独立测试准确率、格式、截断、长度、KL 和样本级 transition 要放在一起看。
