# 学习大纲覆盖表

这张表只解决一个问题：大纲里的每一项，应该去仓库哪里看。`已整理` 表示讲解、最小代码或实验协议已经写进仓库，不表示 GPU/API 实验已经跑出了新结果。

| 大纲模块 | 仓库阶段 | 已覆盖内容 | 状态 |
| --- | --- | --- | --- |
| 前置科研能力 | [01](01-pytorch-basics/README.md) | arXiv、Hugging Face、Google Scholar、DBLP、ICML、NeurIPS、ICLR | 已整理 |
| 深度学习基础 | [01](01-pytorch-basics/README.md) | 前向/反向传播、CNN、ResNet、RNN/LSTM、GCN/GAT/GIN、GPU 与混合精度 | 已整理 |
| Transformer 与 LLM | [02](02-transformer/README.md)、[03](03-huggingface/README.md) | Attention、Encoder/Decoder、RoPE、Mini GPT、BERT/GPT/LLaMA/Qwen、Transformers/Datasets/PEFT | 已整理 |
| 多模态模型 | [04](04-multimodal/README.md) | Patch Embedding、ViT、CLIP、LLaVA、Qwen-VL、处理器、推理与评测 | 已整理 |
| 大模型推理 | [05](05-llm-reasoning/README.md)、[R1-like 路线](08-grpo/R1-LIKE-TRAINING.md) | Zero/Few-shot、CoT、Self-Consistency、答案抽取、R1-like 训练、GSM8K/MATH/MathVista/WeMath 评测 | 已整理 |
| SFT | [06](06-sft/README.md) | 数据校验、Chat Template、Label Mask、LoRA/QLoRA、Trainer 与 Adapter 对比 | 已整理 |
| OPD 与偏好优化 | [07](07-opd/README.md) | Off-policy KD、学生 rollout、教师反馈、On-policy loss、GKD、PPO/DPO/OPD 对照 | 已整理 |
| GRPO | [08](08-grpo/README.md) | 奖励、组采样、组内优势、Token log-prob、策略梯度、KL、TRL 与诊断 | 已整理 |
| 推理过程学习 | [05 rationale 笔记](05-llm-reasoning/RATIONALE-LEARNING.md) | DARE、Cooperative Classification、Federated Self-Explaining GNN 的阅读路线 | 已整理（选修） |
| LLM Agent | [09](09-agents/README.md) | Agent Loop、规划、Tool、Memory、Skill、MCP、Harness、测试与轨迹分析 | 已整理 |
| 任务一 | [数学推理后训练实验](experiments/experiment-001-post-training-math/README.md) | Base/SFT/OPD/GRPO 配置、冻结协议、离线评测和测试 | 框架完整，真实训练待运行 |
| 任务二 | [最小检查表](09-agents/TASK-2-CHECKLIST.md) | OpenClaw/Hermes/Codex/Claude Code 的本地配置记录入口 | 按要求仅保留入口 |

## 怎么走这条路线

1. 依次完成 01–04，建立深度学习、Transformer、Hugging Face 与多模态基础。
2. 完成 05 的推理与评测，再进入 06–08 的后训练方法。
3. 用任务一把 Base、SFT、OPD、GRPO 放进同一数据和评测协议中比较。
4. 最后完成 09，用测试和轨迹分析约束 Agent 行为。

## “学完”按什么算

- 课程代码能单独看懂；不依赖大模型的章节可以直接在本地运行。
- 需要下载模型、调用 API、使用 GPU 或第三方 Agent 的章节，至少有清楚的入口、配置和评测方法。
- 指标只认真实运行结果。模板、参考仓库数据和本仓库实测不能混在一起。
- 训练权重、Token、私有数据与大体积 rollout 不提交 Git。
