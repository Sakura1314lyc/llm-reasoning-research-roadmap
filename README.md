# LLM Reasoning Research Roadmap

> 从深度学习基础一路学到大语言模型、多模态推理、模型后训练和 LLM Agent。

这个仓库用来记录我的人工智能科研学习过程。笔记、论文阅读、代码和实验结果都放在这里，之后复习时不用再到处找。

内容按 **8 周、每天约 4 小时** 排列。01–09 的课程已经整理齐，可以从头学习，也可以直接挑薄弱的部分复习。

* 理解神经网络、Transformer 和大语言模型的核心原理
* 使用 PyTorch 和 Hugging Face 完成模型训练与推理
* 了解大模型推理、多模态模型和 LLM Agent
* 使用 SFT、OPD 和 GRPO 等方法进行模型后训练
* 完成“提出问题 → 文献检索 → 代码复现 → 实验分析”的科研闭环

---

## 课程完成情况

| 阶段 | 学习方向 | 状态 | 当前成果 |
| ---: | --- | --- | --- |
| 1 | [PyTorch 与深度学习基础](01-pytorch-basics/README.md) | 已整理 | PyTorch、CNN/ResNet、RNN、GNN、GPU 与科研检索 |
| 2 | [Transformer 原理与手写实现](02-transformer/README.md) | 已整理 | Attention、RoPE、Mini GPT、训练、生成与模型家族 |
| 3 | [Hugging Face 与大语言模型](03-huggingface/README.md) | 已整理 | Transformers、Datasets、生成、SFT 数据链路与 LoRA |
| 4 | [视觉与多模态大模型](04-multimodal/README.md) | 已整理 | ViT、CLIP、LLaVA、Qwen-VL 与多模态评测 |
| 5 | [LLM Reasoning 与模型评测](05-llm-reasoning/README.md) | 已整理 | Prompt、CoT、Self-Consistency、答案抽取与基准评测 |
| 6 | [SFT 监督微调](06-sft/README.md) | 已整理 | 数据校验、Label Mask、LoRA/QLoRA 与 Adapter 评测 |
| 7 | [OPD 在线策略蒸馏](07-opd/README.md) | 已整理 | KD、Rollout、教师反馈、On-policy loss 与 GKD |
| 8 | [GRPO 强化学习](08-grpo/README.md) | 已整理 | Reward、Group Advantage、策略损失、KL 与 TRL |
| 9 | [LLM Agent、Skill、MCP 与软件测试](09-agents/README.md) | 已整理 | Loop、Planning、Tool、Memory、Skill、MCP 与测试 |

每一项大纲具体落在哪个文件，可以看 [学习大纲覆盖表](CURRICULUM-COVERAGE.md)。“已整理”只说明课程和代码入口已经写好。凡是需要 GPU 或 API 的实验，仍以真正跑出来的记录为准。

### 最近整理

* [01–09 完整课程与大纲覆盖表](CURRICULUM-COVERAGE.md)
* [数学推理后训练任务：Base/SFT/OPD/GRPO](experiments/experiment-001-post-training-math/README.md)
* [数学推理离线评测协议](experiments/experiment-001-post-training-math/PROTOCOL.md)
* [Agent 工具、Harness 与测试路线](09-agents/README.md)

### 平时从这里记录

* [每日与每周学习笔记](notes/README.md)
* [论文阅读指南](papers/README.md)
* [正式实验记录规范](experiments/README.md)

---

## 学习路线

### 1. 深度学习基础

主要内容：

* 神经网络的前向传播与反向传播
* PyTorch Tensor 与自动求导
* Dataset 与 DataLoader
* 损失函数与优化器
* MLP、CNN、ResNet、RNN
* GCN、GAT、GIN 等图神经网络

学完这一段，要能做到：

* 能够独立搭建完整的 PyTorch 训练流程
* 能够分析模型的训练损失、验证准确率和过拟合现象
* 能够保存、加载和评测模型

### 2. Transformer 与大语言模型

主要内容：

* Tokenization 与 Embedding
* Positional Encoding
* Self-Attention
* Multi-Head Attention
* Causal Mask 与 Padding Mask
* Transformer Encoder 与 Decoder
* BERT、GPT、LLaMA、Qwen 等模型

这一段过关时，应该能：

* 能够解释 Transformer 中各个张量的形状
* 能够手写简化版 Self-Attention 和 Transformer
* 能够使用 Hugging Face 加载并运行开源大语言模型

### 3. 视觉与多模态大模型

主要内容：

* Vision Transformer
* Patch Embedding
* LLaVA
* Qwen-VL 系列
* 图片描述、OCR、视觉问答和数学推理

做到下面几件事，就可以继续往后学：

* 理解视觉编码器、Projector 和语言模型之间的连接方式
* 能够运行多模态模型并整理测试案例
* 能够分析视觉识别错误和语言推理错误

### 4. 大模型推理

主要内容：

* Zero-shot 与 Few-shot Prompting
* Chain-of-Thought
* Self-Consistency
* GSM8K、MATH、MathVista、WeMath 等数据集
* 答案抽取与自动评测

这一部分最终要留下的是：

* 建立统一的模型推理与评测流程
* 比较不同 Prompt 方法对模型性能的影响
* 对模型错误进行分类和分析

### 5. 模型后训练

后训练部分放在一起比较：

* SFT：Supervised Fine-Tuning
* OPD：On-Policy Distillation
* PPO：Proximal Policy Optimization
* DPO：Direct Preference Optimization
* GRPO：Group Relative Policy Optimization

计划使用的小规模实验配置：

* 模型：Qwen2.5-Math-0.5B/1.5B
* 数据集：GSM8K，后续可扩展至 MATH
* 微调方法：LoRA 或 QLoRA
* 训练框架：LLaMA-Factory、TRL 或 verl

主要对比：

| 实验             | 方法         | 状态  | 准确率 |
| -------------- | ---------- | --- | --: |
| Experiment 001 | Base Model | 未开始 |   — |
| Experiment 002 | SFT        | 未开始 |   — |
| Experiment 003 | OPD        | 未开始 |   — |
| Experiment 004 | GRPO       | 选做  |   — |

### 6. LLM Agent

主要内容：

* Agent Loop
* Planning
* Tool Calling
* Memory
* Skill
* MCP
* Harness 工程
* Agent 轨迹分析
* 软件测试与安全控制

计划学习和使用：

* OpenClaw
* Hermes
* Codex
* Claude Code
* pytest

Agent 部分学到这里为止：

* 实现一个基础工具调用 Agent
* 配置一个可重复使用的 Skill
* 运行或编写一个 MCP Server
* 为 Agent 工具和运行轨迹编写测试

---

## 仓库结构

```text
llm-reasoning-research-roadmap/
├── 01-pytorch-basics/       # PyTorch 基础、练习、项目与分析结果
│   ├── lessons/             # 编号主线课程（01–19）
│   ├── projects/            # 完整训练项目
│   ├── practice/            # 小型知识点练习
│   ├── python-review/       # Python 复习与环境检查
│   └── outputs/             # 可展示的实验分析图片
├── 02-transformer/          # Transformer 原理与手写实现
├── 03-huggingface/          # Hugging Face 模型使用
├── 04-multimodal/           # ViT 与多模态模型
├── 05-llm-reasoning/        # Prompt、CoT 与模型评测
├── 06-sft/                  # SFT 微调实验
├── 07-opd/                  # OPD 蒸馏实验
├── 08-grpo/                 # GRPO 原理与实验
├── 09-agents/               # Agent、Skill、MCP 与测试
├── papers/                  # 论文阅读笔记
├── notes/                   # 每日笔记与每周总结
├── experiments/             # 正式实验结果与对比
├── requirements.txt         # Python 依赖
├── CURRICULUM-COVERAGE.md   # 老师大纲与 01–09 的逐项映射
├── .gitignore               # Git 忽略规则
├── LICENSE                  # 开源许可证
└── README.md                # 项目说明
```

---

## 环境配置

最好单独建一个 Python 虚拟环境，避免课程依赖和其他项目打架。

### 1. 克隆仓库

```bash
git clone https://github.com/Sakura1314lyc/llm-reasoning-research-roadmap.git
cd llm-reasoning-research-roadmap
```

### 2. 创建虚拟环境

使用 Conda：

```bash
conda create -n llm-roadmap python=3.11
conda activate llm-roadmap
```

或者使用 Python venv：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux/macOS：

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 检查 PyTorch 与 GPU

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## 学习记录怎么写

每日学习笔记统一记录在：

```text
notes/daily/
```

命名示例：

```text
week01-day01.md
week01-day02.md
```

每天不必写成长文，把下面这些信息留住就够了：

```markdown
# Week 1 Day 1

## 今日目标

## 学习内容

## 完成的代码

## 实验结果

## 遇到的问题

## 解决方法

## 尚未解决的问题

## 明日计划
```

每周总结记录在：

```text
notes/weekly/
```

每周需要总结：

* 本周学习的核心知识
* 完成的代码与实验
* 关键实验结果
* 遇到的主要问题
* 下周学习计划
* 准备向老师请教的问题

---

## 论文笔记怎么写

论文笔记统一存放在：

```text
papers/
```

论文笔记可以沿用下面的结构，缺少的信息先留空：

```markdown
# 论文标题

## 基本信息

- 作者：
- 会议或期刊：
- 年份：
- 论文链接：
- 代码链接：

## 研究问题

## 核心方法

## 实验设置

## 主要结果

## 创新点

## 局限性

## 与当前研究方向的联系

## 我的疑问
```

---

## 实验记录怎么留

正式实验至少留下这些文件：

```text
experiments/experiment-xxx/
├── config.yaml
├── command.sh
├── metrics.json
├── predictions.jsonl
├── training.log
└── analysis.md
```

实验记录至少需要包括：

* 模型名称与版本
* 数据集与数据划分
* 随机种子
* 学习率与 Batch Size
* LoRA 参数
* 训练轮数
* 推理参数
* 显存占用
* 训练时间
* 最终指标
* 典型成功案例
* 典型失败案例

大型模型权重、Checkpoint、API Key 和隐私数据不会上传至本仓库。

---

## 不要提交的内容

以下内容不应提交到 GitHub：

* API Key
* Access Token
* 密码
* `.env` 文件
* 模型权重
* 大型 Checkpoint
* 私有数据集
* 个人隐私信息

提交前先执行：

```bash
git status
```

确认没有敏感文件后再提交：

```bash
git add .
git commit -m "描述本次完成的任务"
git push
```

---

## 学完以后

走完这条路线后，我希望自己能独立完成这些事：

1. 理解 Transformer 和主流大语言模型的核心结构
2. 独立使用 PyTorch 完成模型训练与评测
3. 使用 Hugging Face 和 PEFT 微调开源模型
4. 理解并比较 SFT、OPD、DPO、PPO 和 GRPO
5. 完成 Base、SFT 和 OPD 的公平实验对比
6. 搭建并测试一个基础 LLM Agent
7. 阅读并整理人工智能领域论文
8. 独立完成一份科研实验报告

---

## Acknowledgements

本仓库的原始学习路线由课题组老师提供。

仓库中的学习笔记、代码实现、实验记录和个人总结由本人在学习过程中整理完成。

感谢以下开源社区和项目提供的学习资源：

* PyTorch
* Hugging Face
* Datawhale
* LLaMA-Factory
* TRL
* verl

---

## License

本仓库中由本人编写的代码和笔记采用 [MIT License](LICENSE)。

第三方代码、模型、数据集和论文资料仍遵循其原始许可证及使用条款。
