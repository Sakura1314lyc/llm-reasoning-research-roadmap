# 论文阅读指南

论文阅读的目标不是逐句翻译，而是提取研究问题、方法、实验依据和可复现要点。

## 推荐命名

```text
papers/
├── 2023-lora.md
├── 2023-qlora.md
├── 2024-deepseekmath-grpo.md
└── README.md
```

文件名使用：`年份-简短标题.md`。

## 三遍阅读法

### 第一遍：定位

- 论文解决什么问题？
- 为什么现有方法不够？
- 核心贡献是什么？
- 主要结果是否支持结论？

### 第二遍：理解

- 输入、输出和训练信号是什么？
- 方法与基线的差异在哪里？
- 关键公式中的每个变量是什么？
- 实验是否公平？消融实验说明了什么？

### 第三遍：复现

- 需要哪些模型、数据和计算资源？
- 哪些超参数最关键？
- 代码仓库与论文是否一致？
- 可以先复现哪个最小子实验？

## 论文笔记模板

```markdown
# 论文标题

## 基本信息

- 作者：
- 会议/期刊：
- 年份：
- 论文链接：
- 代码链接：

## 一句话总结

## 研究问题

## 核心方法

## 关键公式与变量

## 实验设置

## 主要结果

## 消融实验

## 创新点

## 局限性

## 复现所需资源

## 与当前路线的联系

## 我的疑问

## 下一步行动
```

## 优先论文清单

| 阶段 | 论文 |
| --- | --- |
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| 视觉 | [Vision Transformer](https://arxiv.org/abs/2010.11929) |
| 推理 | [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) |
| 推理 | [Self-Consistency](https://arxiv.org/abs/2203.11171) |
| SFT/PEFT | [LoRA](https://arxiv.org/abs/2106.09685) |
| SFT/PEFT | [QLoRA](https://arxiv.org/abs/2305.14314) |
| OPD | [On-Policy Distillation](https://arxiv.org/abs/2306.13649) |
| GRPO | [DeepSeekMath](https://arxiv.org/abs/2402.03300) |

## 完成标准

一篇论文只有在满足以下条件后才算“读完”：

- [ ] 能用三句话解释问题、方法和结论
- [ ] 能画出方法的数据流
- [ ] 能解释至少一个关键公式
- [ ] 能指出一个局限性
- [ ] 能提出一个最小复现实验
