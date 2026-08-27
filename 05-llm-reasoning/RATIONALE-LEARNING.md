# Rationale Learning、图学习与可信学习（选读）

这是大纲里的非 LLM 选读部分，只负责把三篇论文放进同一条阅读线，不能代替原论文。

## Rationale 到底是什么

Rationale Learning 希望模型不仅输出预测，还选出支持预测的最小、忠实输入子集（rationale）。在图任务中，rationale 往往表现为关键节点、边或子图。

## 三篇论文分别看什么

1. **Dare: Disentanglement-augmented rationale extraction**：关注如何把因果/任务相关信息与干扰因素解耦，以及抽取 rationale 的训练目标。
2. **Cooperative Classification and Rationalization for Graph Generalization**：关注分类器和 rationale 生成器如何协作，以及子图解释对分布外泛化的作用。
3. **Federated Self-Explaining GNNs with Anti-shortcut Augmentations**：先理解联邦学习中数据留在客户端、参数/更新参与聚合的基本流程，再看自解释 GNN 与 anti-shortcut augmentation 如何结合。

## 阅读时别放过这些问题

- rationale 是可解释证据，还是只对预测有用的稀疏特征？
- 忠实性（faithfulness）、充分性（sufficiency）和简洁性如何评测？
- 模型会不会利用数据偏差或 shortcut 产生看似合理的解释？
- 图的离散选择如何训练，是否使用采样、松弛或正则？
- 联邦设置中通信、隐私和客户端异质性如何影响结论？

每读一篇，就在 `papers/` 建一张卡片。问题、方法模块、损失项、数据划分、解释指标、泛化结果和复现风险都记下来，后面横向比较会省很多时间。
