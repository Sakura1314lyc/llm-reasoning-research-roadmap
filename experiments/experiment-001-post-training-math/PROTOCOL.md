# 先冻结协议，再开始训练

## 研究问题

1. SFT 是否提升数值准确率，还是主要改善格式与生成效率？
2. OPD 相对 SFT 是否带来稳定、可复现的样本级收益？
3. GRPO 的 reward 提升是否转化为独立集准确率？

## 数据隔离

- `train`：允许训练。
- `dev-select`：允许选配置和 checkpoint。
- `dev-audit`：协议冻结后只对预先指定 checkpoint 评测一次。
- `test/external`：不用于调参；若已查看并分析，必须在报告中说明污染风险。

数据划分只由 `data_seed` 决定；训练 seed 和生成 seed 不应改变题目集合。Manifest 记录原始文件 SHA-256、样本数、索引与生成脚本版本。

## 保持哪些条件不变

- 相同 Prompt/Chat Template、答案抽取器和最大新 Token。
- 相同 dev/test 样本顺序。
- 明确 Base、SFT、OPD 的 tokenizer 与 reference checkpoint。
- 每种方法至少 3 个训练 Seed；报告均值、标准差和逐题结果。
- 正式结果目录默认不覆盖；smoke test 使用独立输出目录。

## 结果不只看准确率

- Numeric accuracy：能抽取并与标准数值等价。
- Strict accuracy：数值正确且以 `#### <answer>` 结束。
- Format compliance：存在严格结尾。
- Truncation rate：达到生成上限且未自然结束。
- Average generated tokens：输出成本与失控长度。
- Transition：Base/SFT/OPD/GRPO 在同题上的 correct→wrong、wrong→correct。
- McNemar：对配对正确性变化做统计检验；不能只比较两个比例。

## 出现这些情况就先停

- Loss/reward 非有限、KL 或输出长度突然失控。
- 组内 reward 长期零方差。
- 格式指标上升但数值准确率显著下降。
- OPD completion 大量撞到长度上限。
- 发现训练集与 audit/test 重复。
