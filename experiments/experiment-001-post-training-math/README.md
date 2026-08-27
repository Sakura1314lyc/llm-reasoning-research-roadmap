# 任务一：数学推理后训练实验

这个任务把 Base、SFT、OPD 和 GRPO 放进同一套数学数据与评测协议里比较。SFT 和 OPD 必做；如果算力带不动 GRPO，先把课程 08 的数据流和 smoke test 跑通。

实验记录方式参考 [Post-training-Math](https://github.com/Sakura1314lyc/Post-training-Math)：除了宽松数值准确率，还保存格式合规率、截断率、输出长度、逐题 transition 和多 Seed 结果。那个仓库的数据只能当参考，不能写成本仓库的实测结果。

## 第一轮可以这样配

- 模型：`Qwen/Qwen2.5-Math-1.5B` 或资源允许的 3B 模型。
- 数据：GSM8K train；独立 dev-select、dev-audit 与 test。
- SFT：LoRA/QLoRA，可用 LLaMA-Factory 或 TRL。
- OPD：学生 rollout + 教师反馈；可用 TRL GKD/Distillation 接口或课程 07 手写 loss。
- GRPO：TRL/verl，小规模 smoke test 后再决定是否正式训练。
- Seeds：数据划分固定；训练/生成分别记录 42、43、44 等种子。

## 目录

```text
experiment-001-post-training-math/
├── configs/                 # 三种方法的示例配置
├── split-manifest.json      # 分区与哈希模板（真实数据仍放在忽略目录）
├── scripts/                 # 离线评测器与配对方法比较
├── tests/                   # 抽取与指标测试
├── PROTOCOL.md              # 冻结协议与运行顺序
├── REPORT-TEMPLATE.md       # 多 Seed、transition 与失败案例报告模板
└── README.md
```

真实数据、模型权重、Adapter、rollout 和 predictions 通常很大，放在 Git 忽略目录里。仓库只留配置、manifest、metrics、报告和少量去敏样例。

## 实验按这个顺序跑

1. 运行 Base/Instruct 基线，冻结 Prompt、生成参数与评测器。
2. 清洗 SFT 数据，固定 train/dev-select/dev-audit 分区与 SHA-256。
3. 先做 20 条样本 overfit，再运行多 Seed SFT。
4. 以同一 SFT checkpoint 为 OPD/GRPO 起点；保存行为策略与 reference 身份。
5. 只在 dev-select 选方案；dev-audit 只打开一次预先指定的 checkpoint。
6. 报告数值准确率、严格准确率、格式率、截断率、平均生成长度与逐题得失分。

离线评测命令：

```bash
python experiments/experiment-001-post-training-math/scripts/evaluate_math.py predictions.jsonl --output metrics.json
python experiments/experiment-001-post-training-math/scripts/compare_methods.py base.jsonl sft.jsonl --output base-vs-sft.json
pytest experiments/experiment-001-post-training-math/tests -q
```

## 现在做到哪了

- [x] 数据、配置、评测和审计协议已给出。
- [x] SFT/OPD/GRPO 示例配置已给出。
- [x] 离线答案抽取与指标测试可运行。
- [x] 逐题 transition、精确 McNemar 检验与报告模板已给出。
- [ ] 在本地真实算力环境填入模型训练结果。
- [ ] 完成多 Seed 汇总和最终实验报告。

最后两项只能等真实训练结束后再填，示例数字不能算结果。
