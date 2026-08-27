# 科研检索与人工智能会议入门

大纲里的预备工作不是让人背几个网站名字。真正要练的是一次完整检索：找到论文，确认版本和出处，定位官方代码，最后把证据记下来。

## 四个常用入口

| 平台 | 主要用途 | 使用时重点检查 |
| --- | --- | --- |
| arXiv | 查看论文预印本和版本历史 | 提交日期、版本号、分类、PDF 是否为最新版 |
| Google Scholar | 按主题、作者和引用关系检索 | 被引与相关论文、年份过滤、作者主页 |
| DBLP | 核对计算机论文书目信息 | 正式会议/期刊版本、作者同名、DOI |
| Hugging Face | 查模型、数据集、Demo 与代码 | Model/Dataset Card、许可证、revision、文件大小 |

arXiv 论文不等于已经同行评审。引用时应优先确认是否存在正式会议版本；运行模型时应固定模型全名和 revision，而不是只记录一个简称。

## 三个会议先认清

- NeurIPS：机器学习、优化、生成模型和强化学习等方向。
- ICML：机器学习方法、理论与应用。
- ICLR：表示学习、深度学习与开放评审。

会议主页、OpenReview 和论文 PDF 承担不同角色：主页确认会议与日程，OpenReview 查看评审讨论，论文和补充材料提供方法与实验细节。

## 用一个 Topic 走完检索流程

以“group relative policy optimization math reasoning”为例：

1. 用宽关键词找到代表论文和常见术语。
2. 加入年份、模型或数据集缩小范围。
3. 从代表论文的参考文献向前追溯，从引用列表向后追踪。
4. 在 DBLP/会议页核对正式发表信息。
5. 在 GitHub 与 Hugging Face 查官方代码、模型卡和许可证。
6. 建立表格记录问题、方法、数据、指标、资源和局限。

可以从这些检索式开始：

```text
"group relative policy optimization" math reasoning
site:arxiv.org LLM on-policy distillation
site:openreview.net multimodal reasoning SFT RL
```

## 论文初筛卡片

```markdown
- 标题：
- 正式出处 / arXiv：
- 研究问题：
- 核心方法：
- 使用模型与数据：
- 主要指标：
- 是否有官方代码：
- 与当前路线的关系：
- 最大复现风险：
```

## 怎么知道自己会了

- 能找到一个 Topic 最近三年的 10 篇相关论文。
- 能指出其中 2–3 篇代表作及其引用关系。
- 能区分预印本、正式论文、官方代码和第三方复现。
- 能在 `papers/` 中完成一篇结构化阅读笔记。
