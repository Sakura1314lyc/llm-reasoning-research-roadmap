# 阶段 4：视觉与多模态模型

一张图片进入多模态模型后，会先变成视觉表示，再经过 Projector 接到语言模型。这一阶段就沿着这条数据流往下走，并把图文任务的评测方式补上。

建议用时：**5–7 天，每天 3–4 小时**。

## 前置知识

- 完成阶段 3 的模型加载、Chat Template 和批量推理
- 理解 CNN、Embedding 和 Multi-Head Attention
- 能处理图像 Tensor：`[batch, channels, height, width]`

## 学到哪算过关

- 理解 Patch Embedding 和 Vision Transformer
- 使用 `AutoProcessor` 处理图像与文本
- 正确构造多模态 Chat Template
- 运行图片描述、OCR、视觉问答等任务
- 区分视觉感知错误与语言推理错误
- 建立小规模多模态评测流程

## 从图片张量开始学

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | 图像预处理 | [lessons/01-image-preprocessing.py](lessons/01-image-preprocessing.py) | 已实现 resize、normalize、反归一化和 batch 维度 |
| 02 | Patch Embedding | [lessons/02-patch-embedding.py](lessons/02-patch-embedding.py) | 已手写图片分块与线性投影 |
| 03 | Vision Transformer | [lessons/03-vision-transformer.py](lessons/03-vision-transformer.py) | 已实现 CLS Token、位置编码和 Encoder |
| 04 | 图文对齐 | [lessons/04-clip-basics.py](lessons/04-clip-basics.py) | 已实现归一化相似度与对称对比损失 |
| 05 | 多模态处理器 | [lessons/05-multimodal-processor.py](lessons/05-multimodal-processor.py) | 已展示 Processor 的图文批处理 |
| 06 | 图文对话 | [lessons/06-image-text-to-text.py](lessons/06-image-text-to-text.py) | 已构造多模态消息与可选生成流程 |
| 07 | 批量评测 | [lessons/07-multimodal-evaluation.py](lessons/07-multimodal-evaluation.py) | 已定义逐样本记录、错误类型与汇总指标 |

模型脉络见 [ViT、LLaVA 与 Qwen-VL 架构笔记](MODEL-ARCHITECTURES.md)。

## 第 01 章：图像预处理

运行：

```bash
python 04-multimodal/lessons/01-image-preprocessing.py
```

需要持续跟踪的形状与数值变化：

```text
原始图片:    [C, H, W], uint8, [0, 255]
Resize 后:   [C, 224, 224]
浮点化后:    [C, 224, 224], float32, [0, 1]
Normalize:   (pixel - mean) / std
组成 Batch:  [B, C, 224, 224]
```

课程使用通用 ImageNet 统计量演示原理。加载真实预训练视觉模型时，应使用模型配套的 `AutoImageProcessor` 或 `AutoProcessor`，不要自行假设图片尺寸、均值和标准差。

## 可以拿来练手的任务

按难度逐步推进：

1. 图片描述：图片中有哪些主要对象？
2. OCR：读取短文本、数字和表格单元格
3. 视觉问答：根据图片回答封闭式问题
4. 图表理解：读取坐标轴、趋势和极值
5. 视觉数学：对图片中的数学信息进行计算

每类至少准备 10 个样本，不要只测试一张图片。

## 阶段项目：多模态错误分析器

```text
projects/01-multimodal-evaluation/
├── config.yaml
├── samples.jsonl
├── images/
├── run_inference.py
├── evaluate.py
├── predictions.jsonl
└── analysis.md
```

建议错误分类：

- `perception`：没有识别到关键视觉信息
- `ocr`：文字或数字读取错误
- `grounding`：对象与位置对应错误
- `reasoning`：视觉信息正确，但推理过程错误
- `calculation`：算术错误
- `format`：答案正确但格式不符合评测规则

## 实验要求

- 固定图片、问题和生成参数
- 保存原始模型输出，不只保存抽取答案
- 给每个样本分配稳定 ID
- 记录图片分辨率和预处理方式
- 至少人工复核 30 个样本

## 几个容易混在一起的问题

- 多模态模型通常使用 Processor，而不只是 Tokenizer
- 多模态 Chat Template 的消息内容通常是图像/文本元素列表
- 不要把“没看清图片”和“看清后推理错”混在一起
- 图片路径、URL 和 RGB 模式可能导致不同加载结果
- 不要只凭单个案例评价模型能力

## 完成清单

- [x] 能解释 Resize、Normalize、反归一化和 Batch 维度
- [x] 手写 Patch Embedding 并验证形状
- [x] 能解释 ViT 的输入和输出
- [x] 能解释 LLaVA 与 Qwen-VL 的视觉—语言连接
- [x] 能使用 AutoProcessor 构造多模态输入
- [x] 已设计图片描述、OCR、VQA、图表与视觉数学任务
- [x] 建立多模态逐样本评测格式
- [x] 完成 perception/OCR/grounding/reasoning/calculation/format 错误分类

## 权威资料

- [Hugging Face Image-Text-to-Text](https://huggingface.co/docs/transformers/tasks/image_text_to_text)
- [Multimodal Chat Templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [Vision Transformer 论文](https://arxiv.org/abs/2010.11929)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [MathVista 论文与项目](https://mathvista.github.io/)

完成后进入 [阶段 5：LLM Reasoning 与模型评测](../05-llm-reasoning/README.md)。
