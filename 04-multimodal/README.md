# 阶段 4：视觉与多模态模型

这一阶段学习图像如何变成模型可处理的表示，以及视觉编码器、Projector 和语言模型如何协同完成图文任务。

建议用时：**5–7 天，每天 3–4 小时**。

## 前置知识

- 完成阶段 3 的模型加载、Chat Template 和批量推理
- 理解 CNN、Embedding 和 Multi-Head Attention
- 能处理图像 Tensor：`[batch, channels, height, width]`

## 完成后应该具备的能力

- 理解 Patch Embedding 和 Vision Transformer
- 使用 `AutoProcessor` 处理图像与文本
- 正确构造多模态 Chat Template
- 运行图片描述、OCR、视觉问答等任务
- 区分视觉感知错误与语言推理错误
- 建立小规模多模态评测流程

## 推荐学习顺序

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | 图像预处理 | [lessons/01-image-preprocessing.py](lessons/01-image-preprocessing.py) | 已实现 resize、normalize、反归一化和 batch 维度 |
| 02 | Patch Embedding | `lessons/02-patch-embedding.py` | 手写图片分块与线性投影，打印所有形状 |
| 03 | Vision Transformer | `lessons/03-vision-transformer.py` | 理解 CLS Token、位置编码和 Encoder 输出 |
| 04 | 图文对齐 | `lessons/04-clip-basics.py` | 理解图像/文本向量与相似度 |
| 05 | 多模态处理器 | `lessons/05-multimodal-processor.py` | 使用 Processor 同时处理图片和文字 |
| 06 | 图文对话 | `lessons/06-image-text-to-text.py` | 正确运行多模态对话模板和生成流程 |
| 07 | 批量评测 | `lessons/07-multimodal-evaluation.py` | 保存图片 ID、问题、输出、答案和错误类型 |

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

## 建议练习任务

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

## 常见误区

- 多模态模型通常使用 Processor，而不只是 Tokenizer
- 多模态 Chat Template 的消息内容通常是图像/文本元素列表
- 不要把“没看清图片”和“看清后推理错”混在一起
- 图片路径、URL 和 RGB 模式可能导致不同加载结果
- 不要只凭单个案例评价模型能力

## 完成清单

- [x] 能解释 Resize、Normalize、反归一化和 Batch 维度
- [ ] 手写 Patch Embedding 并验证形状
- [ ] 能解释 ViT 的输入和输出
- [ ] 能使用 AutoProcessor 构造多模态输入
- [ ] 完成至少三类图文任务
- [ ] 建立多模态 JSONL 评测数据
- [ ] 完成错误分类与可视化分析

## 权威资料

- [Hugging Face Image-Text-to-Text](https://huggingface.co/docs/transformers/tasks/image_text_to_text)
- [Multimodal Chat Templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [Vision Transformer 论文](https://arxiv.org/abs/2010.11929)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [MathVista 论文与项目](https://mathvista.github.io/)

完成后进入 [阶段 5：LLM Reasoning 与模型评测](../05-llm-reasoning/README.md)。
