# ViT、LLaVA 与 Qwen-VL 架构笔记

## 图片怎么进入 Transformer

ViT 将图片切成固定 Patch，经线性投影形成视觉 Token，再添加位置表示并送入 Transformer Encoder。分类模型通常读取 CLS Token；多模态模型则使用一组视觉 Token 与语言模型交互。

## LLaVA 怎么接视觉和语言

先看最简化的数据流：

```text
Image → CLIP Vision Encoder → Projector → Visual Tokens
                                             ↓
Text Tokens ─────────────────────────→ Language Model
```

Projector 负责把视觉编码器维度映射到语言模型隐藏空间。训练通常包含视觉—语言对齐和指令微调阶段。理解它时要区分：视觉编码器看到了什么、Projector 如何对齐、语言模型如何推理。

## Qwen-VL 不同版本不能混着配

Qwen-VL 同样把视觉表示接入语言模型，但不同代际会更新视觉编码器、动态分辨率、位置编码、视频支持与训练数据。具体实现应以对应版本 Technical Report 和 Model Card 为准，不能把某一版的输入格式套到另一版。

## 答错了，先判断错在哪一段

- perception：视觉编码器没有识别关键对象。
- OCR/grounding：文字或空间对应错误。
- reasoning/calculation：视觉事实正确，后续推理错误。
- format：内容可用但不符合评测格式。

如果没留下原始输出、图片 ID、预处理方式和问题，这几类错误事后很难分清。
