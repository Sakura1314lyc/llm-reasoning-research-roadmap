# PyTorch 与深度学习基础

先把 Tensor 和 Autograd 弄明白，再搭网络、写训练循环。后半段用 CNN 项目把数据、训练、保存和错误分析串起来，最后补上 ResNet、序列模型、GNN 与混合精度。

## 目录说明

```text
01-pytorch-basics/
├── lessons/         # 按编号排列的主线课程
├── projects/        # 可以独立运行的完整项目
├── practice/        # 早期练习和小型代码实验
├── python-review/   # Python 语法复习与环境检查
├── outputs/         # 值得保留并展示的实验图片
├── LEARNING-PROGRESS.md
└── README.md
```

## 主线课程

| 序号 | 主题 | 核心内容 | 文件 |
| ---: | --- | --- | --- |
| 01 | Tensor 基础 | 张量创建、形状、索引和常见运算 | [01-tensor-basics.py](lessons/01-tensor-basics.py) |
| 02 | Broadcasting | 广播规则与维度对齐 | [02-broadcasting.py](lessons/02-broadcasting.py) |
| 03 | NumPy 与 Tensor | 数据转换、复制和共享内存 | [03-numpy-and-tensor.py](lessons/03-numpy-and-tensor.py) |
| 04 | Autograd | 计算图、反向传播和梯度 | [04-autograd.py](lessons/04-autograd.py) |
| 05 | `nn.Module` | 自定义模型、参数与状态字典 | [05-module-basics.py](lessons/05-module-basics.py) |
| 06 | 激活函数与 MLP | 非线性激活和多层网络 | [06-activation-and-mlp.py](lessons/06-activation-and-mlp.py) |
| 07 | 常见学习任务 | 回归、二分类和多分类 | [07-learning-tasks.py](lessons/07-learning-tasks.py) |
| 08 | Dataset 与 DataLoader | 数据集封装、批处理和打乱 | [08-dataset-and-dataloader.py](lessons/08-dataset-and-dataloader.py) |
| 09 | 数据集划分 | 训练集、验证集与测试集 | [09-train-validation-test.py](lessons/09-train-validation-test.py) |
| 10 | 欠拟合与过拟合 | Dropout、权重衰减和早停 | [10-underfitting-and-overfitting.py](lessons/10-underfitting-and-overfitting.py) |
| 11 | Checkpoint | 保存、恢复训练与加载最佳模型 | [11-checkpoint-and-resume.py](lessons/11-checkpoint-and-resume.py) |
| 12 | CNN 基础 | 卷积、池化、展平和图像分类网络 | [12-conv2d-and-pooling.py](lessons/12-conv2d-and-pooling.py) |
| 13 | 模型评估 | 混淆矩阵、分类指标和错误样本分析 | [13-confusion-matrix-analysis.py](lessons/13-confusion-matrix-analysis.py) |
| 14 | 数据增强与归一化 | 随机增强、标准化、早停与训练曲线 | [14-data-augmentation-and-normalization.py](lessons/14-data-augmentation-and-normalization.py) |
| 15 | Adam 与学习率 | Adam/AdamW、学习率与优化过程 | [15-adam-lro.py](lessons/15-adam-lro.py) |
| 16 | ResNet | 残差分支、Shortcut 与形状匹配 | [16-residual-network.py](lessons/16-residual-network.py) |
| 17 | RNN/LSTM | 序列输出、隐藏状态与记忆单元 | [17-rnn-and-lstm.py](lessons/17-rnn-and-lstm.py) |
| 18 | 图神经网络 | GCN、GAT、GIN 与消息传递 | [18-graph-neural-networks.py](lessons/18-graph-neural-networks.py) |
| 19 | GPU 与混合精度 | Device、Autocast、GradScaler 和显存 | [19-device-and-mixed-precision.py](lessons/19-device-and-mixed-precision.py) |

预备科研能力见 [科研检索与人工智能会议入门](RESEARCH-FOUNDATIONS.md)，覆盖 arXiv、Hugging Face、Google Scholar、DBLP、ICML、NeurIPS 和 ICLR。

## 完整项目

### FashionMNIST CNN

[01-fashion-mnist-cnn.py](projects/01-fashion-mnist-cnn.py) 包含一条完整训练链路：

1. 下载并划分 FashionMNIST 数据集
2. 构建 CNN
3. 训练并保存验证集最佳模型
4. 在测试集上评估
5. 输出样本预测与置信度

课程 13 会在这个模型的基础上继续生成混淆矩阵和高置信度错误样本。

课程 14 进一步加入随机裁剪、翻转、旋转和归一化，并保存增强样本、训练曲线与测试预测，形成第二条完整实验链路。

## 练习区

`practice/` 保存学习早期的小实验。它们按知识点重新命名，适合快速复习某个概念，不作为主线课程重复阅读。

- 数据准备：CSV 自定义数据集
- 梯度与模型：手动梯度、`nn.Module`
- 训练组件：损失函数、优化器、DataLoader
- 任务示例：线性回归、二分类、多分类和 MLP

## 如何运行

建议始终从仓库根目录运行，这样数据、权重与输出路径保持一致：

```bash
python 01-pytorch-basics/lessons/01-tensor-basics.py
python 01-pytorch-basics/projects/01-fashion-mnist-cnn.py
python 01-pytorch-basics/lessons/13-confusion-matrix-analysis.py
python 01-pytorch-basics/lessons/14-data-augmentation-and-normalization.py
```

## 数据与实验产物

- `data/`：下载的数据集，仅保留在本地
- `*.pth`：模型权重与训练断点，仅保留在本地
- `outputs/`：适合在 GitHub 展示的分析图片

当前保留的分析结果：

- [原始混淆矩阵](outputs/confusion_matrix.png)
- [归一化混淆矩阵](outputs/normalized_confusion_matrix.png)
- [高置信度错误样本](outputs/wrong_predictions.png)
- [数据增强样本](outputs/data-augmentation/augmented_samples.png)
- [训练与验证损失曲线](outputs/data-augmentation/loss_curve.png)
- [训练与验证准确率曲线](outputs/data-augmentation/accuracy_curve.png)
- [数据增强模型预测](outputs/data-augmentation/test_predictions.png)

当前 FashionMNIST 测试结果：

| 指标 | 结果 |
| --- | ---: |
| 测试集准确率 | 91.38% |
| 识别最好类别 | Sandal（99.10% Recall） |
| 识别最弱类别 | Shirt（74.20% Recall） |

![归一化 FashionMNIST 混淆矩阵](outputs/normalized_confusion_matrix.png)

## 学到这里应当会什么

- [x] 前向传播、反向传播与 Autograd
- [x] Dataset、DataLoader、损失函数与优化器
- [x] MLP、CNN、ResNet 与 RNN/LSTM
- [x] GCN、GAT、GIN 的消息传递基础
- [x] GPU 设备迁移与混合精度
- [x] 训练/验证/测试、Checkpoint 与错误分析
- [x] 基础论文检索与会议认知

完成本阶段的核心内容后，进入 [阶段 2：Transformer 原理与手写实现](../02-transformer/README.md)。
