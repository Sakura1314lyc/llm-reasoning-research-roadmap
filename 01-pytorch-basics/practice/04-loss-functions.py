"""回归、二分类和多分类任务的常用损失函数示例。"""

import torch
from torch import nn


torch.manual_seed(42)


# 1. 回归：预测值和目标值都是连续数值
regression_predictions = torch.tensor([[2.5], [0.0], [2.0]])
regression_targets = torch.tensor([[3.0], [-0.5], [2.0]])

mse_loss = nn.MSELoss()
regression_loss = mse_loss(
    regression_predictions,
    regression_targets,
)


# 2. 二分类：BCEWithLogitsLoss 直接接收未经 Sigmoid 的 logits
binary_logits = torch.tensor([[1.2], [-0.7], [0.3]])
binary_targets = torch.tensor([[1.0], [0.0], [1.0]])

binary_loss_fn = nn.BCEWithLogitsLoss()
binary_loss = binary_loss_fn(binary_logits, binary_targets)
binary_probabilities = torch.sigmoid(binary_logits)


# 3. 多分类：每个样本输出 C 个 logits，标签是类别索引
multiclass_logits = torch.tensor([
    [2.0, 0.3, -1.0],
    [0.1, 1.5, 0.2],
    [-0.5, 0.4, 1.8],
])
multiclass_targets = torch.tensor([0, 1, 2], dtype=torch.long)

cross_entropy = nn.CrossEntropyLoss()
multiclass_loss = cross_entropy(
    multiclass_logits,
    multiclass_targets,
)
multiclass_probabilities = torch.softmax(multiclass_logits, dim=1)


print(f"MSE loss: {regression_loss.item():.6f}")
print(f"Binary loss: {binary_loss.item():.6f}")
print("Binary probabilities:\n", binary_probabilities)
print(f"Cross-entropy loss: {multiclass_loss.item():.6f}")
print("Multiclass probabilities:\n", multiclass_probabilities)
