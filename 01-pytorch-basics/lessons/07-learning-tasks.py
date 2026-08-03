"""课程 07：回归、二分类与多分类任务。

理解 logits、Sigmoid、Softmax 以及不同任务的输出形式。
"""

# Softmax 将多分类 logits 转换为概率分布。
import torch
from torch import nn

logits = torch.tensor([
    [2, 1, -0.5],
    [0.2, 0.5, 1.8]
], dtype=torch.float32
)

probabilities = torch.softmax(logits, dim = 1) #一般用于单独查看概率
print("logits:", logits)
print("概率为", probabilities)
print("每行概率之和为:", probabilities.sum(dim = 1))

# 使用CrossEntropyLoss 不用使用softmax, 因为前者里面已经包含了的
labels = torch.tensor([
    [2, 1, 0],
    [0.1, 0.41, 1.78]
], dtype=torch.float32
)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
print(loss)

# argmax 用于得到预测类别
logits1 = torch.tensor([
    [2.1, 0.3, -1.2],
    [1.7, 0.8, -0.4],
    [-0.2, 2.5, 0.6],
    [0.1, -0.7, 2.2]
])
pre_row = logits1.argmax(dim = 1) #返回索引
label = torch.tensor([0, 0, 2, 2])
print(pre_row) #输出每一行的最大值位于哪一列
p = (
    pre_row == label
).float().mean()
print("相似度:", p.item())
