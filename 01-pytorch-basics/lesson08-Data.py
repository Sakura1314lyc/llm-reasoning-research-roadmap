from typing import Any

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


torch.manual_seed(42)


# 1. 准备数据
x = torch.tensor([
    [-2.0, -1.0],
    [-1.5, -1.0],
    [-2.0, -2.0],
    [-1.0, -1.5],

    [2.0, -1.0],
    [1.5, -1.0],
    [2.0, -2.0],
    [1.0, -1.5],

    [0.0, 2.0],
    [-0.5, 1.5],
    [0.5, 1.5],
    [0.0, 1.0]
])

labels = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
], dtype=torch.long)


# 2. 创建 Dataset
dataset = TensorDataset(x, labels)


# 3. 创建 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# 4. 定义模型
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


model = Classifier()


# 5. 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)


# 6. 训练
model.train()

for epoch in range(1, 201):
    epoch_loss = 0.0
    correct_count = 0
    sample_count = 0

    for batch_x, batch_labels in train_loader:
        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(batch_x)

        # 计算当前 batch 的损失
        loss = loss_fn(logits, batch_labels) ## 返回的是平均损失

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        epoch_loss += loss.item() * batch_x.size(0)

        # 计算当前 batch 的预测
        predictions = logits.argmax(dim=1)

        # 累计预测正确数量
        correct_count += (
            predictions == batch_labels
        ).sum().item()

        # 累计样本数量
        sample_count += batch_x.size(0)

    # 计算整个 epoch 的平均损失
    average_loss = epoch_loss / sample_count

    # 计算整个 epoch 的准确率
    accuracy = correct_count / sample_count

    if epoch % 20 == 0:
        print(
            f"epoch={epoch:3d}, "
            f"loss={average_loss:.6f}, "
            f"accuracy={accuracy:.2%}"
        )


# 7. 最终评估
model.eval()

with torch.no_grad():
    final_logits = model(x)
    final_predictions = final_logits.argmax(dim=1)

    final_accuracy = (
        (final_predictions == labels)
        .float()
        .mean()
        .item()
    )

print("\n最终预测:")
print(final_predictions)

print("\n真实标签:")
print(labels)

print(f"\n最终准确率: {final_accuracy:.2%}")

#自定义dataset

class MyDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.x = torch.tensor([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=torch.float32
        )
        self.lables = torch.tensor(
            [0, 1, 2],
            dtype=torch.long
        )
    def __len__(self): ##有多少个样本
        return len(self.x)
        #实际执行的是dataset.__len__()

    def __getitem__(self, index):
        #实际执行的是dataset.__getitem__()
        return self.x[index], self.lables[index]

dataSet = MyDataSet()
print("长度为 : ", len(dataSet))
sample_x, sample_label = dataSet[1]
print(sample_x, sample_label)

#Dataset 和 DataLoader通常在CPU上准备数据
#在进行训练时,再将每个batch 移动到GPU上