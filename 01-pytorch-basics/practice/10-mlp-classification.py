import torch
from torch import nn

torch.manual_seed(42)


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


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 3)
        )

    def forward(self, x):
        return self.network(x)


model = Classifier()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)


print("x.shape:", x.shape)
print("labels.shape:", labels.shape)
print("labels.dtype:", labels.dtype)


# -----------------------
# 训练前评估
# -----------------------
model.eval()

with torch.no_grad():
    initial_logits = model(x)

    initial_predictions = initial_logits.argmax(dim=1)

    initial_accuracy = (
        (initial_predictions == labels)
        .float()
        .mean()
        .item()
    )

print("\n训练前 logits.shape:")
print(initial_logits.shape)

print("训练前预测:")
print(initial_predictions)

print(f"训练前准确率: {initial_accuracy:.2%}")


# -----------------------
# 模型训练
# -----------------------
model.train()

for epoch in range(1, 501):
    # 1. 清空旧梯度
    optimizer.zero_grad()

    # 2. 前向传播
    logits = model(x)

    # 3. 计算交叉熵损失
    loss = loss_fn(logits, labels)

    # 4. 反向传播
    loss.backward()

    # 5. 更新参数
    optimizer.step()

    if epoch % 50 == 0:
        # 这里使用当前一轮前向传播得到的 logits
        predictions = logits.argmax(dim=1)

        accuracy = (
            (predictions == labels)
            .float()
            .mean()
            .item()
        )

        print(
            f"epoch={epoch:3d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.2%}"
        )


# -----------------------
# 最终评估
# -----------------------
model.eval()

with torch.no_grad():
    final_logits = model(x)

    final_probabilities = torch.softmax(
        final_logits,
        dim=1
    )

    final_predictions = final_logits.argmax(dim=1)

    final_accuracy = (
        (final_predictions == labels)
        .float()
        .mean()
        .item()
    )


print("\n训练后预测:")
print(final_predictions)

print("\n真实类别:")
print(labels)

print(f"\n最终准确率: {final_accuracy:.2%}")


# -----------------------
# 分析第一个样本
# -----------------------
first_logits = final_logits[0]
first_probabilities = final_probabilities[0]

print("\n第一个样本的 logits:")
print(first_logits)

print("\n第一个样本的概率:")
print(first_probabilities)

print("\n第一个样本概率之和:")
print(first_probabilities.sum())

print(
    "概率和是否接近1:",
    torch.isclose(
        first_probabilities.sum(),
        torch.tensor(1.0)
    ).item()
)

print(
    "第一个样本预测类别:",
    first_logits.argmax(dim=0).item()
)

print(
    "第一个样本真实类别:",
    labels[0].item()
)