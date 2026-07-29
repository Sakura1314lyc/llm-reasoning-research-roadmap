import torch
from torch import nn

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

labels = torch.tensor(
    [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    ],
    dtype=torch.long
)

print("x.shape is ", x.shape)
print("labels.shape is", labels.shape)
print("labels.dtype is ", labels.dtype)
# 2. 定义模型
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


# 3. 损失函数
loss_fn = nn.CrossEntropyLoss()


# 4. 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)


# 5. 训练
model.train()

for epoch in range(1, 501):
    # 清空梯度
    optimizer.zero_grad()

    # 前向传播，得到 logits
    logits = model(x)
    if epoch == 1:
        print("logits.shape is ", logits.shape)
    # 直接使用 logits 计算交叉熵
    loss = loss_fn(logits, labels)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 50 == 0:
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


# 6. 最终评估
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
print("\n最终 logits:")
print(final_logits)

print("\n最终概率:")
print(final_probabilities)

print("\n预测类别:")
print(final_predictions)

print("\n真实类别:")
print(labels)

print(f"\n最终准确率: {final_accuracy:.2%}")

