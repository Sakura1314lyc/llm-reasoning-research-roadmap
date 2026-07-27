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
])


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 两个输入特征，三个类别
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        # 2. 返回原始 logits
        return self.linear(x)


model = Classifier()

# 3. 创建 CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 4. 创建 SGD，学习率设为 0.1
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

num_epochs = 201

# 5. 切换为训练模式
model.train()

for epoch in range(num_epochs):
    # 6. 清空梯度
    optimizer.zero_grad()

    # 7. 前向传播
    logits = model(x)

    # 8. 计算损失
    loss = loss_fn(logits, labels)

    # 9. 反向传播
    loss.backward()

    # 10. 更新参数
    optimizer.step()

    if epoch % 20 == 0:
        # 11. 得到预测类别
        predictions = logits.argmax(dim = 1)

        # 12. 计算准确率
        accuracy = (
            predictions == labels
        ).float().mean()

        print(
            f"epoch={epoch:3d}, "
            f"loss={loss.item():.4f}, "
            f"accuracy={accuracy.item():.2%}"
        )


# 13. 切换为评估模式
model.eval()

# 14. 关闭梯度计算
with torch.no_grad():
    final_logits = model(x)
    final_predictions = final_logits.argmax(dim = 1)
    final_accuracy = (final_predictions == labels).float().mean()

print("\n最终预测：", final_predictions)
print("真实标签：", labels)
print("最终准确率：", final_accuracy.item())