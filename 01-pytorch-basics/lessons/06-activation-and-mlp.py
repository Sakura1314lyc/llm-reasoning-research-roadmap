"""课程 06：激活函数与多层感知机。

比较常见激活函数，并用 ``nn.Sequential`` 组合多层网络。
"""

# 激活函数为神经网络引入非线性表达能力。
from typing import Any

import torch
from torch import nn

x = torch.tensor([
    [-2, -1, 0],
    [1, 2, 3]
], dtype=torch.float32
)

relu = nn.ReLU()
y_pred = relu(x)
print(y_pred)

#多层神经网络
class MyModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        #可以用Sequential 简化
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        # #print("输入:", x.shape)
        # x = self.linear1(x)
        # #print("第一层线性变换后, " , x.shape)
        # x = self.relu(x)
        # #print("RELU后, ", x.shape)
        # x = self.linear2(x)
        # #print("第二层线性变换后, ", x.shape)
        # return x
        return self.network(x)

    
model = MyModel()
loss_fn = nn.MSELoss()
target = torch.tensor([
    [-2, -1],
    [1, 2]
], dtype=torch.float32
)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)
for name, parameter in model.named_parameters():
    print(name)
    print(parameter.shape)
model.train()
for epoch in range(1, 501):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, target)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(
            f"epoch = {epoch}, "
            f"loss = {loss.item():.6f}"
        )
model.eval()
with torch.no_grad():
    final_pred = model(x)
    final_loss = loss_fn(final_pred, target)
print("最终预测:")
print(final_pred)
