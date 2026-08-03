"""演示一次完整的优化器更新步骤。"""

import torch
from torch import nn


torch.manual_seed(42)

model = nn.Linear(in_features=4, out_features=3)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)

features = torch.rand(5, 4)
targets = torch.rand(5, 3)

# 1. 清除上一次反向传播留下的梯度
optimizer.zero_grad()

# 2. 前向传播并计算损失
predictions = model(features)
loss = loss_fn(predictions, targets)

# 3. 反向传播，计算每个参数的梯度
loss.backward()

weight_before_step = model.weight.detach().clone()

# 4. 根据梯度和学习率更新参数
optimizer.step()

weight_changed = not torch.equal(
    weight_before_step,
    model.weight.detach(),
)

print(f"loss: {loss.item():.6f}")
print("weight updated:", weight_changed)
