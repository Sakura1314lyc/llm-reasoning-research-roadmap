
import torch
from torch import nn


torch.manual_seed(42)

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)
model = Mymodel()
x = torch.tensor([
    [1, 2, 3],
    [4, 5 ,6]
], dtype=torch.float32
)

y = torch.tensor([
    [1, 0],
    [0, 1]
], dtype=torch.float32
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.03
)
Loss_fn = nn.MSELoss()
#开一个model.train()


model.train()
for epoch in range(301):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = Loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(
            f"epoch = {epoch + 1}, "
            f"loss = {loss.item():.6f}"
        )
model.eval()
with torch.no_grad():
    final_pred = model(x)
    final_loss = Loss_fn(final_pred, y)
print("\n最终预测:")
print(final_pred)

print("目标值:")
print(y)

print(f"最终损失:{final_loss.item():.6f}")