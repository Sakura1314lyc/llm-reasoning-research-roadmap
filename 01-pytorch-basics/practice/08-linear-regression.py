import torch
from torch import nn

torch.manual_seed(42)


# 1. 准备数据
x = torch.arange(
    -5,
    6,
    dtype=torch.float32
).reshape(-1, 1)

y = 3 * x + 2


# 2. 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            in_features=1,
            out_features=1
        )

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()


# 3. 定义损失函数
loss_fn = nn.MSELoss()
 

# 4. 定义优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# 5. 训练
for epoch in range(501):
    optimizer.zero_grad()

    y_pred = model(x)

    loss = loss_fn(y_pred, y)

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()

        print(
            f"epoch={epoch:3d}, "
            f"loss={loss.item():.6f}, "
            f"weight={weight:.4f}, "
            f"bias={bias:.4f}"
        )


# 6. 查看最终参数
print("\n训练后的参数：")

for name, parameter in model.named_parameters():
    print(name, parameter)


# 7. 使用模型预测
test_x = torch.tensor([[10.0]])

model.eval()

with torch.no_grad():
    test_y = model(test_x)

print("\nx=10 时的预测结果：", test_y.item())
print("真实结果：", 3 * 10 + 2)