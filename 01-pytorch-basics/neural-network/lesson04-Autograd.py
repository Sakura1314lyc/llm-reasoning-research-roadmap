import torch


x = torch.tensor(2.0, requires_grad=True)
y = 3 * x * x + 4 * x + 5
y.backward()
print(x)
print(y)
print(x.grad)