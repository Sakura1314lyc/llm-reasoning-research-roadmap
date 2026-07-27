import torch
from torch import nn

#回归
model1 = nn.Linear(3, 2)
X1 = torch.rand(5, 3)
Loss_fn1 = nn.MSELoss()
Optimizer1 = torch.optim.SGD(
    model1.parameters(),
    lr= 0.01
)
Y1 =torch.randn(5, 2)
for epoch in range(100):
    Optimizer1.zero_grad()
    Y_pred = model1(X1)
    loss = Loss_fn1(Y_pred, Y1)
    loss.backward()
    Optimizer1.step()

#二分类
model2 = nn.Linear(3, 1)
X2 = torch.rand(5, 3)
Loss_fn2 = nn.BCEWithLogitsLoss()
Optimizer2 = torch.optim.SGD(
    model2.parameters(),
    lr= 0.01
)
Y2 =torch.randn(5, 2)
for epoch in range(100):
    Optimizer1.zero_grad()
    logits = model1(X2)
    loss = Loss_fn1(logits, Y2.float())
    loss.backward()
    Optimizer1.step()

#多分类
model3 = nn.Linear(3, 2)
X3 = torch.rand(5, 3)
Loss_fn3 = nn.CrossEntropyLoss()
Optimizer3 = torch.optim.SGD(
    model3.parameters(),
    lr= 0.01
)
Y3 =torch.randn(5, 2)
for epoch in range(100):
    Optimizer3.zero_grad()
    logits = model1(X3)
    loss = Loss_fn1(logits, Y3.long())
    loss.backward()
    Optimizer1.step()
