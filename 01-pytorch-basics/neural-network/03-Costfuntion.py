import torch
from torch import nn
#回归任务
Loss_fn = nn.MSELoss(reduction="sum") # 指对平方误差和干什么

#二分类任务
#1 BCELoss()
Loss_fn1 = nn.BCELoss()
model = nn.Linear(4, 3)
x = torch.rand(5, 4)
y = torch.rand(5, 3)
logits = model(x)
probability = torch.sigmoid(logits)
loss = Loss_fn1(probability, y)
# 2 BCEWithLogitLoss()
model = nn.Linear(2, 3)
x1 = torch.tensor([1, 2, 3])
Logits = model(x1)
Loss_fn2 = nn.BCEWithLogitsLoss()
Loss = Loss_fn2(Logits, y)


#多分类任务
Loss_fn3 = nn.CrossEntropyLoss()
logits = model(x)
Losss = Loss_fn3(logits, y)