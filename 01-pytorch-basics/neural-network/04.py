import torch
from torch import nn


model = nn.Linear(4, 3)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

