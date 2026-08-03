import torch
from torch import nn

torch.manual_seed(42)

X = torch.tensor([
    [1, 2, 3],
    [3, 2, 1],
    [5, 4, 3],
    [0, 1, 0]
], dtype=torch.float32
)

Y = torch.tensor([
    [1],
    [0],
    [1],
    [0]
], dtype=torch.float32
)

class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = BinaryModel()
logits = model(X)
Loss_fn = nn.BCEWithLogitsLoss()
loss = Loss_fn(logits, Y)

print("logits = ",  logits)
print("logits.shape = ", logits.shape)
print("Y.shape = ", Y.shape)
print("loss = ", loss)

#概率只用于观测
probability = torch.sigmoid(logits)
predictions = (probability >= 0.5).long()

print("probablity = ", probability)
print("predictions = ", predictions)

