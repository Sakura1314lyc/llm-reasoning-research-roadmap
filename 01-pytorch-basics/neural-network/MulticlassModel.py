import torch
from torch import nn

torch.manual_seed(42)

x = torch.randn(4, 5)

labels = torch.tensor([
    0,
    2,
    1,
    2
])

class MultiClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
    def forward(self, x):
        return self.linear(x)

model = MultiClassModel()

logits = model(x)

Loss_fn = nn.CrossEntropyLoss()
loss = Loss_fn(logits, labels)
print("logits:")
print(logits)

print("logits.shape:", logits.shape)
print("labels.shape:", labels.shape)
print("labels.dtype:", labels.dtype)
print("loss:", loss)

# 找到每一行中分数最大的类别
predictions = logits.argmax(dim=1)

print("预测类别:", predictions)
print("真实类别:", labels)

accuracy = (predictions == labels).float().mean()

print("准确率:", accuracy.item())