"""TensorDataset、DataLoader、batch 和 shuffle 的最小示例。"""

import torch
from torch.utils.data import DataLoader, TensorDataset


features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
labels = torch.arange(10, dtype=torch.long)

dataset = TensorDataset(features, labels)

# 固定生成器种子，使 shuffle 后的顺序可以复现。
generator = torch.Generator().manual_seed(42)

data_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    generator=generator,
)

print("dataset size:", len(dataset))
print("batch count:", len(data_loader))

for batch_index, (batch_features, batch_labels) in enumerate(data_loader):
    print(f"\nbatch {batch_index}")
    print("features:\n", batch_features)
    print("labels:", batch_labels)
