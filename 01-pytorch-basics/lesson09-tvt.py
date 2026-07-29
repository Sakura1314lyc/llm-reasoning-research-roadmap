import copy

import torch
from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)


torch.manual_seed(42)


# ==========================
# 1. 创建人工数据
# ==========================
class_0 = (
    torch.randn(100, 2) * 0.6
    + torch.tensor([-2.0, -1.0])
)

class_1 = (
    torch.randn(100, 2) * 0.6
    + torch.tensor([2.0, -1.0])
)

class_2 = (
    torch.randn(100, 2) * 0.6
    + torch.tensor([0.0, 2.0])
)

x = torch.cat(
    [class_0, class_1, class_2],
    dim=0
)

labels = torch.cat([
    torch.zeros(100, dtype=torch.long),
    torch.ones(100, dtype=torch.long),
    torch.full((100,), 2, dtype=torch.long)
])

dataset = TensorDataset(x, labels)


# ==========================
# 2. 划分数据集
# ==========================
split_generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15],
    generator=split_generator
)

print("完整数据集:", len(dataset))
print("训练集:", len(train_set))
print("验证集:", len(val_set))
print("测试集:", len(test_set))


# ==========================
# 3. 创建 DataLoader
# ==========================
train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)


# ==========================
# 4. 定义模型
# ==========================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Classifier().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)


# ==========================
# 5. 单轮训练函数
# ==========================
def train_one_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for batch_x, batch_labels in data_loader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = loss_fn(logits, batch_labels)

        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size
        sample_count += batch_size

        predictions = logits.argmax(dim=1)

        correct_count += (
            predictions == batch_labels
        ).sum().item()

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# ==========================
# 6. 评估函数
# ==========================
def evaluate(
    model,
    data_loader,
    loss_fn,
    device
):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for batch_x, batch_labels in data_loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_x)
            loss = loss_fn(logits, batch_labels)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            sample_count += batch_size

            predictions = logits.argmax(dim=1)

            correct_count += (
                predictions == batch_labels
            ).sum().item()

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# ==========================
# 7. 正式训练
# ==========================
best_val_loss = float("inf")
best_model_state = None

for epoch in range(1, 101):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        loss_fn,
        device
    )

    # 保存验证集损失最低时的参数
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"epoch={epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2%}"
        )


# ==========================
# 8. 加载验证集最优模型
# ==========================
assert best_model_state is not None

model.load_state_dict(best_model_state)


# ==========================
# 9. 最终测试
# ==========================
test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    loss_fn,
    device
)

print("\n最佳验证集损失:", best_val_loss)
print("测试集损失:", test_loss)
print(f"测试集准确率: {test_accuracy:.2%}")