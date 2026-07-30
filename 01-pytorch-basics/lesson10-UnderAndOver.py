#解决过拟合的方法: 权重衰减 weight_decay
# dropout : 神经网络里面的正则化层
import copy

import torch
from torch import nn
from torch.utils.data import(
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

print("len(train_loader) is ", len(train_loader))
print("len(val_loader) is ", len(val_loader))
print("len(test_loader) is ", len(test_loader))
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),# 30%概率置零

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 3)
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
    lr=0.01,
    weight_decay=1e-4 #权重损失:用于限制模型复杂度，限制参数大小
)


best_val_loss = float("inf")
best_model_state = None
best_epoch = 0

patience = 20
min_delta = 1e-4
epochs_without_improvement = 0

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



for epoch in range(1, 1001):
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

    # 判断验证损失是否真正改善
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"epoch={epoch:4d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2%} | "
            f"wait={epochs_without_improvement}/{patience}"
        )

    # 提前停止
    if epochs_without_improvement >= patience:
        print(
            f"\n验证损失连续 {patience} 轮没有明显改善。"
        )
        print(f"训练在第 {epoch} 轮提前停止。")
        break


assert best_model_state is not None

model.load_state_dict(best_model_state)


test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    loss_fn,
    device
)

print("\n最佳模型轮次:", best_epoch)
print("最佳验证集损失:", best_val_loss)
print("测试集损失:", test_loss)
print(f"测试集准确率: {test_accuracy:.2%}")