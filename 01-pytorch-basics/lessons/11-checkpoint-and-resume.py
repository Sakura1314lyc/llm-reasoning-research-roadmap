"""课程 11：模型 Checkpoint 与断点续训。

同时保存模型、优化器和训练进度，并从最近断点继续训练。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import(
    TensorDataset,
    DataLoader,
    random_split
)

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

checkpoint_path = REPOSITORY_ROOT / "latest_checkpoint.pth"
best_model_path = REPOSITORY_ROOT / "best_model.pth"

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

model = Classifier().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
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
max_epochs = 200

start_epoch = 1
best_epoch = 0
best_val_loss = float("inf")


# =========================
# 加载训练断点
# =========================
if checkpoint_path.exists():
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    best_epoch = checkpoint.get(
        "best_epoch",
        0
    )

    print(
        f"checkpoint 位于第 "
        f"{start_epoch - 1} 轮"
    )

    print(
        f"将从第 {start_epoch} 轮继续训练"
    )

    print(
        "恢复后的学习率:",
        optimizer.param_groups[0]["lr"]
    )
else:
    print("没有发现 checkpoint，从头开始训练")


# =========================
# 恢复或开始训练
# =========================
if start_epoch > max_epochs:
    print(
        f"模型已经训练到第 "
        f"{start_epoch - 1} 轮，"
        f"目标轮数为 {max_epochs}。"
    )
else:
    for epoch in range(
        start_epoch,
        max_epochs + 1
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device
            )
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_fn,
            device
        )

        # 保存验证集最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                best_model_path
            )

        # 保存最新训练断点
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_val_loss":
                    best_val_loss,
                "best_epoch":
                    best_epoch
            },
            checkpoint_path
        )

        if (
            epoch == start_epoch
            or epoch % 10 == 0
        ):
            print(
                f"epoch={epoch:3d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_accuracy:.2%} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.2%}"
            )


# =========================
# 加载最优模型并最终测试
# =========================
if not best_model_path.exists():
    raise FileNotFoundError(
        f"最佳模型文件不存在："
        f"{best_model_path.resolve()}"
    )

best_state_dict = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(best_state_dict)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    loss_fn,
    device
)

print("\n最佳模型轮次:", best_epoch)
print("最佳验证损失:", best_val_loss)
print(f"测试集损失: {test_loss:.6f}")
print(f"测试集准确率: {test_accuracy:.2%}")
