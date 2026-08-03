"""项目 01：使用 CNN 完成 FashionMNIST 图像分类。

覆盖数据准备、训练、验证、最佳权重保存、测试和样本预测。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    random_split
)
from torchvision import datasets
from torchvision.transforms import v2


# =========================
# 1. 基本设置
# =========================
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent

DATA_ROOT = REPOSITORY_ROOT / "data"
BEST_MODEL_PATH = REPOSITORY_ROOT / "fashion_cnn_best.pth"

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("训练设备:", device)


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# =========================
# 2. 图像预处理
# =========================
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(
        torch.float32,
        scale=True
    )
])


# =========================
# 3. 下载数据
# =========================
full_train_dataset = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=False,
    transform=transform,
    download=True
)


# =========================
# 4. 划分训练集和验证集
# =========================
split_generator = (
    torch.Generator().manual_seed(42)
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [54000, 6000],
    generator=split_generator
)


# =========================
# 5. DataLoader
# =========================
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# =========================
# 6. CNN 模型
# =========================
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


model = FashionCNN().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 7. 单轮训练
# =========================
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

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item() * current_batch_size
        )

        sample_count += current_batch_size

        predictions = logits.argmax(dim=1)

        correct_count += (
            predictions == labels
        ).sum().item()

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# =========================
# 8. 评估
# =========================
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
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            current_batch_size = images.size(0)

            total_loss += (
                loss.item() * current_batch_size
            )

            sample_count += current_batch_size

            predictions = logits.argmax(dim=1)

            correct_count += (
                predictions == labels
            ).sum().item()

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# =========================
# 9. 正式训练
# =========================
best_model_path = BEST_MODEL_PATH

best_val_loss = float("inf")
best_epoch = 0

epochs = 50

for epoch in range(1, epochs + 1):
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            best_model_path
        )

    print(
        f"epoch={epoch:2d}/{epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_accuracy:.2%} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_accuracy:.2%}"
    )


# =========================
# 10. 最终测试
# =========================
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

print("\n最佳轮次:", best_epoch)
print("最佳验证损失:", best_val_loss)
print(f"测试损失: {test_loss:.4f}")
print(f"测试准确率: {test_accuracy:.2%}")


# =========================
# 11. 查看前10个预测
# =========================
test_images, test_labels = next(
    iter(test_loader)
)

test_images = test_images.to(device)
test_labels = test_labels.to(device)

model.eval()

with torch.no_grad():
    logits = model(test_images)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = logits.argmax(dim=1)


print("\n前10个预测结果:")

for index in range(10):
    true_label = test_labels[index].item()
    predicted_label = predictions[index].item()

    confidence = (
        probabilities[index, predicted_label].item()
    )

    print(
        f"样本 {index:2d} | "
        f"真实: {class_names[true_label]:12s} | "
        f"预测: {class_names[predicted_label]:12s} | "
        f"置信度: {confidence:.6%}"
    )
