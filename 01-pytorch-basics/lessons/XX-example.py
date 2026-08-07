"""课程 14：用 FashionMNIST 实践数据增强与归一化。

本课为训练集添加随机裁剪、水平翻转和旋转，为训练/验证/测试集
统一应用标准化，并完成训练、早停、曲线绘制和预测结果展示。
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Subset
)
from torchvision import datasets
from torchvision.transforms import v2


# ============================================================
# 1. 全局配置
# ============================================================
SEED = 42

BATCH_SIZE = 128
EPOCHS = 50

LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4

# Early Stopping
PATIENCE = 10
MIN_DELTA = 1e-4

# 基于脚本位置构造路径，避免从不同工作目录启动时读写到错误位置。
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent

DATA_ROOT = REPOSITORY_ROOT / "data"

# 使用新的文件名，避免加载旧的未归一化模型
BEST_MODEL_PATH = (
    REPOSITORY_ROOT / "fashion_cnn_aug_best.pth"
)

OUTPUT_DIR = (
    MODULE_ROOT / "outputs" / "data-augmentation"
)


# FashionMNIST 十个类别
CLASS_NAMES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot"    # 9
]


# FashionMNIST 像素均值和标准差
MEAN = (0.2860,)
STD = (0.3530,)


# ============================================================
# 2. 固定随机种子
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 反归一化
#
# Normalize:
#     normalized = (image - mean) / std
#
# Denormalize:
#     image = normalized * std + mean
# ============================================================
def denormalize(
    image: torch.Tensor
) -> torch.Tensor:
    mean = torch.tensor(
        MEAN,
        dtype=image.dtype,
        device=image.device
    ).view(-1, 1, 1)

    std = torch.tensor(
        STD,
        dtype=image.dtype,
        device=image.device
    ).view(-1, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


# ============================================================
# 4. 创建 Dataset 和 DataLoader
# ============================================================
def build_dataloaders(
    device: torch.device
):
    # --------------------------------------------------------
    # 训练集预处理：随机增强 + 归一化
    # --------------------------------------------------------
    train_transform = v2.Compose([
        # PIL Image → Image Tensor
        v2.ToImage(),

        # 先填充到 32×32，再随机裁剪为 28×28
        # 相当于产生轻微的位置变化
        v2.RandomCrop(
            size=(28, 28),
            padding=2
        ),

        # 50% 概率水平翻转
        v2.RandomHorizontalFlip(
            p=0.5
        ),

        # 在 -8° 到 8° 之间随机旋转
        v2.RandomRotation(
            degrees=8
        ),

        # 转成 float32，并缩放到 [0,1]
        v2.ToDtype(
            torch.float32,
            scale=True
        ),

        # 标准化
        v2.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    # --------------------------------------------------------
    # 验证集和测试集：
    # 不使用随机增强，只做格式转换与归一化
    # --------------------------------------------------------
    eval_transform = v2.Compose([
        v2.ToImage(),

        v2.ToDtype(
            torch.float32,
            scale=True
        ),

        v2.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    # --------------------------------------------------------
    # 训练版本
    # --------------------------------------------------------
    train_source = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        transform=train_transform,
        download=True
    )

    # --------------------------------------------------------
    # 验证版本
    #
    # 和 train_source 使用相同原始图片，
    # 但 transform 不同
    # --------------------------------------------------------
    val_source = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        transform=eval_transform,
        download=True
    )

    # --------------------------------------------------------
    # 官方测试集
    # --------------------------------------------------------
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        transform=eval_transform,
        download=True
    )

    # --------------------------------------------------------
    # 生成固定划分下标
    # --------------------------------------------------------
    split_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    all_indices = torch.randperm(
        len(train_source),
        generator=split_generator
    ).tolist()

    # 前 54000 张用于训练
    train_indices = all_indices[:54000]

    # 后 6000 张用于验证
    val_indices = all_indices[54000:]

    # --------------------------------------------------------
    # 构造 Subset
    # --------------------------------------------------------
    train_dataset = Subset(
        train_source,
        train_indices
    )

    val_dataset = Subset(
        val_source,
        val_indices
    )

    # CUDA 下使用锁页内存
    pin_memory = device.type == "cuda"

    # 单独控制训练集 shuffle
    shuffle_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=shuffle_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    return (
        train_source,
        train_indices,
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# 5. CNN 模型
# ============================================================
class BetterFashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # =====================
            # Block 1
            # =====================
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =====================
            # Block 2
            # =====================
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =====================
            # Block 3
            # =====================
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Dropout(p=0.2),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 6. 单轮训练
# ============================================================
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # 清空上一批次的梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)

        # 计算交叉熵损失
        loss = loss_fn(
            logits,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        sample_count += batch_size

        predictions = logits.argmax(dim=1)

        correct_count += (
            predictions == labels
        ).sum().item()

    if sample_count == 0:
        raise RuntimeError(
            "训练数据集为空。"
        )

    average_loss = (
        total_loss / sample_count
    )

    accuracy = (
        correct_count / sample_count
    )

    return average_loss, accuracy


# ============================================================
# 7. 验证或测试
# ============================================================
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = loss_fn(
                logits,
                labels
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

            predictions = logits.argmax(dim=1)

            correct_count += (
                predictions == labels
            ).sum().item()

    if sample_count == 0:
        raise RuntimeError(
            "评估数据集为空。"
        )

    average_loss = (
        total_loss / sample_count
    )

    accuracy = (
        correct_count / sample_count
    )

    return average_loss, accuracy


# ============================================================
# 8. 显示同一张图片的多次随机增强
# ============================================================
def show_augmented_samples(
    train_source,
    sample_index: int
) -> None:
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(10, 6)
    )

    for axis in axes.flatten():
        # 每次访问相同 index，
        # 都会重新执行随机增强
        image, label = train_source[
            sample_index
        ]

        # 显示之前需要反归一化
        image = denormalize(image)

        axis.imshow(
            image.squeeze(0).numpy(),
            cmap="gray"
        )

        axis.set_title(
            CLASS_NAMES[label]
        )

        axis.axis("off")

    figure.suptitle(
        "Same Image with Random Augmentation"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "augmented_samples.png",
        dpi=200,
        bbox_inches="tight"
    )


# ============================================================
# 9. 绘制损失与准确率曲线
# ============================================================
def plot_training_history(
    history: dict[str, list[float]]
) -> None:
    epoch_numbers = range(
        1,
        len(history["train_loss"]) + 1
    )

    # --------------------------------------------------------
    # 损失曲线
    # --------------------------------------------------------
    loss_figure, loss_axis = plt.subplots(
        figsize=(8, 5)
    )

    loss_axis.plot(
        epoch_numbers,
        history["train_loss"],
        label="Train Loss"
    )

    loss_axis.plot(
        epoch_numbers,
        history["val_loss"],
        label="Validation Loss"
    )

    loss_axis.set_title(
        "Training and Validation Loss"
    )

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()

    loss_figure.tight_layout()

    loss_figure.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    # --------------------------------------------------------
    # 准确率曲线
    # --------------------------------------------------------
    accuracy_figure, accuracy_axis = (
        plt.subplots(figsize=(8, 5))
    )

    accuracy_axis.plot(
        epoch_numbers,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    accuracy_axis.plot(
        epoch_numbers,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    accuracy_axis.set_title(
        "Training and Validation Accuracy"
    )

    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()

    accuracy_figure.tight_layout()

    accuracy_figure.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=200,
        bbox_inches="tight"
    )


# ============================================================
# 10. 显示测试集前十张图片的预测
# ============================================================
def show_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> None:
    images, labels = next(
        iter(test_loader)
    )

    images_gpu = images.to(
        device,
        non_blocking=True
    )

    model.eval()

    with torch.inference_mode():
        logits = model(images_gpu)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = logits.argmax(
            dim=1
        )

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(13, 6)
    )

    for index, axis in enumerate(
        axes.flatten()
    ):
        image = denormalize(
            images[index]
        )

        true_label = labels[index].item()

        predicted_label = (
            predictions[index].item()
        )

        confidence = probabilities[
            index,
            predicted_label
        ].item()

        axis.imshow(
            image.squeeze(0).numpy(),
            cmap="gray"
        )

        axis.set_title(
            f"True: "
            f"{CLASS_NAMES[true_label]}\n"
            f"Pred: "
            f"{CLASS_NAMES[predicted_label]}\n"
            f"Conf: {confidence:.2%}",
            fontsize=9
        )

        axis.axis("off")

    figure.suptitle(
        "FashionMNIST Predictions"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / "test_predictions.png",
        dpi=200,
        bbox_inches="tight"
    )


# ============================================================
# 11. 主程序
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("训练设备:", device)

    # --------------------------------------------------------
    # 创建数据集与 DataLoader
    # --------------------------------------------------------
    (
        train_source,
        train_indices,
        train_loader,
        val_loader,
        test_loader
    ) = build_dataloaders(device)

    print(
        "训练集:",
        len(train_loader.dataset)
    )

    print(
        "验证集:",
        len(val_loader.dataset)
    )

    print(
        "测试集:",
        len(test_loader.dataset)
    )

    print(
        "train_loader 批次数:",
        len(train_loader)
    )

    print(
        "val_loader 批次数:",
        len(val_loader)
    )

    print(
        "test_loader 批次数:",
        len(test_loader)
    )

    # --------------------------------------------------------
    # 查看一个 batch
    # --------------------------------------------------------
    sample_images, sample_labels = next(
        iter(train_loader)
    )

    print(
        "\n图片 batch 形状:",
        sample_images.shape
    )

    print(
        "标签 batch 形状:",
        sample_labels.shape
    )

    print(
        "图片 dtype:",
        sample_images.dtype
    )

    print(
        "标签 dtype:",
        sample_labels.dtype
    )

    print(
        "归一化后的最小值:",
        sample_images.min().item()
    )

    print(
        "归一化后的最大值:",
        sample_images.max().item()
    )

    # --------------------------------------------------------
    # 展示同一图片的多次增强结果
    # --------------------------------------------------------
    show_augmented_samples(
        train_source=train_source,
        sample_index=train_indices[0]
    )

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------
    model = BetterFashionCNN().to(device)
    # CrossEntropyLoss 输入 logits，
    # 不需要提前使用 Softmax
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    
    print(
        "\n模型参数总量:",
        parameter_count
    )

    # --------------------------------------------------------
    # 正式训练
    # --------------------------------------------------------
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # ----------------------------------------------------
        # 判断验证损失是否真正改善
        # ----------------------------------------------------
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            epochs_without_improvement = 0

            # 保存最佳训练状态
            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_loss":
                        best_val_loss,

                    "best_val_accuracy":
                        best_val_accuracy,

                    "mean": MEAN,
                    "std": STD
                },
                BEST_MODEL_PATH
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:2d}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2%} | "
            f"wait="
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )
        scheduler.step()

        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------
        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                f"\n验证损失连续 {PATIENCE} "
                "轮没有明显改善。"
            )

            print(
                f"训练在第 {epoch} 轮"
                "提前停止。"
            )

            break

    # --------------------------------------------------------
    # 检查最佳模型文件
    # --------------------------------------------------------
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"没有找到最佳模型文件："
            f"{BEST_MODEL_PATH.resolve()}"
        )

    # --------------------------------------------------------
    # 加载最佳模型
    # --------------------------------------------------------
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # 在测试集上最终评价
    # --------------------------------------------------------
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    print("\n" + "=" * 55)
    print("训练完成")
    print("=" * 55)

    print(
        "最佳轮次:",
        checkpoint["epoch"]
    )

    print(
        "最佳验证损失:",
        checkpoint["best_val_loss"]
    )

    print(
        f"最佳验证准确率: "
        f"{checkpoint['best_val_accuracy']:.2%}"
    )

    print(
        f"测试损失: {test_loss:.4f}"
    )

    print(
        f"测试准确率: {test_accuracy:.2%}"
    )

    # --------------------------------------------------------
    # 绘制训练曲线
    # --------------------------------------------------------
    plot_training_history(history)

    # --------------------------------------------------------
    # 展示预测结果
    # --------------------------------------------------------
    show_predictions(
        model=model,
        test_loader=test_loader,
        device=device
    )

    print(
        "\n最佳模型保存位置:",
        BEST_MODEL_PATH.resolve()
    )

    print(
        "分析图片保存位置:",
        OUTPUT_DIR.resolve()
    )

    # 显示所有图像窗口
    plt.show()


# ============================================================
# 12. 程序入口
# ============================================================
if __name__ == "__main__":
    main()
