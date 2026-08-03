"""课程 13：FashionMNIST 模型评估与错误分析。

生成分类报告、原始/归一化混淆矩阵和高置信度错误样本图。
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


# ============================================================
# 1. 基本配置
# ============================================================
SEED = 42

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

# True：强制重新训练
# False：如果最佳模型文件存在，就直接加载
FORCE_TRAIN = False

# 基于脚本位置构造路径，避免因当前工作目录不同而读写到错误位置。
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent

DATA_ROOT = REPOSITORY_ROOT / "data"
BEST_MODEL_PATH = REPOSITORY_ROOT / "fashion_cnn_best.pth"
OUTPUT_DIR = MODULE_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# FashionMNIST 的类别名称
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


# ============================================================
# 2. 固定随机种子
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. 定义 CNN 模型
# ============================================================
class FashionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 输入：[N, 1, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # [N, 16, 28, 28]
            # → [N, 16, 14, 14]
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # [N, 32, 14, 14]
            # → [N, 32, 7, 7]
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            # [N, 32, 7, 7]
            # → [N, 1568]
            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 输出 10 个类别的 logits
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 4. 单轮训练函数
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

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)

        # 计算损失
        loss = loss_fn(
            logits,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item()
            * current_batch_size
        )

        sample_count += current_batch_size

        predictions = logits.argmax(dim=1)

        correct_count += (
            predictions == labels
        ).sum().item()

    if sample_count == 0:
        raise RuntimeError("训练集为空。")

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# ============================================================
# 5. 评估函数
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

            current_batch_size = images.size(0)

            total_loss += (
                loss.item()
                * current_batch_size
            )

            sample_count += current_batch_size

            predictions = logits.argmax(dim=1)

            correct_count += (
                predictions == labels
            ).sum().item()

    if sample_count == 0:
        raise RuntimeError("评估数据集为空。")

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


# ============================================================
# 6. 训练模型并保存验证集最佳参数
# ============================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    best_model_path: Path
) -> tuple[int, float]:
    best_epoch = 0
    best_val_loss = float("inf")

    # Early Stopping
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.2%} | "
            f"wait={epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"\n验证损失连续 {patience} 轮没有改善，"
                f"训练在第 {epoch} 轮提前停止。"
            )
            break

    if not best_model_path.exists():
        raise RuntimeError("最佳模型文件保存失败。")

    return best_epoch, best_val_loss


# ============================================================
# 7. 收集测试集预测、混淆矩阵与错误样本
# ============================================================
def collect_test_results(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str]
) -> tuple[torch.Tensor, list[dict]]:
    num_classes = len(class_names)

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    wrong_samples: list[dict] = []

    model.eval()

    with torch.inference_mode():
        for images, labels in data_loader:
            images_gpu = images.to(
                device,
                non_blocking=True
            )

            labels_gpu = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images_gpu)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = logits.argmax(dim=1)

            # 移动到 CPU，方便统计和绘图
            labels_cpu = labels_gpu.cpu()
            predictions_cpu = predictions.cpu()
            probabilities_cpu = probabilities.cpu()

            # ---------------------------------------------
            # 向量化构造混淆矩阵
            #
            # 真实类别 true，预测类别 pred：
            # 编码为 true * num_classes + pred
            # ---------------------------------------------
            encoded_indices = (
                labels_cpu * num_classes
                + predictions_cpu
            )

            batch_confusion_matrix = torch.bincount(
                encoded_indices,
                minlength=num_classes * num_classes
            ).reshape(
                num_classes,
                num_classes
            )

            confusion_matrix += batch_confusion_matrix

            # 找到当前 batch 中预测错误的样本
            wrong_mask = (
                predictions_cpu != labels_cpu
            )

            wrong_indices = torch.nonzero(
                wrong_mask,
                as_tuple=False
            ).flatten()

            for index in wrong_indices.tolist():
                true_label = labels_cpu[index].item()

                predicted_label = (
                    predictions_cpu[index].item()
                )

                confidence = probabilities_cpu[
                    index,
                    predicted_label
                ].item()

                wrong_samples.append({
                    "image": images[index].clone(),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })

    return confusion_matrix, wrong_samples


# ============================================================
# 8. 安全除法
# ============================================================
def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor
) -> torch.Tensor:
    return (
        numerator.float()
        / denominator.float().clamp_min(1)
    )


# ============================================================
# 9. 输出完整分类指标
# ============================================================
def print_classification_report(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
    top_error_count: int = 10
) -> dict[str, torch.Tensor]:
    num_classes = len(class_names)

    # 主对角线：各类别预测正确数量
    true_positive = confusion_matrix.diag()

    # 每一行的和：真实属于该类别的数量
    actual_count = confusion_matrix.sum(dim=1)

    # 每一列的和：被模型预测成该类别的数量
    predicted_count = confusion_matrix.sum(dim=0)

    total_count = confusion_matrix.sum()
    correct_count = true_positive.sum()
    wrong_count = total_count - correct_count

    overall_accuracy = safe_divide(
        correct_count,
        total_count
    )

    # Precision：
    # 预测为当前类别的样本中，有多少是真的
    precision = safe_divide(
        true_positive,
        predicted_count
    )

    # Recall：
    # 真实属于当前类别的样本中，有多少被找到
    recall = safe_divide(
        true_positive,
        actual_count
    )

    # F1：Precision 和 Recall 的调和平均
    f1_score = (
        2 * precision * recall
        / (precision + recall).clamp_min(1e-12)
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    print("\n" + "=" * 78)
    print("FashionMNIST 测试集分类报告")
    print("=" * 78)

    print(f"测试集样本总数：{total_count.item()}")
    print(f"预测正确数量：{correct_count.item()}")
    print(f"预测错误数量：{wrong_count.item()}")
    print(f"总体准确率：{overall_accuracy.item():.2%}")

    name_width = max(
        12,
        max(len(name) for name in class_names)
    )

    print("\n逐类别指标：")

    print(
        f"{'类别':<{name_width}} "
        f"{'正确/总数':>12} "
        f"{'Precision':>11} "
        f"{'Recall':>11} "
        f"{'F1':>11}"
    )

    print("-" * (name_width + 51))

    for index, class_name in enumerate(class_names):
        count_text = (
            f"{true_positive[index].item()}/"
            f"{actual_count[index].item()}"
        )

        print(
            f"{class_name:<{name_width}} "
            f"{count_text:>12} "
            f"{precision[index].item():>10.2%} "
            f"{recall[index].item():>10.2%} "
            f"{f1_score[index].item():>10.2%}"
        )

    print("-" * (name_width + 51))

    print(
        f"{'Macro Average':<{name_width}} "
        f"{'':>12} "
        f"{macro_precision.item():>10.2%} "
        f"{macro_recall.item():>10.2%} "
        f"{macro_f1.item():>10.2%}"
    )

    best_class_index = recall.argmax().item()
    worst_class_index = recall.argmin().item()

    print(
        "\n识别最好的类别："
        f"{class_names[best_class_index]} "
        f"({recall[best_class_index].item():.2%})"
    )

    print(
        "识别最差的类别："
        f"{class_names[worst_class_index]} "
        f"({recall[worst_class_index].item():.2%})"
    )

    # --------------------------------------------------------
    # 找最常见的错误类别组合
    # --------------------------------------------------------
    error_matrix = confusion_matrix.clone()

    # 将主对角线清零，排除正确预测
    error_matrix.fill_diagonal_(0)

    nonzero_error_count = (
        error_matrix > 0
    ).sum().item()

    actual_top_count = min(
        top_error_count,
        nonzero_error_count
    )

    print("\n最常见的错误类别组合：")

    if actual_top_count == 0:
        print("模型没有预测错误的样本。")
    else:
        top_values, top_indices = torch.topk(
            error_matrix.flatten(),
            k=actual_top_count
        )

        for rank, (
            error_number,
            flat_index
        ) in enumerate(
            zip(top_values, top_indices),
            start=1
        ):
            true_index = (
                flat_index.item()
                // num_classes
            )

            predicted_index = (
                flat_index.item()
                % num_classes
            )

            print(
                f"{rank:2d}. "
                f"真实 "
                f"{class_names[true_index]:12s} "
                f"→ 预测 "
                f"{class_names[predicted_index]:12s} "
                f": {error_number.item()} 张"
            )

    return {
        "overall_accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct_count": correct_count,
        "wrong_count": wrong_count
    }


# ============================================================
# 10. 绘制混淆矩阵
# ============================================================
def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
    normalized: bool,
    save_path: Path
) -> None:
    if normalized:
        row_total = confusion_matrix.sum(
            dim=1,
            keepdim=True
        )

        matrix_to_show = (
            confusion_matrix.float()
            / row_total.clamp_min(1)
        )

        title = (
            "Normalized FashionMNIST "
            "Confusion Matrix"
        )
    else:
        matrix_to_show = confusion_matrix.float()

        title = (
            "FashionMNIST Confusion Matrix"
        )

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    image = axis.imshow(
        matrix_to_show.numpy(),
        cmap="Blues"
    )

    figure.colorbar(
        image,
        ax=axis
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    axis.set_xticks(
        range(len(class_names))
    )

    axis.set_yticks(
        range(len(class_names))
    )

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    axis.set_yticklabels(
        class_names
    )

    maximum_value = matrix_to_show.max().item()
    threshold = maximum_value / 2

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix_to_show[
                row,
                column
            ].item()

            if normalized:
                display_text = f"{value:.1%}"
            else:
                display_text = str(int(value))

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            axis.text(
                column,
                row,
                display_text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color
            )

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )


# ============================================================
# 11. 打印高置信度错误
# ============================================================
def print_high_confidence_errors(
    wrong_samples: list[dict],
    class_names: list[str],
    count: int = 10
) -> list[dict]:
    if not wrong_samples:
        print("\n模型没有错误预测。")
        return []

    sorted_samples = sorted(
        wrong_samples,
        key=lambda sample: sample["confidence"],
        reverse=True
    )

    selected_samples = sorted_samples[:count]

    print("\n置信度最高的错误预测：")

    for index, sample in enumerate(
        selected_samples,
        start=1
    ):
        true_name = class_names[
            sample["true_label"]
        ]

        predicted_name = class_names[
            sample["predicted_label"]
        ]

        print(
            f"{index:2d}. "
            f"真实：{true_name:12s} | "
            f"预测：{predicted_name:12s} | "
            f"置信度：{sample['confidence']:.6%}"
        )

    return selected_samples


# ============================================================
# 12. 绘制高置信度错误图片
# ============================================================
def plot_wrong_samples(
    wrong_samples: list[dict],
    class_names: list[str],
    rows: int,
    columns: int,
    save_path: Path
) -> None:
    if rows <= 0 or columns <= 0:
        raise ValueError(
            "rows 和 columns 必须大于 0。"
        )

    if not wrong_samples:
        print("没有错误样本可以绘制。")
        return

    sorted_samples = sorted(
        wrong_samples,
        key=lambda sample: sample["confidence"],
        reverse=True
    )

    image_count = rows * columns

    selected_samples = sorted_samples[
        :image_count
    ]

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 3,
            rows * 3
        )
    )

    axes = axes.flatten()

    for index, axis in enumerate(axes):
        if index >= len(selected_samples):
            axis.axis("off")
            continue

        sample = selected_samples[index]

        image = sample["image"]

        # [1, 28, 28] → [28, 28]
        if (
            image.ndim == 3
            and image.size(0) == 1
        ):
            image = image.squeeze(0)

        true_name = class_names[
            sample["true_label"]
        ]

        predicted_name = class_names[
            sample["predicted_label"]
        ]

        confidence = sample["confidence"]

        axis.imshow(
            image.numpy(),
            cmap="gray"
        )

        axis.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}\n"
            f"Conf: {confidence:.2%}",
            fontsize=9
        )

        axis.axis("off")

    figure.suptitle(
        "Highest-confidence Wrong Predictions",
        fontsize=14
    )

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )


# ============================================================
# 13. 主函数
# ============================================================
def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("训练设备:", device)

    # --------------------------------------------------------
    # 图像预处理
    # --------------------------------------------------------
    transform = v2.Compose([
        v2.ToImage(),

        v2.ToDtype(
            torch.float32,
            scale=True
        )
    ])

    # --------------------------------------------------------
    # 下载并加载数据集
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 训练集划分为 54000 训练 + 6000 验证
    # --------------------------------------------------------
    split_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [54000, 6000],
        generator=split_generator
    )

    print("训练集数量:", len(train_dataset))
    print("验证集数量:", len(val_dataset))
    print("测试集数量:", len(test_dataset))

    pin_memory = device.type == "cuda"

    train_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=train_generator
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

    # --------------------------------------------------------
    # 查看一个 batch
    # --------------------------------------------------------
    sample_images, sample_labels = next(
        iter(train_loader)
    )

    print(
        "一个 batch 的图片形状:",
        sample_images.shape
    )

    print(
        "一个 batch 的标签形状:",
        sample_labels.shape
    )

    # --------------------------------------------------------
    # 创建模型、损失函数和优化器
    # --------------------------------------------------------
    model = FashionCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("模型参数总量:", parameter_count)

    # --------------------------------------------------------
    # 训练或加载模型
    # --------------------------------------------------------
    should_train = (
        FORCE_TRAIN
        or not BEST_MODEL_PATH.exists()
    )

    if should_train:
        print("\n开始训练模型……")

        best_epoch, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            best_model_path=BEST_MODEL_PATH
        )

        print("\n最佳模型轮次:", best_epoch)
        print("最佳验证损失:", best_val_loss)
    else:
        print(
            f"\n发现已有模型文件："
            f"{BEST_MODEL_PATH.resolve()}"
        )

        print("跳过训练，直接加载模型。")

    # --------------------------------------------------------
    # 加载验证集最佳模型
    # --------------------------------------------------------
    best_state_dict = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        best_state_dict
    )

    # --------------------------------------------------------
    # 最终测试
    # --------------------------------------------------------
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    print("\n最终测试结果：")
    print(f"测试损失：{test_loss:.4f}")
    print(f"测试准确率：{test_accuracy:.2%}")

    # --------------------------------------------------------
    # 收集混淆矩阵和错误样本
    # --------------------------------------------------------
    confusion_matrix, wrong_samples = (
        collect_test_results(
            model=model,
            data_loader=test_loader,
            device=device,
            class_names=CLASS_NAMES
        )
    )

    # --------------------------------------------------------
    # 输出分类报告
    # --------------------------------------------------------
    print_classification_report(
        confusion_matrix=confusion_matrix,
        class_names=CLASS_NAMES,
        top_error_count=10
    )

    print(
        "\n混淆矩阵每行样本数量：",
        confusion_matrix.sum(
            dim=1
        ).tolist()
    )

    print(
        "错误样本列表长度：",
        len(wrong_samples)
    )

    # --------------------------------------------------------
    # 输出高置信度错误
    # --------------------------------------------------------
    print_high_confidence_errors(
        wrong_samples=wrong_samples,
        class_names=CLASS_NAMES,
        count=10
    )

    # --------------------------------------------------------
    # 绘制原始混淆矩阵
    # --------------------------------------------------------
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=CLASS_NAMES,
        normalized=False,
        save_path=(
            OUTPUT_DIR
            / "confusion_matrix.png"
        )
    )

    # --------------------------------------------------------
    # 绘制归一化混淆矩阵
    # --------------------------------------------------------
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=CLASS_NAMES,
        normalized=True,
        save_path=(
            OUTPUT_DIR
            / "normalized_confusion_matrix.png"
        )
    )

    # --------------------------------------------------------
    # 绘制高置信度错误图片
    # --------------------------------------------------------
    plot_wrong_samples(
        wrong_samples=wrong_samples,
        class_names=CLASS_NAMES,
        rows=3,
        columns=4,
        save_path=(
            OUTPUT_DIR
            / "wrong_predictions.png"
        )
    )

    print(
        "\n分析图片已经保存到：",
        OUTPUT_DIR.resolve()
    )

    plt.show()


# ============================================================
# 14. 程序入口
# ============================================================
if __name__ == "__main__":
    main()
