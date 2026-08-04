"""课程 14：图像数据增强与归一化。

本课使用一张随机生成的灰度图演示两件事：
1. 训练阶段的数据增强具有随机性；
2. 归一化会按照 (x - mean) / std 调整数据分布。

示例不需要下载数据集，可以直接运行。
"""

import torch
from torchvision.transforms import v2


IMAGE_SIZE = 28
MEAN = (0.5,)
STD = (0.5,)


def build_train_transform() -> v2.Compose:
    """构建训练变换：先随机增强，再转为浮点数并归一化。"""
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD)
    ])


def build_eval_transform() -> v2.Compose:
    """构建评估变换：不使用随机增强，保证结果可复现。"""
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD)
    ])


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    """打印张量的形状和值域，帮助观察变换前后的差异。"""
    # mean() 不支持 uint8，统计时统一转为 float，不改变原张量。
    tensor_for_statistics = tensor.float()

    print(
        f"{name:<16} | shape={tuple(tensor.shape)} | "
        f"min={tensor.min().item():>7.3f} | "
        f"max={tensor.max().item():>7.3f} | "
        f"mean={tensor_for_statistics.mean().item():>7.3f}"
    )


def main() -> None:
    torch.manual_seed(42)

    # torchvision 常用的单张灰度图格式为 [C, H, W]。
    # uint8 的像素范围是 [0, 255]。
    image = torch.randint(
        low=0,
        high=256,
        size=(1, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.uint8
    )

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    # 同一张图经过随机训练变换，结果通常不同。
    augmented_image_1 = train_transform(image)
    augmented_image_2 = train_transform(image)

    # 评估变换没有随机操作，相同输入会得到相同输出。
    eval_image_1 = eval_transform(image)
    eval_image_2 = eval_transform(image)

    describe_tensor("原始图像", image)
    describe_tensor("训练变换 1", augmented_image_1)
    describe_tensor("训练变换 2", augmented_image_2)
    describe_tensor("评估变换", eval_image_1)

    print(
        "训练变换结果相同：",
        torch.equal(augmented_image_1, augmented_image_2)
    )
    print(
        "评估变换结果相同：",
        torch.equal(eval_image_1, eval_image_2)
    )

    # 原像素先缩放到 [0, 1]，再执行 (x - 0.5) / 0.5，
    # 因此归一化后的理论范围是 [-1, 1]。
    assert eval_image_1.min() >= -1.0
    assert eval_image_1.max() <= 1.0
    assert torch.equal(eval_image_1, eval_image_2)


if __name__ == "__main__":
    main()
