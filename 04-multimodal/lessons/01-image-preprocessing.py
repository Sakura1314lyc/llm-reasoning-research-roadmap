"""多模态课程 01：图像 Resize、浮点化、归一化与 Batch。

视觉模型通常接收 ``pixel_values: [B, C, H, W]``。本课使用随机 RGB 图片
演示通用预处理流程，不需要下载数据集或模型。真实预训练模型应优先使用其
``AutoImageProcessor`` 或 ``AutoProcessor``，因为不同模型的尺寸和统计量不同。
"""

import torch
from torchvision.transforms import v2


IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_image_transform() -> v2.Compose:
    """构造 [C, H, W] uint8 图片到标准化 float32 张量的变换。"""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            antialias=True
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])


def denormalize(pixel_values: torch.Tensor) -> torch.Tensor:
    """撤销 Normalize，方便将模型输入还原到 [0, 1] 进行观察。"""
    mean = torch.tensor(
        IMAGE_MEAN,
        dtype=pixel_values.dtype,
        device=pixel_values.device
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGE_STD,
        dtype=pixel_values.dtype,
        device=pixel_values.device
    ).view(3, 1, 1)

    return (pixel_values * std + mean).clamp(0.0, 1.0)


def main() -> None:
    torch.manual_seed(42)

    # 单张 RGB 图片使用 [C, H, W]；uint8 像素范围为 [0, 255]。
    image = torch.randint(
        low=0,
        high=256,
        size=(3, 240, 320),
        dtype=torch.uint8
    )

    transform = build_image_transform()
    pixel_values = transform(image)
    restored_image = denormalize(pixel_values)

    # 多张图片沿第 0 维堆叠，得到模型常用的 [B, C, H, W]。
    batch = torch.stack((pixel_values, pixel_values), dim=0)

    print("原始图片形状：", image.shape)
    print("原始 dtype / 值域：", image.dtype, (image.min().item(), image.max().item()))
    print("处理后形状：", pixel_values.shape)
    print(
        "处理后 dtype / 值域：",
        pixel_values.dtype,
        (pixel_values.min().item(), pixel_values.max().item())
    )
    print("Batch 形状：", batch.shape)
    print(
        "反归一化值域：",
        (restored_image.min().item(), restored_image.max().item())
    )

    assert pixel_values.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert pixel_values.dtype == torch.float32
    assert batch.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert restored_image.min() >= 0.0
    assert restored_image.max() <= 1.0


if __name__ == "__main__":
    main()
