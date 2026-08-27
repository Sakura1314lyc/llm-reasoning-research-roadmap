"""把图片切成 Patch，再投影成视觉 Token。"""

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """用 kernel=stride=patch_size 的卷积完成分块与线性投影。"""

    def __init__(self, image_size: int, patch_size: int, channels: int, d_model: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        self.number_of_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(channels, d_model, patch_size, patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images)              # [B,D,H/P,W/P]
        return patches.flatten(2).transpose(1, 2)      # [B,N,D]


def main() -> None:
    images = torch.randn(2, 3, 224, 224)
    layer = PatchEmbedding(224, 16, 3, 192)
    tokens = layer(images)
    print("图片：", images.shape)
    print("Patch tokens：", tokens.shape)
    assert tokens.shape == (2, 196, 192)


if __name__ == "__main__":
    main()
