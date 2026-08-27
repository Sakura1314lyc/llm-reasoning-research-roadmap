"""用一个残差块看清 shortcut 何时需要调整形状。"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """两层卷积残差块；尺寸或通道变化时使用 1×1 shortcut。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


def main() -> None:
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    block = ResidualBlock(16, 32, stride=2)
    output = block(x)
    output.mean().backward()

    print("输入：", x.shape)
    print("主分支 + shortcut 输出：", output.shape)
    print("输入梯度：", x.grad.shape)

    assert output.shape == (2, 32, 16, 16)
    assert x.grad is not None


if __name__ == "__main__":
    main()
