"""手写 LoRA 的低秩增量：W + scale × B @ A。"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.a.T @ self.b.T)


def main() -> None:
    layer = LoRALinear(16, 32, rank=4)
    output = layer(torch.randn(2, 16))
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print("输出 / 可训练参数：", output.shape, trainable)
    assert output.shape == (2, 32)
    assert trainable == 4 * 16 + 32 * 4


if __name__ == "__main__":
    main()
