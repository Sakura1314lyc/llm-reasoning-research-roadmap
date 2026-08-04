"""Transformer 的逐位置前馈神经网络（Feed-Forward Network）。

FFN 对序列中的每个 token 独立执行相同的两层 MLP：
[B, T, D] -> [B, T, D_FF] -> [B, T, D]。
"""

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """使用 GELU 激活和 Dropout 的 Transformer FFN。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model 和 d_ff 必须大于 0")

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 4
    d_model = 8

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        requires_grad=True
    )

    ffn = FeedForwardNetwork(
        d_model=d_model,
        d_ff=4 * d_model,
        dropout=0.1
    )

    output = ffn(x)

    # 简单反向传播，确认 FFN 的梯度链路完整。
    loss = output.square().mean()
    loss.backward()

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print("输入梯度形状：", x.grad.shape)

    assert output.shape == x.shape
    assert x.grad is not None


if __name__ == "__main__":
    main()
