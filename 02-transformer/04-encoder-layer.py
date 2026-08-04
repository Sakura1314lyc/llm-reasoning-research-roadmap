"""使用 PyTorch MultiheadAttention 实现 Pre-Norm Encoder Layer。"""

import torch
from torch import nn


class EncoderLayer(nn.Module):
    """Self-Attention、FFN、残差连接和 LayerNorm 的组合。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """接收 [B, T, D]，返回编码结果和平均到各头的注意力权重。"""
        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=True
        )

        x = x + self.attention_dropout(attention_output)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))

        return x, attention_weights


def main() -> None:
    torch.manual_seed(42)

    x = torch.randn(4, 10, 64)
    encoder_layer = EncoderLayer(
        d_model=64,
        n_heads=8,
        d_ff=256,
        dropout=0.1
    )

    output, attention_weights = encoder_layer(x)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print("注意力权重形状：", attention_weights.shape)

    assert output.shape == x.shape
    assert attention_weights.shape == (4, 10, 10)


if __name__ == "__main__":
    main()
