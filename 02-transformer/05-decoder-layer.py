"""实现 Encoder-Decoder 与 GPT 风格的 Decoder Layer。

标准 Decoder Layer 包含因果自注意力、交叉注意力和 FFN；
GPT 的 Decoder-only Layer 不需要交叉注意力。
"""

import torch
from torch import nn


def create_causal_mask(
    sequence_length: int,
    device: torch.device | None = None
) -> torch.Tensor:
    """创建上三角布尔掩码；True 表示禁止关注未来位置。"""
    return torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device
        ),
        diagonal=1
    )


class DecoderLayer(nn.Module):
    """采用 Pre-Norm 结构的标准 Transformer Decoder Layer。"""

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
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        source_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """接收 Decoder/Encoder 隐状态，返回输出和两组注意力权重。"""
        normalized_x = self.norm1(x)
        self_attention_output, self_attention_weights = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
            need_weights=True
        )
        x = x + self.dropout1(self_attention_output)

        normalized_x = self.norm2(x)
        cross_attention_output, cross_attention_weights = self.cross_attention(
            query=normalized_x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=source_padding_mask,
            need_weights=True
        )
        x = x + self.dropout2(cross_attention_output)

        x = x + self.dropout3(self.ffn(self.norm3(x)))

        return x, self_attention_weights, cross_attention_weights


class GPTDecoderLayer(nn.Module):
    """只包含因果自注意力和 FFN 的 Decoder-only Layer。"""

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
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized_x = self.norm1(x)
        attention_output, _ = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        x = x + self.attention_dropout(attention_output)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))

        return x


def main() -> None:
    torch.manual_seed(42)

    batch_size = 4
    target_length = 10
    source_length = 15
    d_model = 64

    decoder_input = torch.randn(batch_size, target_length, d_model)
    encoder_output = torch.randn(batch_size, source_length, d_model)
    causal_mask = create_causal_mask(target_length)

    decoder_layer = DecoderLayer(
        d_model=d_model,
        n_heads=8,
        d_ff=4 * d_model
    )

    output, self_weights, cross_weights = decoder_layer(
        x=decoder_input,
        encoder_output=encoder_output,
        causal_mask=causal_mask
    )

    print("Decoder 输入：", decoder_input.shape)
    print("Encoder 输出：", encoder_output.shape)
    print("Decoder 输出：", output.shape)
    print("自注意力权重：", self_weights.shape)
    print("交叉注意力权重：", cross_weights.shape)

    assert output.shape == decoder_input.shape
    assert self_weights.shape == (batch_size, target_length, target_length)
    assert cross_weights.shape == (batch_size, target_length, source_length)


if __name__ == "__main__":
    main()
