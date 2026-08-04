"""从零组合一个教学版 Encoder-Decoder Transformer。

本文件串联位置编码、多头注意力、FFN、Encoder、Decoder 和掩码，
重点展示各模块的数据流与张量形状，不包含真实数据的训练流程。
"""

import math

import torch
from torch import nn


# ============================================================
# 1. 位置编码
# ============================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码。

    输入和输出形状：
        [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("教学版要求 d_model 是偶数")

        # position: [max_seq_len, 1]
        position = torch.arange(
            max_seq_len,
            dtype=torch.float32
        ).unsqueeze(1)

        # div_term: [d_model / 2]
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        # pe: [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加上 batch 维度：
        # [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # 不参与训练，但会随着模型一起移动到 GPU
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


# ============================================================
# 2. 多头注意力
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    手动实现多头注意力。

    query: [B, Tq, D]
    key:   [B, Tk, D]
    value: [B, Tk, D]

    输出：
        [B, Tq, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model 必须能够被 n_heads 整除"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        [B, T, D]
            ↓
        [B, T, H, Dh]
            ↓
        [B, H, T, Dh]
        """

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        [B, H, T, Dh]
            ↓
        [B, T, H, Dh]
            ↓
        [B, T, D]
        """

        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        attention_mask:
            通常是因果掩码，形状 [Tq, Tk]

            True 表示这个位置不能看。

        key_padding_mask:
            padding 掩码，形状 [B, Tk]

            True 表示该 token 是 PAD，需要屏蔽。
        """

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 拆成多个头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # q: [B, H, Tq, Dh]
        # k: [B, H, Tk, Dh]
        #
        # scores: [B, H, Tq, Tk]
        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # 因果掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [Tq, Tk]
                #   ↓
                # [1, 1, Tq, Tk]
                attention_mask = attention_mask[
                    None,
                    None,
                    :,
                    :
                ]

            scores = scores.masked_fill(
                attention_mask,
                torch.finfo(scores.dtype).min
            )

        # Padding 掩码
        if key_padding_mask is not None:
            # [B, Tk]
            #   ↓
            # [B, 1, 1, Tk]
            padding_mask = key_padding_mask[
                :,
                None,
                None,
                :
            ]

            scores = scores.masked_fill(
                padding_mask,
                torch.finfo(scores.dtype).min
            )

        # 注意力概率
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # [B, H, Tq, Tk] @ [B, H, Tk, Dh]
        # =
        # [B, H, Tq, Dh]
        context = torch.matmul(
            attention_weights,
            v
        )

        # 合并多个头
        context = self.combine_heads(context)

        # 最后的输出投影
        output = self.out_proj(context)

        return output


# ============================================================
# 3. FFN
# ============================================================

class FeedForwardNetwork(nn.Module):
    """
    每个 token 独立经过同一个 MLP。

    D → d_ff → D
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# 4. Encoder Layer
# ============================================================

class EncoderLayer(nn.Module):
    """
    Pre-Norm Encoder Layer。

    x = x + SelfAttention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x:
            [B, T_src, D]

        source_padding_mask:
            [B, T_src]
        """

        # Self-Attention
        normalized_x = self.norm1(x)

        attention_output = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            key_padding_mask=source_padding_mask
        )

        x = x + self.dropout1(attention_output)

        # FFN
        normalized_x = self.norm2(x)

        ffn_output = self.ffn(normalized_x)

        x = x + self.dropout2(ffn_output)

        return x


# ============================================================
# 5. Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):
    """
    Pre-Norm Decoder Layer。

    1. Causal Self-Attention
    2. Cross-Attention
    3. FFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

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
    ) -> torch.Tensor:
        """
        x:
            Decoder 输入，[B, T_tgt, D]

        encoder_output:
            Encoder 输出，[B, T_src, D]

        causal_mask:
            [T_tgt, T_tgt]
        """

        # 1. Masked Self-Attention
        normalized_x = self.norm1(x)

        self_attention_output = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            attention_mask=causal_mask,
            key_padding_mask=target_padding_mask
        )

        x = x + self.dropout1(
            self_attention_output
        )

        # 2. Cross-Attention
        normalized_x = self.norm2(x)

        cross_attention_output = self.cross_attention(
            query=normalized_x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=source_padding_mask
        )

        x = x + self.dropout2(
            cross_attention_output
        )

        # 3. FFN
        normalized_x = self.norm3(x)

        ffn_output = self.ffn(normalized_x)

        x = x + self.dropout3(ffn_output)

        return x


# ============================================================
# 6. Encoder
# ============================================================

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len
        )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        source_tokens:
            [B, T_src]
        """

        # [B, T_src]
        #   ↓ Embedding
        # [B, T_src, D]
        x = self.token_embedding(source_tokens)

        # Transformer 原论文中会乘 sqrt(d_model)
        x = x * math.sqrt(self.d_model)

        x = self.position_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                source_padding_mask=source_padding_mask
            )

        return self.final_norm(x)


# ============================================================
# 7. Decoder
# ============================================================

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len
        )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 把隐藏向量转换成词表 logits
        self.output_projection = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        source_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        target_tokens:
            [B, T_tgt]

        返回 logits：
            [B, T_tgt, target_vocab_size]
        """

        x = self.token_embedding(target_tokens)

        x = x * math.sqrt(self.d_model)

        x = self.position_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                causal_mask=causal_mask,
                target_padding_mask=target_padding_mask,
                source_padding_mask=source_padding_mask
            )

        x = self.final_norm(x)

        logits = self.output_projection(x)

        return logits


# ============================================================
# 8. 完整 Transformer
# ============================================================

class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        pad_token_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pad_token_id = pad_token_id

        self.encoder = Encoder(
            vocab_size=source_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=target_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

    @staticmethod
    def create_causal_mask(
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        创建上三角因果掩码。

        True 表示不能关注。
        """

        return torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device
            ),
            diagonal=1
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        source_tokens:
            [B, T_src]

        target_tokens:
            [B, T_tgt]

        返回：
            [B, T_tgt, target_vocab_size]
        """

        # PAD 的位置为 True
        source_padding_mask = (
            source_tokens == self.pad_token_id
        )

        target_padding_mask = (
            target_tokens == self.pad_token_id
        )

        causal_mask = self.create_causal_mask(
            seq_len=target_tokens.size(1),
            device=target_tokens.device
        )

        encoder_output = self.encoder(
            source_tokens=source_tokens,
            source_padding_mask=source_padding_mask
        )

        logits = self.decoder(
            target_tokens=target_tokens,
            encoder_output=encoder_output,
            causal_mask=causal_mask,
            target_padding_mask=target_padding_mask,
            source_padding_mask=source_padding_mask
        )

        return logits

def main() -> None:
    """用带 PAD 的小批量输入验证完整前向传播。"""
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pad_token_id = 0

    model = Transformer(
        source_vocab_size=1000,
        target_vocab_size=1200,
        pad_token_id=pad_token_id,
        d_model=64,
        n_heads=8,
        d_ff=256,
        n_layers=2,
        max_seq_len=128,
        dropout=0.1
    ).to(device)

    source_tokens = torch.tensor([
        [5, 18, 29, 46, 2, 0, 0],
        [7, 31, 55, 12, 9, 4, 2]
    ], device=device)

    target_tokens = torch.tensor([
        [1, 20, 38, 2, 0, 0],
        [1, 17, 63, 25, 8, 2]
    ], device=device)

    logits = model(source_tokens, target_tokens)

    print("运行设备：", device)
    print("source_tokens：", source_tokens.shape)
    print("target_tokens：", target_tokens.shape)
    print("logits：", logits.shape)

    expected_shape = (
        target_tokens.size(0),
        target_tokens.size(1),
        1200
    )
    assert logits.shape == expected_shape
    assert torch.isfinite(logits).all()


if __name__ == "__main__":
    main()
