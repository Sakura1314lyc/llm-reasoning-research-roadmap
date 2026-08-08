"""Transformer 课程 09：从零实现 Decoder-only Causal Language Model。

本章把前面的知识点串成一个最小 GPT：

1. Token Embedding 把 token ID 映射为隐藏向量；
2. RoPE 旋转每层注意力中的 Query 与 Key；
3. Causal Mask 阻止当前位置看到未来 token；
4. Pre-Norm Transformer Block 组合注意力、SwiGLU 和残差连接；
5. LM Head 为每个位置输出下一个 token 的词表 logits；
6. 右移一位的 targets 用于计算 Causal LM Cross-Entropy Loss。

这还是随机初始化的教学模型，生成结果没有语义；第 10 章再加入训练。
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Mini GPT 的结构超参数。"""

    vocab_size: int = 128
    max_sequence_length: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE 要求每个注意力头的维度为偶数")
        if self.vocab_size <= 0 or self.max_sequence_length <= 0:
            raise ValueError("词表大小和最大序列长度必须大于 0")


class RMSNorm(nn.Module):
    """只根据均方根缩放特征，不减去均值。"""

    def __init__(
        self,
        d_model: int,
        epsilon: float = 1e-6
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 计算方差，提高半精度训练时的数值稳定性。
        rms_inverse = torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True)
            + self.epsilon
        )
        normalized = x.float() * rms_inverse

        return (normalized * self.weight.float()).to(dtype=x.dtype)


def build_rope_cache(
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """构建形状为 [1, 1, T, D/2] 的 RoPE cos/sin。"""
    positions = torch.arange(
        sequence_length,
        dtype=torch.float32,
        device=device
    )
    dimensions = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
        device=device
    )
    inverse_frequency = 1.0 / (base ** (dimensions / head_dim))
    angles = torch.outer(positions, inverse_frequency)

    return (
        angles.cos()[None, None, :, :],
        angles.sin()[None, None, :, :]
    )


def apply_rope(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor
) -> torch.Tensor:
    """旋转形状为 [B, H, T, D] 的 Query 或 Key。"""
    paired = x.float().reshape(*x.shape[:-1], -1, 2)
    even = paired[..., 0]
    odd = paired[..., 1]

    rotated = torch.stack(
        (
            even * cosine - odd * sine,
            even * sine + odd * cosine
        ),
        dim=-1
    ).flatten(start_dim=-2)

    return rotated.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    """带 RoPE 和因果掩码的多头自注意力。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        # 一次线性投影同时产生 Q、K、V，随后沿最后一维拆分。
        self.qkv_projection = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False
        )
        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # [1, 1, max_T, max_T]；True 表示未来位置，需要遮挡。
        causal_mask = torch.triu(
            torch.ones(
                config.max_sequence_length,
                config.max_sequence_length,
                dtype=torch.bool
            ),
            diagonal=1
        )[None, None, :, :]

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        cosine, sine = build_rope_cache(
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            device=x.device
        )
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)

        # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:, :, :sequence_length, :sequence_length],
            float("-inf")
        )

        attention_weights = torch.softmax(
            attention_scores.float(),
            dim=-1
        ).to(dtype=query.dtype)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.d_model
        )

        return self.residual_dropout(self.output_projection(context))


class SwiGLU(nn.Module):
    """现代 LLM 常用的门控前馈网络。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )
        self.up_projection = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )
        self.down_projection = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_projection(x)) * self.up_projection(x)
        return self.dropout(self.down_projection(gated))


class TransformerBlock(nn.Module):
    """RMSNorm -> Attention/FFN -> Residual 的 Pre-Norm Block。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    """输入 token ID，输出每个位置对整个词表的预测 logits。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        self.apply(self._initialize_weights)

        # 输入 embedding 与输出分类头共享参数，减少参数量。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B, T] 二维张量")
        if token_ids.size(1) > self.config.max_sequence_length:
            raise ValueError(
                f"序列长度不能超过 {self.config.max_sequence_length}"
            )

        x = self.embedding_dropout(self.token_embedding(token_ids))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets 与 token_ids 的形状必须相同")

            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """根据最后一个位置的分布，逐 token 自回归采样。"""
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.max_sequence_length:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                actual_top_k = min(top_k, next_token_logits.size(-1))
                threshold = torch.topk(
                    next_token_logits,
                    k=actual_top_k,
                    dim=-1
                ).values[:, -1, None]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf")
                )

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        if was_training:
            self.train()

        return token_ids


def verify_causality(model: MiniGPT) -> None:
    """修改未来 token，确认较早位置的 logits 不受影响。"""
    sequence_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sequence_b = torch.tensor([[1, 2, 3, 9, 8, 7]])

    model.eval()
    with torch.no_grad():
        logits_a, _ = model(sequence_a)
        logits_b, _ = model(sequence_b)

    torch.testing.assert_close(
        logits_a[:, :3],
        logits_b[:, :3],
        atol=1e-6,
        rtol=1e-6
    )


def main() -> None:
    torch.manual_seed(42)

    config = GPTConfig()
    model = MiniGPT(config)

    # Causal LM 使用前 T-1 个 token 预测后 T-1 个 token。
    batch = torch.randint(0, config.vocab_size, (2, 9))
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    verify_causality(model)

    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=20
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("输入形状：", inputs.shape)
    print("logits 形状：", logits.shape)
    print("Causal LM loss：", loss.item())
    print("参数量：", parameter_count)
    print("因果性检查：通过")
    print("生成后的 token：", generated.tolist())

    assert logits.shape == (
        inputs.size(0),
        inputs.size(1),
        config.vocab_size
    )
    assert generated.shape == (1, prompt.size(1) + 5)
    assert model.token_embedding.weight.grad is not None


if __name__ == "__main__":
    main()
