"""Transformer 课程 08：旋转位置编码（RoPE）。

RoPE 不把位置向量直接加到 token embedding 上，而是按照 token 位置
旋转注意力中的 Query 和 Key。每两个相邻特征组成一个二维平面，位置越靠后，
旋转角度越大。这样，Q 与 K 的点积能够自然携带相对位置信息。

形状约定：
    x: [B, H, T, D]
    B 为批量大小，H 为注意力头数，T 为序列长度，D 为每个头的维度。
"""

import torch
from torch import nn


def apply_rotary_embedding(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor
) -> torch.Tensor:
    """将预先计算的 cos/sin 旋转应用到最后一个维度。"""
    if x.ndim != 4:
        raise ValueError(
            f"x 必须是 [B, H, T, D] 四维张量，当前为 {tuple(x.shape)}"
        )
    if x.size(-1) % 2 != 0:
        raise ValueError("每个注意力头的维度必须是偶数")

    # 把最后一维拆成 D/2 对二维坐标：[..., D/2, 2]。
    paired = x.float().reshape(*x.shape[:-1], -1, 2)
    even = paired[..., 0]
    odd = paired[..., 1]

    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine

    rotated = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1
    ).flatten(start_dim=-2)

    return rotated.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """根据位置生成旋转角度，并同时旋转 Query 与 Key。"""

    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0
    ) -> None:
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim 必须是大于 0 的偶数")
        if base <= 0:
            raise ValueError("base 必须大于 0")

        # inv_freq 的形状为 [D/2]。
        # 第 i 对维度的角频率为 base^(-2i/D)。
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        self.head_dim = head_dim
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

    def build_cache(
        self,
        sequence_length: int,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回可广播到 [B, H, T, D/2] 的 cos 与 sin。"""
        if sequence_length <= 0:
            raise ValueError("sequence_length 必须大于 0")

        positions = torch.arange(
            sequence_length,
            dtype=self.inv_freq.dtype,
            device=device
        )

        # [T] × [D/2] -> [T, D/2]
        angles = torch.outer(positions, self.inv_freq)

        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]

        return cosine, sine

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != key.shape:
            raise ValueError("教学示例要求 query 与 key 形状相同")
        if query.size(-1) != self.head_dim:
            raise ValueError(
                f"输入最后一维应为 {self.head_dim}，"
                f"当前为 {query.size(-1)}"
            )

        cosine, sine = self.build_cache(
            sequence_length=query.size(-2),
            device=query.device
        )

        query_rotated = apply_rotary_embedding(query, cosine, sine)
        key_rotated = apply_rotary_embedding(key, cosine, sine)

        return query_rotated, key_rotated


def main() -> None:
    torch.manual_seed(42)

    query = torch.randn(2, 4, 6, 8, requires_grad=True)
    key = torch.randn(2, 4, 6, 8, requires_grad=True)

    rope = RotaryEmbedding(head_dim=8)
    query_rotated, key_rotated = rope(query, key)

    # 二维旋转不会改变每一对特征的长度，因此完整向量范数保持不变。
    torch.testing.assert_close(
        query_rotated.float().norm(dim=-1),
        query.float().norm(dim=-1),
        atol=1e-5,
        rtol=1e-5
    )

    # 位置 0 的角度为 0，所以第一个 token 不发生旋转。
    torch.testing.assert_close(query_rotated[:, :, 0], query[:, :, 0])

    loss = (query_rotated.square().mean() + key_rotated.square().mean())
    loss.backward()

    print("输入形状：", query.shape)
    print("RoPE 输出形状：", query_rotated.shape)
    print("位置 0 保持不变：True")
    print("旋转前后向量范数保持不变：True")

    assert query_rotated.shape == query.shape
    assert key_rotated.shape == key.shape
    assert query.grad is not None
    assert key.grad is not None


if __name__ == "__main__":
    main()
