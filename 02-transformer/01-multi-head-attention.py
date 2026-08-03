"""Transformer 课程 01：缩放点积注意力与多头注意力。

实现 Q/K/V 投影、多头拆分与合并、因果掩码和注意力 Dropout。
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    # Transformer 隐藏层维度
    dim: int = 512

    # 注意力头数
    n_heads: int = 8

    # Dropout 概率
    dropout: float = 0.1

    # 最大序列长度
    max_seq_len: int = 1024


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力。

    query: [B, H, Tq, D]
    key:   [B, H, Tk, D]
    value: [B, H, Tk, D]

    mask:
        布尔类型张量，可以广播到 [B, H, Tq, Tk]
        True  表示该位置需要被遮挡
        False 表示该位置可以参与注意力计算
    """

    # 每个注意力头的维度 D
    d_k = query.size(-1)

    # query: [B, H, Tq, D]
    # key.transpose: [B, H, D, Tk]
    #
    # scores: [B, H, Tq, Tk]
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) / math.sqrt(d_k)

    # 加入注意力掩码
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(
                f"mask 必须是 torch.bool 类型，当前类型为 {mask.dtype}"
            )

        # mask 中 True 的位置会被填充为负无穷
        # 经过 Softmax 后，这些位置的概率会变成 0
        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

    # Softmax 对最后一维 Tk 归一化
    #
    # [B, H, Tq, Tk]
    attn_weights = F.softmax(
        scores.float(),
        dim=-1,
    ).type_as(query)

    # 注意力权重 Dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # attn_weights: [B, H, Tq, Tk]
    # value:        [B, H, Tk, D]
    #
    # output:       [B, H, Tq, D]
    output = torch.matmul(
        attn_weights,
        value,
    )

    return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        args: ModelArgs,
        is_causal: bool = False,
    ):
        super().__init__()

        if args.dim % args.n_heads != 0:
            raise ValueError(
                f"dim 必须能被 n_heads 整除，"
                f"当前 dim={args.dim}，n_heads={args.n_heads}"
            )

        # 这里原来的代码漏写了 self.dim
        self.dim = args.dim

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len
        self.is_causal = is_causal

        # 输入：[B, T, dim]
        # 输出：[B, T, dim]
        self.wq = nn.Linear(
            args.dim,
            args.dim,
            bias=False,
        )

        self.wk = nn.Linear(
            args.dim,
            args.dim,
            bias=False,
        )

        self.wv = nn.Linear(
            args.dim,
            args.dim,
            bias=False,
        )

        # 合并多头后的输出投影
        self.wo = nn.Linear(
            args.dim,
            args.dim,
            bias=False,
        )

        # 注意力权重上的 Dropout
        self.attn_dropout = nn.Dropout(
            args.dropout
        )

        # 最终输出上的 Dropout
        self.resid_dropout = nn.Dropout(
            args.dropout
        )

        # 无论是否启用因果注意力，
        # 都注册一个真正的 Tensor，避免 Tensor | None 类型问题
        if self.is_causal:
            # 创建布尔类型上三角矩阵
            #
            # True  表示未来位置，需要被遮挡
            # False 表示当前位置或过去位置，可以被看到
            causal_mask = torch.triu(
                torch.ones(
                    (
                        1,
                        1,
                        args.max_seq_len,
                        args.max_seq_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        else:
            # 非因果注意力不会使用它，
            # 但仍然注册一个空 Tensor
            causal_mask = torch.empty(
                0,
                dtype=torch.bool,
            )

        # 不直接写 self.mask
        #
        # persistent=False 表示不保存进 state_dict，
        # 但仍会随 model.to(device) 移动到对应设备
        self.register_buffer(
            "_causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: [B, Tq, C]
        k: [B, Tk, C]
        v: [B, Tk, C]

        B  = batch_size
        Tq = query 序列长度
        Tk = key/value 序列长度
        C  = 模型维度 dim
        """

        # --------------------------------------------------
        # 1. 检查输入维度
        # --------------------------------------------------

        if q.ndim != 3:
            raise ValueError(
                f"q 必须是三维张量 [B, Tq, C]，"
                f"当前形状为 {tuple(q.shape)}"
            )

        if k.ndim != 3:
            raise ValueError(
                f"k 必须是三维张量 [B, Tk, C]，"
                f"当前形状为 {tuple(k.shape)}"
            )

        if v.ndim != 3:
            raise ValueError(
                f"v 必须是三维张量 [B, Tk, C]，"
                f"当前形状为 {tuple(v.shape)}"
            )

        batch_size, query_len, q_dim = q.shape
        k_batch_size, key_len, k_dim = k.shape
        v_batch_size, value_len, v_dim = v.shape

        # 三者批次大小必须一致
        if not (
            batch_size
            == k_batch_size
            == v_batch_size
        ):
            raise ValueError(
                "q、k、v 的 batch_size 必须相同，"
                f"当前分别为 {batch_size}、"
                f"{k_batch_size}、{v_batch_size}"
            )

        # K 和 V 必须一一对应
        if key_len != value_len:
            raise ValueError(
                "key 和 value 的序列长度必须相同，"
                f"当前 key_len={key_len}，"
                f"value_len={value_len}"
            )

        # 输入隐藏维度必须等于模型维度
        if q_dim != self.dim:
            raise ValueError(
                f"q 的最后一维必须是 {self.dim}，"
                f"当前为 {q_dim}"
            )

        if k_dim != self.dim:
            raise ValueError(
                f"k 的最后一维必须是 {self.dim}，"
                f"当前为 {k_dim}"
            )

        if v_dim != self.dim:
            raise ValueError(
                f"v 的最后一维必须是 {self.dim}，"
                f"当前为 {v_dim}"
            )

        # --------------------------------------------------
        # 2. 计算 Q、K、V
        # --------------------------------------------------

        # [B, Tq, C]
        xq = self.wq(q)

        # [B, Tk, C]
        xk = self.wk(k)

        # [B, Tk, C]
        xv = self.wv(v)

        # --------------------------------------------------
        # 3. 拆分多头
        # --------------------------------------------------

        # [B, Tq, C]
        # -> [B, Tq, H, D]
        # -> [B, H, Tq, D]
        xq = xq.view(
            batch_size,
            query_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # [B, Tk, C]
        # -> [B, Tk, H, D]
        # -> [B, H, Tk, D]
        xk = xk.view(
            batch_size,
            key_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # [B, Tk, C]
        # -> [B, Tk, H, D]
        # -> [B, H, Tk, D]
        xv = xv.view(
            batch_size,
            value_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # --------------------------------------------------
        # 4. 准备因果掩码
        # --------------------------------------------------

        attn_mask: torch.Tensor | None = None

        if self.is_causal:
            # 当前实现针对普通 GPT 自注意力
            #
            # q、k、v 来自同一个序列时，
            # query_len 应该等于 key_len
            if query_len != key_len:
                raise ValueError(
                    "当前因果注意力实现要求 "
                    "query_len == key_len，"
                    f"当前 query_len={query_len}，"
                    f"key_len={key_len}"
                )

            if query_len > self.max_seq_len:
                raise ValueError(
                    f"当前序列长度 {query_len} 超过了 "
                    f"max_seq_len={self.max_seq_len}"
                )

            # 关键修改：
            # 不访问 self.mask 或 self._causal_mask，
            # 而是通过 get_buffer 获取
            causal_mask: torch.Tensor = self.get_buffer(
                "_causal_mask"
            )

            # [1, 1, max_seq_len, max_seq_len]
            # -> [1, 1, query_len, key_len]
            attn_mask = causal_mask[
                :,
                :,
                :query_len,
                :key_len,
            ]

        # --------------------------------------------------
        # 5. 注意力计算
        # --------------------------------------------------

        # output:
        # [B, H, Tq, D]
        output, _ = attention(
            query=xq,
            key=xk,
            value=xv,
            mask=attn_mask,
            dropout=self.attn_dropout,
        )

        # --------------------------------------------------
        # 6. 合并多个注意力头
        # --------------------------------------------------

        # [B, H, Tq, D]
        # -> [B, Tq, H, D]
        output = output.transpose(1, 2)

        # transpose 后内存可能不连续
        output = output.contiguous()

        # [B, Tq, H, D]
        # -> [B, Tq, H * D]
        # -> [B, Tq, C]
        output = output.view(
            batch_size,
            query_len,
            self.dim,
        )

        # --------------------------------------------------
        # 7. 输出投影
        # --------------------------------------------------

        output = self.wo(output)

        output = self.resid_dropout(output)

        return output


def demo() -> None:
    """运行一个最小自注意力示例并检查输入输出形状。"""

    torch.manual_seed(42)

    args = ModelArgs(
        dim=32,
        n_heads=4,
        dropout=0.0,
        max_seq_len=16,
    )

    model = MultiHeadAttention(
        args,
        is_causal=True,
    ).eval()

    hidden_states = torch.randn(2, 8, args.dim)

    with torch.no_grad():
        output = model(
            hidden_states,
            hidden_states,
            hidden_states,
        )

    print("input shape:", hidden_states.shape)
    print("output shape:", output.shape)

    assert output.shape == hidden_states.shape


if __name__ == "__main__":
    demo()
