"""对比 BatchNorm 与 LayerNorm 的归一化维度。

对于形状为 [B, D] 的输入：
- BatchNorm 沿 batch 维度统计每个特征；
- LayerNorm 在每个样本内部统计最后一个特征维度。

Transformer 通常使用 LayerNorm，因为它不依赖 batch 中的其他样本。
"""

import torch
from torch import nn


def main() -> None:
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0]
    ])

    # 关闭可学习缩放/偏移和运行统计，方便直接观察归一化结果。
    batch_norm = nn.BatchNorm1d(
        num_features=3,
        affine=False,
        track_running_stats=False
    )
    layer_norm = nn.LayerNorm(
        normalized_shape=3,
        elementwise_affine=False
    )

    batch_normalized = batch_norm(x)
    layer_normalized = layer_norm(x)

    print("输入：\n", x)
    print("\nBatchNorm 输出：\n", batch_normalized)
    print("BatchNorm 每列均值：", batch_normalized.mean(dim=0))
    print("BatchNorm 每列方差：", batch_normalized.var(dim=0, unbiased=False))

    print("\nLayerNorm 输出：\n", layer_normalized)
    print("LayerNorm 每行均值：", layer_normalized.mean(dim=-1))
    print("LayerNorm 每行方差：", layer_normalized.var(dim=-1, unbiased=False))

    expected_zeros = torch.zeros(3)
    assert torch.allclose(
        batch_normalized.mean(dim=0),
        expected_zeros,
        atol=1e-6
    )
    assert torch.allclose(
        layer_normalized.mean(dim=-1),
        torch.zeros(x.size(0)),
        atol=1e-6
    )


if __name__ == "__main__":
    main()
