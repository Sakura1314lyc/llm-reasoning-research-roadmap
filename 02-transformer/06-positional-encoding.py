"""使用 NumPy 实现 Transformer 的正弦位置编码。

偶数维使用 sin，奇数维使用 cos。位置编码形状为 [T, D]，
与 token embedding 相加后即可为模型提供位置信息。
"""

import numpy as np


def positional_encoding(
    sequence_length: int,
    d_model: int,
    base: float = 10_000.0
) -> np.ndarray:
    """返回形状为 [sequence_length, d_model] 的位置编码矩阵。"""
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须大于 0")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model 必须是大于 0 的偶数")
    if base <= 0:
        raise ValueError("base 必须大于 0")

    # position: [T, 1]；dimension: [D/2]
    position = np.arange(sequence_length, dtype=np.float64)[:, None]
    dimension = np.arange(0, d_model, 2, dtype=np.float64)

    # 对应论文中的 10000^(2i/d_model)。
    denominator = np.power(base, dimension / d_model)
    angles = position / denominator

    encoding = np.zeros((sequence_length, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)

    return encoding


def main() -> None:
    encoding = positional_encoding(
        sequence_length=4,
        d_model=4,
        base=100.0
    )

    np.set_printoptions(precision=4, suppress=True)
    print("位置编码形状：", encoding.shape)
    print(encoding)

    assert encoding.shape == (4, 4)
    # 位置 0 的 sin 维为 0，cos 维为 1。
    assert np.allclose(encoding[0], [0.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    main()
