"""不用图学习框架，直接拆开 GCN、GAT 和 GIN 的消息传递。"""

import torch
from torch import nn


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """计算带自环的 D^-1/2 A D^-1/2。"""
    adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device)
    degree = adjacency.sum(dim=1)
    inv_sqrt = degree.clamp_min(1).pow(-0.5)
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return normalized_adjacency(adjacency) @ self.linear(x)


class GINLayer(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(features, features), nn.ReLU(), nn.Linear(features, features))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_sum = adjacency @ x
        return self.mlp((1 + self.epsilon) * x + neighbor_sum)


def gat_attention(x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """教学版点积 GAT：只对邻居做 Softmax，不含可学习投影。"""
    scores = x @ x.T / x.size(-1) ** 0.5
    allowed = (adjacency + torch.eye(adjacency.size(0))).bool()
    weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    return weights @ x


def main() -> None:
    x = torch.randn(4, 6)
    adjacency = torch.tensor([
        [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]
    ], dtype=torch.float32)

    gcn_output = GCNLayer(6, 8)(x, adjacency)
    gat_output = gat_attention(x, adjacency)
    gin_output = GINLayer(6)(x, adjacency)

    print("节点特征：", x.shape)
    print("GCN / GAT / GIN：", gcn_output.shape, gat_output.shape, gin_output.shape)
    assert gcn_output.shape == (4, 8)
    assert gat_output.shape == gin_output.shape == x.shape


if __name__ == "__main__":
    main()
