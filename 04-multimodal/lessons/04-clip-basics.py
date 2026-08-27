"""用小张量演示 CLIP 的图文对比学习目标。"""

import torch
import torch.nn.functional as F


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    images = F.normalize(image_features, dim=-1)
    texts = F.normalize(text_features, dim=-1)
    logits = images @ texts.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def main() -> None:
    torch.manual_seed(42)
    images = torch.randn(4, 32, requires_grad=True)
    texts = images.detach() + 0.05 * torch.randn(4, 32)
    loss = clip_loss(images, texts)
    loss.backward()
    similarities = F.normalize(images.detach(), dim=-1) @ F.normalize(texts, dim=-1).T
    print("相似度矩阵：\n", similarities)
    print("对称对比损失：", loss.item())
    assert similarities.argmax(dim=1).tolist() == [0, 1, 2, 3]
    assert images.grad is not None


if __name__ == "__main__":
    main()
