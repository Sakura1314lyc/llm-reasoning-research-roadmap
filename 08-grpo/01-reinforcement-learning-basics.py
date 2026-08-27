"""用几个张量手算一次带 Mask 的 REINFORCE Loss。"""

import torch


def policy_gradient_loss(log_probs: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = -log_probs * advantages[:, None]
    return (per_token * mask).sum() / mask.sum().clamp_min(1)


def main() -> None:
    log_probs = torch.tensor([[-0.2, -0.5, -0.1], [-0.4, -0.3, -0.6]], requires_grad=True)
    advantages = torch.tensor([1.0, -0.5])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    loss = policy_gradient_loss(log_probs, advantages, mask)
    loss.backward()
    print("Policy loss：", loss.item())
    print("梯度：\n", log_probs.grad)
    assert log_probs.grad[0].sum() < 0  # 正优势提高动作概率
    assert log_probs.grad[1, :2].sum() > 0


if __name__ == "__main__":
    main()
