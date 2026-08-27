"""在固定教师序列上计算 Token 级蒸馏 Loss。"""

import torch

from distillation_utils import token_kl


def main() -> None:
    torch.manual_seed(42)
    teacher = torch.randn(2, 4, 16)
    student = (teacher + 0.2 * torch.randn_like(teacher)).requires_grad_()
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    loss = token_kl(student, teacher, mask, temperature=2.0)
    loss.backward()
    print("KD loss：", loss.item())
    assert loss >= 0 and student.grad is not None


if __name__ == "__main__":
    main()
