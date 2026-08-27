"""沿学生序列计算教师与学生的 Token KL。"""

import torch

from distillation_utils import token_kl


def main() -> None:
    teacher = torch.randn(1, 6, 32)
    student = torch.randn(1, 6, 32, requires_grad=True)
    # 前两个位置是 Prompt，后三个有效 completion，最后一个是 Padding。
    completion_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    loss = token_kl(student, teacher, completion_mask)
    loss.backward()
    print("On-policy KD loss：", loss.item())
    assert student.grad[:, :2].abs().sum() == 0


if __name__ == "__main__":
    main()
