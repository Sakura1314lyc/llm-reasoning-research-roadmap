"""蒸馏示例共用的 Token KL、Mask 和混合损失。"""

import torch
import torch.nn.functional as F


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("教师与学生 logits 必须逐 Token 对齐")
    if mask.shape != student_logits.shape[:-1]:
        raise ValueError("mask 应为 [B,T]")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * temperature**2
    return (per_token * mask).sum() / mask.sum().clamp_min(1)


def mixed_loss(sft_loss: torch.Tensor, online_kd_loss: torch.Tensor, online_ratio: float) -> torch.Tensor:
    if not 0 <= online_ratio <= 1:
        raise ValueError("online_ratio 必须位于 [0,1]")
    return (1 - online_ratio) * sft_loss + online_ratio * online_kd_loss
