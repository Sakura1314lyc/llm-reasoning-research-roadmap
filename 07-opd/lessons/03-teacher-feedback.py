"""在学生真正生成的 Token 上读取教师 logprob。"""

import torch
import torch.nn.functional as F


def selected_token_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("logits [B,T,V] 与 token_ids [B,T] 必须对齐")
    return F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    logits = torch.randn(2, 5, 20)
    ids = torch.randint(0, 20, (2, 5))
    feedback = selected_token_log_probs(logits, ids)
    print("教师选中 Token logprob：", feedback.shape)
    assert feedback.shape == ids.shape


if __name__ == "__main__":
    main()
