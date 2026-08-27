"""定义学生 On-policy Rollout 的逐 Token 记录结构。"""


def build_rollout(prompt: str, completion: str, token_ids: list[int], log_probs: list[float]) -> dict:
    if len(token_ids) != len(log_probs):
        raise ValueError("每个生成 Token 必须对应一个行为策略 logprob")
    return {"prompt": prompt, "completion": completion, "completion_ids": token_ids, "behavior_log_probs": log_probs, "length": len(token_ids)}


def main() -> None:
    rollout = build_rollout("2+3?", "#### 5", [10, 11, 12], [-0.2, -0.1, -0.05])
    print(rollout)
    assert rollout["length"] == 3


if __name__ == "__main__":
    main()
