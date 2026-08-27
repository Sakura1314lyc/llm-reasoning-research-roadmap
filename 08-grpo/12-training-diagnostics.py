"""汇总 GRPO 训练中的 Reward、KL、长度和组内方差。"""

from statistics import mean


def diagnose(steps: list[dict]) -> dict:
    flat_rewards = [reward for step in steps for reward in step["rewards"]]
    zero_variance_groups = sum(len(set(step["rewards"])) == 1 for step in steps)
    return {
        "mean_reward": mean(flat_rewards),
        "mean_kl": mean(step["kl"] for step in steps),
        "mean_length": mean(length for step in steps for length in step["lengths"]),
        "zero_variance_group_rate": zero_variance_groups / len(steps),
    }


def main() -> None:
    metrics = diagnose([
        {"rewards": [1.1, 0.0, 1.1, 0.1], "kl": 0.02, "lengths": [80, 90, 75, 120]},
        {"rewards": [0.1, 0.1, 0.1, 0.1], "kl": 0.05, "lengths": [100, 100, 100, 100]},
    ])
    print(metrics)
    assert metrics["zero_variance_group_rate"] == 0.5


if __name__ == "__main__":
    main()
