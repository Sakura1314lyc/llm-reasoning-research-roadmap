"""从轨迹中统计成功率、平均步数和失败原因。"""

from collections import Counter


def analyze(trajectories: list[dict]) -> dict:
    failures = Counter(item.get("failure_type", "success") for item in trajectories)
    return {"success_rate": sum(item["success"] for item in trajectories) / len(trajectories), "average_steps": sum(item["steps"] for item in trajectories) / len(trajectories), "failure_counts": dict(failures)}


def main() -> None:
    metrics = analyze([
        {"success": True, "steps": 2},
        {"success": False, "steps": 6, "failure_type": "max_steps"},
        {"success": False, "steps": 3, "failure_type": "wrong_tool"},
    ])
    print(metrics)
    assert metrics["average_steps"] == 11 / 3


if __name__ == "__main__":
    main()
