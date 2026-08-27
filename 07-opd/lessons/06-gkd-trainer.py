"""记录一组 TRL GKD/On-policy 蒸馏实验配置。"""

from dataclasses import asdict, dataclass


@dataclass
class GKDExperiment:
    teacher_model: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    student_model: str = "Qwen/Qwen2.5-Math-1.5B"
    prompts: int = 500
    max_new_tokens: int = 256
    temperature: float = 0.9
    beta: float = 0.5
    online_ratio: float = 1.0
    seed: int = 42


def main() -> None:
    config = GKDExperiment()
    print(asdict(config))
    print("固定 TRL/Transformers/PEFT 版本后，将这些字段映射到当前版本 GKDTrainer。")
    assert config.teacher_model.split("/")[0] == config.student_model.split("/")[0]


if __name__ == "__main__":
    main()
