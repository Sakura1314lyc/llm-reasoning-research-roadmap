"""一张不会自动启动训练的 TRL SFTTrainer 配置卡。"""

from dataclasses import asdict, dataclass


@dataclass
class SFTExperimentConfig:
    model: str = "Qwen/Qwen2.5-Math-1.5B"
    max_length: int = 1024
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def main() -> None:
    config = SFTExperimentConfig()
    print(asdict(config))
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print("单卡有效 Batch：", effective_batch)
    assert effective_batch == 16


if __name__ == "__main__":
    main()
