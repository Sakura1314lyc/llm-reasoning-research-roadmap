"""分清 QLoRA 的 4-bit 存储、计算 dtype 和 Adapter。"""

from dataclasses import asdict, dataclass


@dataclass
class QLoRAConfig:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    trainable_components: str = "LoRA adapters"


def main() -> None:
    config = QLoRAConfig()
    print(asdict(config))
    print("4-bit 是基础权重存储；前向/反向仍使用 compute_dtype；保存的是 Adapter。")
    assert config.load_in_4bit and config.trainable_components == "LoRA adapters"


if __name__ == "__main__":
    main()
