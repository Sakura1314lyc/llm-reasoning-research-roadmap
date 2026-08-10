"""Hugging Face 课程 01：查看 Hub 模型配置与本地缓存。

本课只下载体积很小的 ``config.json``，不加载模型权重。配置对象可以帮助
我们在推理前确认模型家族、隐藏维度、层数、注意力头数和上下文长度。
"""

from huggingface_hub import hf_hub_download
from transformers import AutoConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    # from_pretrained() 会从 Hub 下载资源并缓存在本地；再次运行会复用缓存。
    config = AutoConfig.from_pretrained(MODEL_NAME)

    config_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="config.json"
    )

    print("模型仓库：", MODEL_NAME)
    print("配置类型：", type(config).__name__)
    print("模型家族：", config.model_type)
    print("模型架构：", config.architectures)
    print("词表大小：", config.vocab_size)
    print("隐藏维度：", config.hidden_size)
    print("Transformer 层数：", config.num_hidden_layers)
    print("Attention Head 数：", config.num_attention_heads)
    print("KV Head 数：", getattr(config, "num_key_value_heads", "未提供"))
    print("FFN 中间维度：", config.intermediate_size)
    print("最大位置长度：", config.max_position_embeddings)
    print("config.json 本地路径：", config_path)

    assert config.vocab_size > 0
    assert config.hidden_size > 0
    assert config.num_hidden_layers > 0


if __name__ == "__main__":
    main()
