"""用模型自己的 Chat Template 格式化 messages。"""

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [{"role": "user", "content": "2 + 3 等于多少？"}, {"role": "assistant", "content": "2 + 3 = 5。 #### 5"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    print(text)
    print("Token 数：", len(encoded["input_ids"]))
    assert len(encoded["input_ids"]) == len(encoded["attention_mask"])


if __name__ == "__main__":
    main()
