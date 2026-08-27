"""拼出 Zero-shot/Few-shot Prompt 和 OpenAI-compatible API 请求。"""

import json
import os
from urllib.request import Request, urlopen


def zero_shot_prompt(question: str) -> str:
    return f"Solve the problem. End with #### <answer>.\n\nQuestion: {question}"


def few_shot_prompt(question: str) -> str:
    example = "Question: Tom has 5 books and buys 2. How many?\nAnswer: 5 + 2 = 7. #### 7"
    return f"{example}\n\nQuestion: {question}\nAnswer:"


def call_openai_compatible(prompt: str) -> str:
    """按需调用兼容 Chat Completions 的服务；密钥只从环境变量读取。"""
    base_url = os.environ["LLM_API_BASE"].rstrip("/")
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    request = Request(f"{base_url}/chat/completions", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main() -> None:
    question = "Janet has 12 apples and gives 3 to each of 2 friends. How many remain?"
    print("Zero-shot:\n", zero_shot_prompt(question))
    print("\nFew-shot:\n", few_shot_prompt(question))
    print("\n设置 LLM_API_BASE / LLM_API_KEY / LLM_MODEL 后可调用 call_openai_compatible。")


if __name__ == "__main__":
    main()
