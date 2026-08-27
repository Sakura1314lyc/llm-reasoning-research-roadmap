"""训练前检查 SFT 样本的角色、空值和重复项。"""

from collections import Counter


def validate(records: list[dict]) -> dict:
    valid, errors, seen = [], Counter(), set()
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            errors["invalid_messages"] += 1
            continue
        if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
            errors["invalid_roles"] += 1
            continue
        if not all(str(message.get("content", "")).strip() for message in messages):
            errors["empty_content"] += 1
            continue
        key = tuple((message["role"], message["content"].strip()) for message in messages)
        if key in seen:
            errors["duplicate"] += 1
            continue
        seen.add(key)
        valid.append({"id": record.get("id", str(index)), "messages": messages})
    return {"valid": valid, "errors": dict(errors)}


def main() -> None:
    result = validate([
        {"id": "1", "messages": [{"role": "user", "content": "2+3?"}, {"role": "assistant", "content": "#### 5"}]},
        {"id": "2", "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]},
    ])
    print(result)
    assert len(result["valid"]) == 1


if __name__ == "__main__":
    main()
