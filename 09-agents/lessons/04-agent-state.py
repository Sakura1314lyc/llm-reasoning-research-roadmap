"""把会话状态、长期记忆和外部资源分开保存。"""

from dataclasses import dataclass, field


@dataclass
class SessionState:
    current_goal: str
    messages: list[str] = field(default_factory=list)
    temporary_facts: dict[str, str] = field(default_factory=dict)


def main() -> None:
    state = SessionState("完成 Transformer 复习")
    state.messages.append("用户要求按 01–11 顺序复习")
    state.temporary_facts["current_stage"] = "02-transformer"
    print(state)
    print("长期记忆应经过筛选后写入独立存储；README 是外部资源，不是会话状态。")


if __name__ == "__main__":
    main()
