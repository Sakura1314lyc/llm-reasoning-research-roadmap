"""跑一个有最大步数的 observe → decide → act 循环。"""

from pathlib import Path

from agent_core import RepositoryTools, run_agent


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    calls = iter([
        {"type": "tool", "name": "list_stages", "arguments": {}},
        {"type": "finish", "answer": "已经列出全部学习阶段。"},
    ])
    state = run_agent("仓库有哪些学习阶段？", lambda _: next(calls), RepositoryTools(root))
    print(state)
    assert state.done and len(state.actions) == 1


if __name__ == "__main__":
    main()
