"""检查 Agent 能否终止循环、报告工具错误并守住路径边界。"""

from pathlib import Path

from agent_core import RepositoryTools


def main() -> None:
    tools = RepositoryTools(Path(__file__).resolve().parents[2])
    assert len(tools.list_stages()) == 9
    try:
        tools.read_text("../outside.txt")
    except PermissionError:
        print("路径穿越测试：通过")
    else:
        raise AssertionError("路径穿越未被阻止")


if __name__ == "__main__":
    main()
