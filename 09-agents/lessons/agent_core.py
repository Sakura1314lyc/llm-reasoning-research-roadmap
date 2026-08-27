"""只读学习仓库 Agent 的核心状态、工具与有限步循环。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentState:
    goal: str
    observations: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    answer: str | None = None
    done: bool = False


class RepositoryTools:
    """所有路径都限制在 root 内；不提供写入、删除或 shell。"""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError("路径越过仓库根目录")
        return candidate

    def list_stages(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir() and path.name[:2].isdigit())

    def read_text(self, relative_path: str, maximum_characters: int = 8000) -> str:
        path = self.resolve(relative_path)
        if path.suffix.lower() not in {".md", ".py", ".txt", ".json", ".yaml", ".yml"}:
            raise ValueError("只允许读取文本课程文件")
        return path.read_text(encoding="utf-8")[:maximum_characters]

    def search(self, keyword: str, maximum_results: int = 20) -> list[str]:
        if not keyword.strip():
            raise ValueError("keyword 不能为空")
        results = []
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".py"}:
                try:
                    if keyword.lower() in path.read_text(encoding="utf-8").lower():
                        results.append(str(path.relative_to(self.root)))
                except UnicodeDecodeError:
                    continue
                if len(results) >= maximum_results:
                    break
        return results


def run_agent(goal: str, decide, tools: RepositoryTools, maximum_steps: int = 6) -> AgentState:
    state = AgentState(goal=goal)
    registry = {"list_stages": tools.list_stages, "read_text": tools.read_text, "search": tools.search}
    for _ in range(maximum_steps):
        decision = decide(state)
        if decision.get("type") == "finish":
            state.answer = str(decision.get("answer", ""))
            state.done = True
            return state
        if decision.get("type") != "tool" or decision.get("name") not in registry:
            raise ValueError("决策必须是已注册工具调用或 finish")
        name, arguments = decision["name"], decision.get("arguments", {})
        result = registry[name](**arguments)
        state.actions.append(decision)
        state.observations.append(str(result))
    state.answer = "达到最大步数，任务未完成。"
    return state
