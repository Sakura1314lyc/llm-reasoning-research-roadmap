"""一个只暴露仓库 README 和阶段列表的 MCP Server。

需要安装 ``mcp`` 后运行：python 09-agents/lessons/06-mcp-server.py
"""

from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as error:
    raise SystemExit("请先安装 MCP Python SDK：pip install mcp") from error


ROOT = Path(__file__).resolve().parents[2]
mcp = FastMCP("roadmap-reader")


@mcp.resource("roadmap://root-readme")
def root_readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


@mcp.tool()
def list_learning_stages() -> list[str]:
    """列出仓库内 01–09 学习阶段；无副作用。"""
    return sorted(path.name for path in ROOT.iterdir() if path.is_dir() and path.name[:2].isdigit())


if __name__ == "__main__":
    mcp.run()
