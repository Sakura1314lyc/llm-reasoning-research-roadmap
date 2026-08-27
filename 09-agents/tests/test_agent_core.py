from pathlib import Path

import pytest

from agent_core import RepositoryTools, run_agent


ROOT = Path(__file__).resolve().parents[2]


def test_tools_are_read_only_and_block_path_traversal() -> None:
    tools = RepositoryTools(ROOT)
    assert len(tools.list_stages()) == 9
    with pytest.raises(PermissionError):
        tools.read_text("../secret.txt")


def test_agent_stops_at_maximum_steps() -> None:
    tools = RepositoryTools(ROOT)
    state = run_agent("loop", lambda _: {"type": "tool", "name": "list_stages", "arguments": {}}, tools, maximum_steps=2)
    assert not state.done
    assert len(state.actions) == 2


def test_agent_can_finish() -> None:
    tools = RepositoryTools(ROOT)
    decisions = iter([{"type": "tool", "name": "search", "arguments": {"keyword": "Transformer", "maximum_results": 2}}, {"type": "finish", "answer": "done"}])
    state = run_agent("search", lambda _: next(decisions), tools)
    assert state.done and state.answer == "done"
