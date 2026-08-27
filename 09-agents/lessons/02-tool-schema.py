"""给只读搜索工具写一份边界明确的 JSON Schema。"""

import json


SEARCH_SCHEMA = {"name": "search", "description": "在课程 Markdown/Python 文件中搜索关键词；只读。", "input_schema": {"type": "object", "properties": {"keyword": {"type": "string", "minLength": 1}, "maximum_results": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["keyword"], "additionalProperties": False}}


def main() -> None:
    print(json.dumps(SEARCH_SCHEMA, ensure_ascii=False, indent=2))
    assert SEARCH_SCHEMA["input_schema"]["additionalProperties"] is False


if __name__ == "__main__":
    main()
