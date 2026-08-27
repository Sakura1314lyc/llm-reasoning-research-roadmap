"""执行白名单工具，并把输入错误整理成结构化结果。"""


def execute(registry: dict, call: dict) -> dict:
    name = call.get("name")
    if name not in registry:
        return {"ok": False, "error": "unknown_tool"}
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "invalid_arguments"}
    try:
        return {"ok": True, "result": registry[name](**arguments)}
    except (TypeError, ValueError, PermissionError) as error:
        return {"ok": False, "error": type(error).__name__, "message": str(error)}


def main() -> None:
    result = execute({"add": lambda a, b: a + b}, {"name": "add", "arguments": {"a": 2, "b": 3}})
    print(result)
    assert result == {"ok": True, "result": 5}


if __name__ == "__main__":
    main()
