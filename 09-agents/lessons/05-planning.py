"""把目标拆成有限步骤，并给每一步写清成功条件。"""


def build_plan(goal: str) -> list[dict]:
    return [
        {"step": "inspect", "success": "读取目标范围和当前状态"},
        {"step": "change", "success": f"完成与目标相关的最小变更：{goal}"},
        {"step": "verify", "success": "测试或证据通过"},
        {"step": "report", "success": "说明结果、限制与未完成项"},
    ]


def main() -> None:
    plan = build_plan("补全课程")
    print(*plan, sep="\n")
    assert plan[-1]["step"] == "report"


if __name__ == "__main__":
    main()
