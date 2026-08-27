# 阶段 9：LLM Agent、Skill、MCP 与软件测试

到了这一阶段，模型不再只是生成一段文本。它要观察当前状态、选择工具、执行动作，再看结果决定下一步；整个过程还得能测试和复盘。

建议用时：**7–10 天，每天 3–4 小时**。

## 前置知识

- 能调用一个支持工具描述或结构化输出的模型
- 熟悉 Python 函数、类型标注、JSON 和异常处理
- 能使用 pytest 编写基本测试
- 理解外部工具调用会产生真实副作用

## 做到这些才算跑通

- 实现基本 Agent Loop
- 为工具设计清晰 Schema、类型和错误返回
- 区分 Tool、Resource 和 Prompt
- 构建并测试一个最小 MCP Server
- 将重复工作流整理为可复用 Skill
- 记录 Agent 轨迹并分析失败原因
- 对文件、网络和外部操作设置安全边界

## 先写循环，再接 MCP

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | Agent Loop | [lessons/01-agent-loop.py](lessons/01-agent-loop.py) | 已完成有限步 observe → decide → act 循环 |
| 02 | Tool Schema | [lessons/02-tool-schema.py](lessons/02-tool-schema.py) | 已定义参数、返回值、副作用和附加字段限制 |
| 03 | Tool Execution | [lessons/03-tool-executor.py](lessons/03-tool-executor.py) | 已实现白名单、输入验证与结构化错误 |
| 04 | Memory/State | [lessons/04-agent-state.py](lessons/04-agent-state.py) | 已区分会话状态、长期记忆和外部资源 |
| 05 | Planning | [lessons/05-planning.py](lessons/05-planning.py) | 已拆分 inspect/change/verify/report 步骤 |
| 06 | MCP Server | [lessons/06-mcp-server.py](lessons/06-mcp-server.py) | 已暴露只读 Resource 和 Tool |
| 07 | Skill | [lessons/07-reusable-skill.md](lessons/07-reusable-skill.md) | 已写明触发、输入、步骤、验证和安全边界 |
| 08 | Testing | [lessons/08-agent-testing.py](lessons/08-agent-testing.py) | 已测试路径越界、循环终止与工具选择 |
| 09 | Trajectory Analysis | [lessons/09-trajectory-analysis.py](lessons/09-trajectory-analysis.py) | 已统计成功率、步骤与失败类型 |

OpenClaw、Hermes、Codex、Claude Code 与 Harness 的实践框架见 [工具与 Harness 笔记](TOOLING-AND-HARNESS.md)。

## 第一个 Agent 越简单越好

建议实现一个“本地学习仓库助手”，只提供只读能力：

- 列出课程文件
- 查询某个阶段 README
- 搜索关键词
- 汇总学习进度

先把循环、日志和测试跑稳，再考虑写文件或联网工具。

## 工具要把边界写清楚

每个工具应包含：

- 明确且单一的职责
- 类型化参数与返回值
- 可理解的名称和 docstring
- 输入验证
- 超时与异常处理
- 是否有副作用的说明
- 最小必要权限

不要用一个“万能 shell 工具”代替边界清晰的工具集合。

## MCP 从只读能力开始

1. 理解 Client、Server 与 capability negotiation
2. 实现只读 Resource
3. 实现无副作用 Tool
4. 使用 Inspector/客户端测试 Schema 和返回值
5. 加入日志、超时和错误处理
6. 最后再考虑带写操作的工具

MCP 中常见的三类服务能力：

- Resources：提供上下文或数据
- Prompts：提供用户可选择的模板化工作流
- Tools：让模型发现并调用函数

## 测试矩阵

| 测试层级 | 重点 |
| --- | --- |
| 单元测试 | 参数校验、边界值、异常与返回格式 |
| 集成测试 | 模型产生的调用能否被工具执行 |
| 轨迹测试 | Agent 是否选择正确工具并及时停止 |
| 安全测试 | 路径穿越、Prompt Injection、越权与敏感信息泄露 |
| 回归测试 | 固定任务集上的成功率、步数、Token 和耗时 |

## 阶段项目：可测试的研究助手 Agent

```text
projects/01-research-agent/
├── agent.py
├── tools.py
├── state.py
├── mcp_server.py
├── policies.md
├── tests/
├── fixtures/
├── trajectories/
└── analysis.md
```

至少准备 20 个固定任务，统计：

- 任务成功率
- 平均工具调用次数
- 无效调用率
- 平均 Token/耗时
- 未终止或重复循环次数
- 各类失败占比

## 权限宁可先收紧

- 默认只读；写入、删除、发送消息等能力单独授权
- 外部内容可能包含 Prompt Injection，不能直接视为可信指令
- 工具返回值也是不可信输入，需要验证和限制长度
- 对路径、域名、命令和资源范围建立 allowlist
- 保存必要日志，但避免记录密钥和隐私数据
- 为循环设置最大步数、超时和预算

## 完成清单

- [x] 手写有限步 Agent Loop
- [x] 实现列阶段、读文本、搜关键词三个只读工具
- [x] 完成只读 MCP Resource 与 Tool 示例
- [x] 为工具、循环终止和路径边界编写测试
- [x] 编写可复用 Skill
- [x] 给出 20 个固定任务的回归指标规范
- [x] 完成轨迹与安全失败分类

## 权威资料

- [MCP Server Concepts](https://modelcontextprotocol.io/specification/latest/server)
- [Build an MCP Server](https://modelcontextprotocol.io/docs/develop/build-server)
- [MCP Tools Specification](https://modelcontextprotocol.io/specification/latest/server/tools)
- [pytest Getting Started](https://docs.pytest.org/en/stable/getting-started.html)

完成这一阶段后，回到根目录整理最终实验报告，并总结整个研究闭环。
