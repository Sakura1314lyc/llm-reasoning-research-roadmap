# 阶段 9：LLM Agent、Skill、MCP 与软件测试

这一阶段把模型从“生成文本”扩展为“观察环境、选择工具、执行动作、检查结果并继续”的可测试系统。

建议用时：**7–10 天，每天 3–4 小时**。

## 前置知识

- 能调用一个支持工具描述或结构化输出的模型
- 熟悉 Python 函数、类型标注、JSON 和异常处理
- 能使用 pytest 编写基本测试
- 理解外部工具调用会产生真实副作用

## 完成后应该具备的能力

- 实现基本 Agent Loop
- 为工具设计清晰 Schema、类型和错误返回
- 区分 Tool、Resource 和 Prompt
- 构建并测试一个最小 MCP Server
- 将重复工作流整理为可复用 Skill
- 记录 Agent 轨迹并分析失败原因
- 对文件、网络和外部操作设置安全边界

## 推荐学习顺序

| 顺序 | 学习内容 | 建议代码文件 | 完成标准 |
| ---: | --- | --- | --- |
| 01 | Agent Loop | `lessons/01-agent-loop.py` | 完成 observe → decide → act → observe 循环 |
| 02 | Tool Schema | `lessons/02-tool-schema.py` | 为函数定义参数、返回值、异常和文档 |
| 03 | Tool Execution | `lessons/03-tool-executor.py` | 能验证输入、执行工具并序列化结果 |
| 04 | Memory/State | `lessons/04-agent-state.py` | 区分会话状态、长期记忆和外部资源 |
| 05 | Planning | `lessons/05-planning.py` | 把复杂任务拆为可验证的有限步骤 |
| 06 | MCP Server | `lessons/06-mcp-server.py` | 暴露一个只读 Resource 和一个 Tool |
| 07 | Skill | `lessons/07-reusable-skill.md` | 把稳定流程写成带触发条件和验证步骤的说明 |
| 08 | Testing | `lessons/08-agent-testing.py` | 测试工具、循环终止、错误处理和权限边界 |
| 09 | Trajectory Analysis | `lessons/09-trajectory-analysis.py` | 对轨迹做失败分类和成本统计 |

## 第一个 Agent 不要做太复杂

建议实现一个“本地学习仓库助手”，只提供只读能力：

- 列出课程文件
- 查询某个阶段 README
- 搜索关键词
- 汇总学习进度

确认循环、日志与测试稳定后，再增加写文件或网络工具。

## 工具设计原则

每个工具应包含：

- 明确且单一的职责
- 类型化参数与返回值
- 可理解的名称和 docstring
- 输入验证
- 超时与异常处理
- 是否有副作用的说明
- 最小必要权限

不要用一个“万能 shell 工具”代替边界清晰的工具集合。

## MCP 学习顺序

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

## 安全原则

- 默认只读；写入、删除、发送消息等能力单独授权
- 外部内容可能包含 Prompt Injection，不能直接视为可信指令
- 工具返回值也是不可信输入，需要验证和限制长度
- 对路径、域名、命令和资源范围建立 allowlist
- 保存必要日志，但避免记录密钥和隐私数据
- 为循环设置最大步数、超时和预算

## 完成清单

- [ ] 手写一个有限步 Agent Loop
- [ ] 至少实现三个边界清晰的工具
- [ ] 完成只读 MCP Server
- [ ] 为 MCP Tool 编写测试
- [ ] 编写一个可复用 Skill
- [ ] 建立 20 个任务的回归集
- [ ] 完成轨迹与安全失败分析

## 权威资料

- [MCP Server Concepts](https://modelcontextprotocol.io/specification/latest/server)
- [Build an MCP Server](https://modelcontextprotocol.io/docs/develop/build-server)
- [MCP Tools Specification](https://modelcontextprotocol.io/specification/latest/server/tools)
- [pytest Getting Started](https://docs.pytest.org/en/stable/getting-started.html)

完成这一阶段后，回到根目录整理最终实验报告，并总结整个研究闭环。
