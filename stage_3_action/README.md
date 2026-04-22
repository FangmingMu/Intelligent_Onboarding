# Stage 3: 动作执行与工具调用 (Action Layer)

## 核心职责
赋能 Agent 具备“查库”与“写工单”的实操能力，实现业务闭环。

## 技术栈
- **LangChain Tool Calling**: 规范化的外部工具绑定。
- **Pydantic**: 严格的参数类型强校验。
- **ReAct Framework**: 思考-行动-观察的循环逻辑。

## 关键安全机制
- **JSON 强制化**: 确保模型输出严格符合 API Schema。
- **人机交互拦截 (Human-in-the-loop)**：对高危权限或 P0 工单进行拦截，要求人工审批。
- **自动重试 (Retry Logic)**: 内置指数退避重试，应对真实 API 的网络抖动。
- **多步推理**: 自动发现前置依赖（如：提工单前先自动反查工号）。
