# Stage 3: 受控动作执行层 (Action & LangGraph)

本模块将 Agent 从“聊天机器人”升级为具备“业务办理能力”的智能助手。

---

## 🏗️ 基于 LangGraph 的状态机架构

### 1. 为什么弃用 ReAct 循环？
传统的 `while` 循环 Agent 在面对复杂场景时极易陷入死循环，且状态难以保存。本项目升级为 **LangGraph 有向图架构**：
*   **状态节点化**：思考、工具调用、人工审批被定义为明确的顶点。
*   **路由透明化**：逻辑流转通过条件边（Conditional Edges）严格控制。

### 2. Human-in-the-loop (人类在环审批)
系统内置了生产级的安全防御机制：
*   **触发条件**：当 Agent 试图调用 `submit_it_ticket` 且 `priority` 为 `P0` 或涉及 `Root` 权限时。
*   **执行逻辑**：图会自动流向 `approval` 节点并挂起。
*   **恢复机制**：状态被保存在 SQLite 中，直至人类管理员在 UI 界面点击“批准”。

### 3. 持久化存储 (SQLite Checkpointer)
*   **断点续传**：每一个节点的变动都会自动序列化。
*   **会话隔离**：通过 `thread_id` 区分不同用户的对话状态。
*   **可靠性**：即使服务器重启，Agent 也能想起之前处理到一半的审批任务。

---

## 🛠️ 工具箱 (Tools)
*   `query_employee_info`: 反查员工档案，获取入职日期与工号。
*   `submit_it_ticket`: 提交结构化工单，支持自动参数校验（Pydantic）。
