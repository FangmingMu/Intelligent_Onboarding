# Stage 1: 接入与路由层 (Gateway & Routing Layer)

## 核心职责
作为系统的“大脑前额叶”，负责感知用户意图并分发任务。

## 技术栈
- **Streamlit**: 构建现代化、响应式的聊天 UI。
- **LangChain Intent Router**: 基于大模型的语义理解，将请求分发至 RAG、ACTION 或 CHAT 路径。
- **Context Management**: 维护滑动窗口式的对话记忆，支持多轮交互。

## 关键功能
- **意图路由**: 自动识别知识问答、业务办理与闲聊。
- **状态感知**: 实时展示 Agent 的思考过程与决策路径。
- **Session 持久化**: 在内存中管理用户对话上下文。
