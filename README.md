# Intelligent Onboarding & IT Agent

这是一个生产级的“文档驱动型”智能入职与 IT 工单处理 Agent 系统。采用解耦的四层架构，集成了 RAG 检索、工具调用、安全防御与实时监控。

## 🏗️ 架构概览

- **Stage 1: Gateway (网关层)**: 基于 Streamlit 的 UI 与基于上下文的意图路由引擎。
- **Stage 2: RAG (知识检索层)**: 纯 LangChain 实现，支持语义切分、向量搜索与 Rerank。
- **Stage 3: Action (执行层)**: 工具自动化调度、Pydantic 参数强校验与安全拦截。
- **Stage 4: OBS (观测层)**: 全链路 Tracing、Token 消耗统计与用户反馈闭环。

## 🚀 快速启动

1.  **环境安装**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **配置变量**: 修改 `.env` 文件，注入 LLM/Embedding/Rerank 接口信息。
3.  **运行应用**:
    ```bash
    streamlit run stage_1_gateway/app.py
    ```

## 🛠️ 技术亮点
- **工业级 RAG**: 召回+重排两阶段架构，解决长文本噪音问题。
- **确定性执行**: 拒绝代码生成，采用 Tool Calling 实现受控的业务逻辑。
- **可观测性**: 自建 JSONL 日志追踪，支持基于真实数据的持续迭代。
- **上下文感知**: 路由与执行均具备滑动窗口记忆，支持复杂多轮对话。
