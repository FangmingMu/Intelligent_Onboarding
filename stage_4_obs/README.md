# Stage 4: 可观测性与评测层 (Observability & Eval Layer)

## 概述
本模块负责整个 Agent 系统的**全链路追踪**与**持续评估**。通过记录真实运行数据与用户反馈，为系统的性能优化和效果迭代提供数据支撑。

## 核心实现

1.  **全链路追踪 (Tracing)**：
    *   **指标捕获**：集成 LangChain 的 `get_openai_callback`，实时统计单次请求的 Token 消耗（Prompt/Completion/Total）。
    *   **耗时监控**：记录端到端响应延迟（Latency），单位为毫秒。
    *   **日志持久化**：采用 JSONL 格式存储于 `stage_4_obs/logs.jsonl`，确保数据可读性与易扩展性。

2.  **用户反馈与评测 (Eval)**：
    *   **显式点赞/踩**：在 UI 层集成 `st.feedback("thumbs")`。
    *   **反馈关联**：通过唯一的 `request_id` 将用户评价与对应的追踪日志进行强关联，便于定位坏例 (Bad Cases)。

3.  **监控大屏 (Dashboard)**：
    *   提供独立的可视化视图，包含：
        *   **核心 Metrics**：总请求数、平均耗时、累计 Token 消耗、用户点赞率。
        *   **流量分布**：RAG、ACTION、CHAT 各路径的触发频率。
        *   **性能趋势**：响应延迟的波动情况。

## 运行说明
1.  启动应用：`streamlit run stage_1_gateway/app.py`
2.  在左侧侧边栏切换至“监控看板”即可查看实时统计数据。
