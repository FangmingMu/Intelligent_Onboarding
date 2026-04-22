# Stage 4: 全链路可观测性 (Observability)

本模块确保了 Agent 系统的“可审计”与“持续进化”。

---

## 📊 核心监控维度

### 1. 运行时追踪 (Tracer)
*   **Latency**：监控意图识别、RAG 检索、LLM 生成及工具调用的分步耗时。
*   **Token Usage**：统计每一轮对话的 Prompt、Completion 及 Total Tokens，辅助成本控制。
*   **Destination**：记录流量在 RAG、Action 与 Chat 之间的分布比例。

### 2. 用户反馈闭环 (Feedback Loop)
*   **点赞/点踩**：集成 Streamlit 反馈组件。
*   **关联存储**：评价数据通过 `request_id` 与原始日志自动关联。
*   **坏例分析**：日志以 JSONL 格式持久化，便于通过脚本提取被点踩的案例进行 Case Study。

### 3. 系统看板 (Monitoring Dashboard)
在 `app.py` 的监控视图中，实时呈现：
*   **满意度趋势**：基于反馈计算的点赞率。
*   **性能热力**：平均响应耗时的波动情况。
*   **实时日志流**：展示最近 10 次交互的详细技术参数。
