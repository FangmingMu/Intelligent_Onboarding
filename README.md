# 🚀 企业级智能入职与 IT 运维 Agent (LangGraph 版)

这是一个专为企业内网设计的“文档驱动型”智能助手，采用了 **LangChain + LangGraph** 的下一代 Agent 架构。系统实现了从语义检索到受控执行的全链路闭环，并具备高度的可观测性与自动化评测能力。

---

## 🏗️ 4-Tier 模块化架构演进

### 1️⃣ Stage 1: 智能网关与状态持久化 (Gateway & Persistence)
*   **状态机接入**：基于 **LangGraph** 实现状态流转，支持多轮对话的“指代消解”。
*   **断点续传**：利用 **Thread ID** 与 SQLite Checkpointer，实现对话状态的跨 Session 持久化。
*   **实时追踪**：Streamlit UI 实时渲染 Agent 的思考链路（Thought Trace）。

### 2️⃣ Stage 2: 工业级 Hybrid RAG 引擎
*   **混合检索架构**：结合 **BM25** (关键词) 与 **Vector** (语义) 的 Ensemble 召回，解决 IP/专有名词召回痛点。
*   **双阶段精排**：集成 **BGE-Reranker**，实现从召回到精排的精度飞跃。
*   **评测闭环**：内置 **Ragas 自动化打分看板**，实时监控 Faithfulness、Recall 等四项核心指标。

### 3️⃣ Stage 3: 受控执行与人类在环 (Action & HIL)
*   **状态机架构 (DAG)**：弃用简单循环，采用 LangGraph 构建 Agent 执行图，确保逻辑透明、可控。
*   **Human-in-the-loop**：针对 P0 工单、Root 权限申请等高危操作实现**自动挂起与人工审批**。
*   **强类型校验**：基于 **Pydantic** 实现工具调用的 Runtime 参数验证。

### 4️⃣ Stage 4: 全链路观测与反馈 (OBS & Feedback)
*   **原子级追踪**：记录每次请求的 Latency、Token 消耗及决策路径。
*   **反馈闭环**：集成点赞/点踩机制，通过 `request_id` 自动关联坏例，驱动知识库持续迭代。

---

## 📊 核心技术栈
| 组件 | 选型 | 核心价值 |
| :--- | :--- | :--- |
| **Agent 架构** | **LangGraph** | 解决循环死循环问题，支持受控状态机流转 |
| **持久化层** | **SQLite Checkpointer** | 实现 HIL（人类在环）场景下的断点续传 |
| **RAG 引擎** | **Hybrid Search + Rerank** | 召回率 (Recall) 达 0.90，精确度 (Precision) 提升 5% |
| **评测框架** | **Ragas** | 实现 RAG 系统从“感性感觉”向“理性量化”的转变 |

---

## 🚀 快速启动
1. **安装环境**: `pip install -r requirements.txt`
2. **配置环境**: 完善 `.env` 中的 API 密钥。
3. **启动应用**: `streamlit run stage_1_gateway/app.py`

---

## 📄 许可证
本项目仅用于技术交流与面试演示。
