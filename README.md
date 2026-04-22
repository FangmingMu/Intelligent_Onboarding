# 🚀 企业级智能入职与 IT 运维 Agent (Intelligent Onboarding & IT Agent)

这是一个专为企业内网环境设计的“文档驱动型”智能助手。它不只是一个聊天机器人，而是一个具备**自主决策、知识检索、业务办理及运行监控**能力的生产级 Agent 系统。

---

## 🌟 核心价值与业务场景
*   **新员工破冰**：自动解答入职指南、行政政策、财务报销等琐碎问询。
*   **IT 工单拦截**：通过 RAG 自动解决 VPN、邮箱、环境配置等技术问题，将无法解决的问题自动转化为结构化工单。
*   **业务自动化**：集成内部 API，实现员工信息反查、权限申请、工单提交等闭环操作。

---

## 🏗️ 4-Tier 模块化架构设计
系统采用严格的解耦架构，确保了每一层的可维护性与扩展性：

### 1️⃣ Stage 1: 智能网关层 (Gateway & Routing)
*   **大脑前额叶**：利用结构化输出（Structured Output）实现**意图路由**，精准分发请求至 RAG、Action 或 Chat 路径。
*   **上下文感应**：采用滑动窗口记忆（Window Memory），解决 Agent 在多轮交互中的**指代消解**问题。
*   **现代交互**：基于 Streamlit 构建，支持实时思考过程展示（Thought Trace）。

### 2️⃣ Stage 2: 工业级 Hybrid RAG 引擎
*   **混合检索架构 (Hybrid Search)**：结合了 **BM25 (关键词检索)** 与 **Vector (语义搜索)** 的加权融合（Ensemble Retriever），有效解决了纯向量检索在处理强特征关键词（如 IP 地址、密码规则）时召回率低的问题。
*   **两阶段精排**：召回后再引入 **BGE-Reranker** 进行二次过滤，确保最终注入 Prompt 的知识片段具备极高的相关性。
*   **细粒度语义切分**：采用二级切分策略（Markdown Header + Recursive Overlap），在保持段落逻辑完整性的同时，增加片段间的上下文重叠，减少检索“碎片化”现象。

### 3️⃣ Stage 3: 安全执行层 (Action & Safety)
*   **确定性执行**：拒绝不可控的代码生成（Code Interpreter），采用 **Tool Calling** 范式实现受控的业务操作。
*   **强类型契约**：使用 **Pydantic** 进行 API 参数的 Runtime 校验，确保非法参数在进入业务系统前被拦截。
*   **防御机制**：内置 **Human-in-the-loop (人机拦截)** 与**指数退避重试**，确保高危动作可控，网络波动下具备鲁棒性。

### 4️⃣ Stage 4: 可观测性与评测层 (Observability & Eval)
*   **全链路追踪**：自建日志系统，捕获每一次调用的**耗时、Token 消耗、决策路径**。
*   **反馈闭环**：集成点赞/点踩评价组件，通过 `request_id` 自动关联坏例，驱动 RAG 知识库持续优化。
*   **运行看板**：内置监控 Dashboard，实时呈现用户满意度与系统性能指标。

---

## 🛠️ 技术栈核心
| 维度 | 选型 | 理由 |
| :--- | :--- | :--- |
| **LLM 框架** | LangChain | 灵活的组件绑定与成熟的 Tool Calling 支持 |
| **文档解析** | LlamaIndex (Parser) | 优秀的 Markdown 语义切分能力 |
| **向量存储** | FAISS | 极轻量、高性能，支持私有化部署 |
| **模型接口** | OpenAI 兼容 SDK | 适配私有化部署的大模型（gpt-oss-120b） |
| **监控分析** | Pandas + JSONL | 高性能流式日志写入与数据分析 |

---

## 🚀 快速启动

### 1. 环境准备
```bash
# 克隆项目并安装依赖
pip install -r requirements.txt
```

### 2. 配置文件
创建 `.env` 文件并配置以下项：
```env
OPENAI_API_KEY=your_key
OPENAI_API_BASE=your_url
LLM_MODEL=gpt-oss-120b
QWEN_EMBEDDING_API_FULL_URL=...
RERANK_API_URL=...
```

### 3. 启动应用
```bash
streamlit run stage_1_gateway/app.py
```

---

## 📊 RAG 自动化评测成果 (Ragas Framework)

为了量化评估系统的可靠性，引入了 **Ragas** 评测框架，基于手写的 20 组“黄金问答对”进行了全量评估：

| 评估指标 | 最终得分 | 指标含义 | 优化贡献 |
| :--- | :--- | :--- | :--- |
| **Context Recall** | **0.9000** | 召回率：搜到的东西是否有答案 | **Hybrid Search** 引入 BM25 解决强特征召回 |
| **Context Precision** | **0.9167** | 精确度：最相关的片段是否排在前面 | **BGE-Reranker** 对候选片段的精准二次过滤 |
| **Faithfulness** | **0.8262** | 忠实度：是否根据文档回答（无幻觉） | **Prompt Engineering** 与严格的上下文约束 |
| **Answer Relevancy** | **0.7654** | 相关性：回答是否切中要害 | 大模型语义理解力与系统 Prompt 调优 |

> **技术结论**：通过从单向量检索演进到 **Hybrid Search + Re-rank** 架构，`Context Recall` 实现了从 0 到 0.9 的跨越式增长，系统具备了处理企业级复杂术语与硬核知识的能力。

---

## 📄 许可证
本项目仅用于技术交流与面试演示。
