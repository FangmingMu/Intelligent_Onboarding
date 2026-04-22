# 研发中心智能入职助手 (Intelligent Onboarding)

本项目旨在通过 RAG（检索增强生成）与自动化脚本技术，帮助新员工快速完成从“文档查询”到“环境配置”的全流程入职。

## 🚀 项目架构 (Stage-based)

### Stage 1: Gateway (进行中)
- 统一接入层，负责用户交互路由。

### Stage 2: RAG Engine (已完成核心构建)
- **核心逻辑**：采用 LlamaIndex 框架实现标准 RAG 流程。
- **解耦设计**：
  - `config.py`: 适配私有 LLM、向量模型 (Qwen) 与 重排模型 (Reranker)。
  - `indexer.py`: 业务流水线，清晰标注 [解析-索引-重排-生成] 流程。
- **数据源**：支持 Markdown 格式的研发环境配置指南、审批矩阵及合规手册。

### Stage 3: Action System (待开始)
- **目标**：根据 RAG 提供的方案，自动执行环境检测与配置脚本。

### Stage 4: Observation (待开始)
- **目标**：监控配置进度，记录执行日志。

## 🛠️ 当前进度
- [x] 适配私有 LLM 接口与参数过滤。
- [x] 集成自定义向量模型与 Reranker。
- [x] 实现索引持久化与缓存加载。
- [x] 重构代码实现配置与逻辑分离。
- [ ] **下一阶段：混合检索 (Hybrid Search) 优化**

## 📦 快速启动
1. 配置 `.env` 文件。
2. 运行 `python stage_2_rag/indexer.py` 进行 RAG 测试。
