# 模块二：语义级检索引擎 (RAG Engine) 笔记

## 1. 核心目标
解决传统“基于字数”切分导致的以下痛点：
- **表格乱码**：随机切分会打碎表格结构，导致数据检索不可读。
- **嵌套列表断裂**：步骤类指令（如配置指南）被切开后，LLM 会因为缺乏上下文而无法正确回答。

## 2. 语义解析方案 (Semantic Parsing)
- **框架**：LlamaIndex
- **方案**：`MarkdownNodeParser`
  - **KISS 实现**：利用标题层级（H1, H2...）作为逻辑分界线进行切分。
  - **Metadata 注入**：每一个 Node 自动携带 `{"Header 1": "..."}` 的层级信息，这在检索阶段能极大地辅助 LLM 理解当前内容的语义上下文。
  - **列表/表格保护**：因为表格和列表通常归属于某个标题之下，该方案能确保它们在语义节点内的完整性。

## 3. 架构优势 (面试高光点)
- **“语义感”优先**：相比 LangChain 的 `MarkdownHeaderTextSplitter`，LlamaIndex 的实现更轻量，且 Node 结构天然适合后续的多路召回。
- **最小代码量实现**：仅通过约 20 行核心代码即完成了复杂的企业文档语义建模。

---
[GitHub Repo: Intelligent_Onboarding](https://github.com/FangmingMu/Intelligent_Onboarding.git)
