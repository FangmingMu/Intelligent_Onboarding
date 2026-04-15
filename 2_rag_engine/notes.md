# 模块二：语义级检索引擎 (RAG Engine) 笔记

## 1. 核心目标
解决传统文本切分（Fixed-size Chunking）在处理企业文档时的弊端：
- **表格乱码**：传统切分会随机切断表格行，导致语义丢失。
- **上下文断裂**：多级步骤列表（1.1.1）如果被切分在两个 Chunk 中，模型会因缺乏前置步骤而无法回答。

## 2. 语义解析方案 (Semantic Parsing)
- **框架**：LlamaIndex
- **解析器**：
  - `MarkdownNodeParser`：用于保持文档的 `Header` 层级关系，作为 Metadata 注入 Node。
  - `MarkdownElementNodeParser`：处理 Markdown 表格的专用解析器，支持表格摘要提取。
- **存储**：Qdrant (本地/生产)。

## 3. 开源仓库连接
[GitHub Repo: Intelligent_Onboarding](https://github.com/FangmingMu/Intelligent_Onboarding.git)
