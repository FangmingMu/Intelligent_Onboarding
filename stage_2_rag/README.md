# 2. 深度检索层 (Advanced RAG Engine)

## 职责描述
本层级是系统的核心知识库，负责将非结构化文档转化为可检索的语义节点：
- **文档解析 (Parsing)**：处理 Markdown、PDF 等企业内部文档。
- **语义切分 (Semantic Chunking)**：基于 LlamaIndex 的 MarkdownNodeParser 进行标题级切分，保留上下文。
- **多路召回 (Hybrid Search)**：结合向量检索与 BM25 关键词检索，确保专有名词的准确性。
- **重排 (Rerank)**：对检索结果进行精排，提升回答的精确度。
