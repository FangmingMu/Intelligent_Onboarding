import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from typing import List

def run_semantic_parsing_pipeline(data_path: str = "2_rag_engine/data"):
    """
    运行语义级文档解析 Pipeline。
    
    1. 使用 SimpleDirectoryReader 加载 .md 文件。
    2. 使用 MarkdownNodeParser 按标题层级进行切分。
    3. 打印解析出的 Node 数量及元数据，验证层级关系。
    """
    print(f"[RAG Engine] 开始加载文档: {data_path}...")
    
    # 1. 加载本地知识库
    reader = SimpleDirectoryReader(
        input_dir=data_path,
        required_exts=[".md"]
    )
    documents = reader.load_data()
    
    # 2. 核心解析器：MarkdownNodeParser (标题级切分)
    # 相比于按字数切分的 SentenceSplitter，它能确保每一段逻辑都在同一个标题下。
    parser = MarkdownNodeParser()
    
    nodes = parser.get_nodes_from_documents(documents)
    
    # 3. 验证解析结果
    print(f"✅ 语义解析完成，共生成 {len(nodes)} 个语义节点 (Nodes)。")
    
    for i, node in enumerate(nodes[:5]):  # 展示前 5 个节点以供校验
        print(f"\n--- Node {i+1} ---")
        # 打印元数据，展示标题层级
        print(f"元数据 (Metadata): {node.metadata}")
        # 打印内容片段（前 100 字）
        content_preview = node.get_content().strip().replace('\n', ' ')[:100]
        print(f"内容预览: {content_preview}...")

    return nodes

if __name__ == "__main__":
    # 运行解析流程
    all_nodes = run_semantic_parsing_pipeline()
