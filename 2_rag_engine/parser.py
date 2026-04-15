from __future__ import annotations
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

def simple_semantic_parse(data_dir: str = "2_rag_engine/data"):
    """
    遵循 KISS 原则的语义解析流程：
    1. 自动加载 Markdown 文档。
    2. 按标题层级（# ## ###）切分，确保表格与步骤列表不被截断。
    3. 每个 Node 自动携带标题元数据。
    """
    # 1. 加载文档
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # 2. 核心解析器：MarkdownNodeParser (LlamaIndex 内置语义解析)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    # 3. 结果验证 (Minimalist Output)
    print(f"✅ 解析成功: 已将文档转化为 {len(nodes)} 个语义节点。")
    for i, node in enumerate(nodes[:3]):
        print(f"\n[Node {i+1}] Metadata: {node.metadata}")
        print(f"Content Preview: {node.get_content()[:80].strip()}...")
        
    return nodes

if __name__ == "__main__":
    if not os.path.exists("2_rag_engine/data"):
        print("❌ 错误: 未找到数据目录 2_rag_engine/data")
    else:
        simple_semantic_parse()
