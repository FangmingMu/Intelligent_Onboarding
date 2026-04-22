import os
from langchain_text_splitters import MarkdownHeaderTextSplitter

def simple_semantic_parse(data_dir: str = "stage_2_rag/data"):
    """
    使用 LangChain 原生的 MarkdownHeaderTextSplitter 进行语义切分。
    按 # ## ### 层级切分，确保逻辑块完整。
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_docs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 语义切分
                splits = splitter.split_text(content)
                # 注入文件名元数据
                for split in splits:
                    split.metadata["source"] = filename
                all_docs.extend(splits)

    print(f"✅ LangChain 解析完成: 共生成 {len(all_docs)} 个文档片段。")
    return all_docs

if __name__ == "__main__":
    docs = simple_semantic_parse()
    if docs:
        print(f"预览第一个片段内容: {docs[0].page_content[:50]}...")
        print(f"预览第一个片段元数据: {docs[0].metadata}")
