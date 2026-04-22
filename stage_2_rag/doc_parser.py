import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def simple_semantic_parse(data_dir: str = "stage_2_rag/data"):
    """
    二级切分策略：
    1. 第一层：按 Markdown 标题 (#, ##, ###) 进行语义切分，保持逻辑块完整。
    2. 第二层：对切分后的块进行 RecursiveCharacterTextSplitter 细化，确保片段大小适中并增加 Overlap。
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # 初始化切分器
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 更细粒度的块
        chunk_overlap=50, # 增加重叠，减少截断
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    all_docs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # 第一步：标题切分
                header_splits = header_splitter.split_text(content)
                
                # 第二步：对每个块进行长度细分
                final_splits = text_splitter.split_documents(header_splits)
                
                # 注入文件名元数据
                for split in final_splits:
                    split.metadata["source"] = filename
                all_docs.extend(final_splits)

    print(f"✅ 细粒度语义解析完成: 共生成 {len(all_docs)} 个文档片段。")
    return all_docs

if __name__ == "__main__":
    docs = simple_semantic_parse()
    if docs:
        print(f"预览第一个片段内容: {docs[0].page_content[:100]}...")
        print(f"预览第一个片段元数据: {docs[0].metadata}")
