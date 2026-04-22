import os
import httpx
from dotenv import load_dotenv
from typing import List

# LangChain 相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 导入解析器
from doc_parser import simple_semantic_parse

# 加载配置
load_dotenv()

# --- 1. 初始化 Embedding 模型 (禁用代理) ---
def get_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("QWEN_EMBEDDING_MODEL_NAME"),
        openai_api_key=os.getenv("QWEN_EMBEDDING_API_KEY"),
        openai_api_base=os.getenv("QWEN_EMBEDDING_API_FULL_URL"),
        http_client=httpx.Client(proxy=None, timeout=60.0)
    )

# --- 2. 初始化大模型 (禁用代理) ---
def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        http_client=httpx.Client(proxy=None, timeout=60.0),
        temperature=0
    )

# --- 3. 真实 Rerank 实现 (禁用代理) ---
def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    调用您的私有 Reranker 接口进行精排。
    """
    api_url = os.getenv("RERANK_API_URL")
    api_key = os.getenv("RERANK_API_KEY")
    model_name = os.getenv("RERANK_MODEL")

    texts = [doc.page_content for doc in docs]
    payload = {
        "model": model_name,
        "query": query,
        "documents": texts
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(proxy=None, timeout=30.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()["results"]
            
            # 按分数排序并筛选
            ranked_docs = []
            for res in results[:top_n]:
                doc = docs[res["index"]]
                ranked_docs.append(doc)
            return ranked_docs
    except Exception as e:
        print(f"⚠️ Rerank 失败，降级使用原始排序: {e}")
        return docs[:top_n]

# --- 4. 构建或加载向量数据库 ---
FAISS_INDEX_PATH = "stage_2_rag/faiss_index"

def build_or_load_db():
    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"[Indexer] 正在加载 FAISS 索引...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"[Indexer] 正在构建索引...")
        langchain_docs = simple_semantic_parse()
        vector_db = FAISS.from_documents(langchain_docs, embeddings)
        vector_db.save_local(FAISS_INDEX_PATH)
        return vector_db

if __name__ == "__main__":
    db = build_or_load_db()
    llm = get_llm()
    
    query = "研发中心新员工如何配置 VPN？"
    
    # 第一步：初步召回 (Recall)
    initial_docs = db.similarity_search(query, k=10)
    
    # 第二步：精排 (Rerank)
    print(f"[Rerank] 正在对 {len(initial_docs)} 个片段进行精排...")
    final_docs = rerank_documents(query, initial_docs)
    
    # 第三步：生成回答
    context = "\n\n".join([d.page_content for d in final_docs])
    prompt = f"你是一个企业入职助手。请根据以下已知信息，简明扼要地回答。如果信息不足，请直说不知道。\n\n已知信息：\n{context}\n\n问题：{query}"
    
    print("[LLM] 正在生成回答...")
    response = llm.invoke(prompt)
    print(f"\n🤖 最终回答：\n{response.content}")
