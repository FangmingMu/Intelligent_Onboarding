import os
import httpx
from dotenv import load_dotenv
from typing import List

# LangChain 相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

# 导入解析器 (采用兼容性导入)
try:
    from doc_parser import simple_semantic_parse
except ImportError:
    from stage_2_rag.doc_parser import simple_semantic_parse

# 加载配置
load_dotenv()

# --- 1. 初始化模型 ---
def get_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("QWEN_EMBEDDING_MODEL_NAME"),
        openai_api_key=os.getenv("QWEN_EMBEDDING_API_KEY"),
        openai_api_base=os.getenv("QWEN_EMBEDDING_API_FULL_URL"),
        http_client=httpx.Client(proxy=None, timeout=60.0)
    )

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        http_client=httpx.Client(proxy=None, timeout=60.0),
        temperature=0
    )

# --- 2. 真实 Rerank 实现 ---
def rerank_documents(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    api_url = os.getenv("RERANK_API_URL")
    api_key = os.getenv("RERANK_API_KEY")
    model_name = os.getenv("RERANK_MODEL")

    if not docs: return []
    
    texts = [doc.page_content for doc in docs]
    payload = {"model": model_name, "query": query, "documents": texts}
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(proxy=None, timeout=30.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()["results"]
            ranked_docs = [docs[res["index"]] for res in results[:top_n]]
            return ranked_docs
    except Exception as e:
        print(f"⚠️ Rerank 失败，降级使用原始排序: {e}")
        return docs[:top_n]

# --- 3. 检索器构建 ---
FAISS_INDEX_PATH = "stage_2_rag/faiss_index"

def get_vector_retriever(k=10):
    """【模式 A】纯向量检索"""
    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        langchain_docs = simple_semantic_parse()
        vector_db = FAISS.from_documents(langchain_docs, embeddings)
        vector_db.save_local(FAISS_INDEX_PATH)
    return vector_db.as_retriever(search_kwargs={"k": k})

def get_hybrid_retriever(k=10):
    """【模式 B】混合检索"""
    embeddings = get_embeddings()
    langchain_docs = simple_semantic_parse()
    
    if os.path.exists(FAISS_INDEX_PATH):
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_db = FAISS.from_documents(langchain_docs, embeddings)
        vector_db.save_local(FAISS_INDEX_PATH)
    
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = k
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

def build_or_load_db(mode="Hybrid"):
    """
    提供统一入口，支持 mode 参数进行检索模式切换。
    mode: "Vector" 或 "Hybrid"
    """
    print(f"[DEBUG] 正在使用检索模式: {mode}")
    if mode == "Vector":
        return get_vector_retriever()
    return get_hybrid_retriever()

if __name__ == "__main__":
    # 冒烟测试
    retriever = build_or_load_db(mode="Hybrid")
    print("✅ 混合检索器初始化成功")
