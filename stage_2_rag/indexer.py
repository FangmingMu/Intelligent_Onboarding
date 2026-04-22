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

# 导入解析器
from doc_parser import simple_semantic_parse

# 加载配置
load_dotenv()

# --- 1. 初始化模型 (与之前一致) ---
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

# --- 3. 混合检索构建 (Hybrid Search) ---
FAISS_INDEX_PATH = "stage_2_rag/faiss_index"

def get_hybrid_retriever():
    """
    构建混合检索器：BM25 (关键词) + FAISS (向量)
    """
    embeddings = get_embeddings()
    
    # 1. 获取所有文档用于 BM25
    langchain_docs = simple_semantic_parse()
    
    # 2. 构建或加载向量库
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"[Indexer] 加载本地 FAISS 索引...")
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"[Indexer] 构建新 FAISS 索引...")
        vector_db = FAISS.from_documents(langchain_docs, embeddings)
        vector_db.save_local(FAISS_INDEX_PATH)
    
    # 3. 构建 BM25 检索器 (关键词匹配)
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = 10  # 关键词召回数量
    
    # 4. 构建向量检索器
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # 5. 组合检索器 (Ensemble)
    # weights 为分配给每个检索器的权重，通常各占 0.5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def build_or_load_db():
    # 为了兼容 app.py，保持该函数名，但内部返回混合检索逻辑
    # 注意：在大型项目中，建议将 retriever 单独管理
    return get_hybrid_retriever()

if __name__ == "__main__":
    # 冒烟测试
    retriever = get_hybrid_retriever()
    llm = get_llm()
    
    query = "研发中心新员工 VPN 初始密码是什么？"
    
    # 1. 混合检索 (Recall)
    print(f"\n[Hybrid Search] 正在检索: {query}")
    initial_docs = retriever.invoke(query)
    
    # 2. 精排 (Rerank)
    print(f"[Rerank] 正在对 {len(initial_docs)} 个片段进行精排...")
    final_docs = rerank_documents(query, initial_docs)
    
    # 3. 生成回答
    context = "\n\n".join([d.page_content for d in final_docs])
    prompt = f"你是一个企业入职助手。请严格根据以下已知信息，详细回答问题。如果信息中没有提到，请说不知道。\n\n已知信息：\n{context}\n\n问题：{query}"
    
    print("[LLM] 正在生成回答...")
    response = llm.invoke(prompt)
    print(f"\n🤖 最终回答：\n{response.content}")
