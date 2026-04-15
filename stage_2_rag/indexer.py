import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from stage_2_rag.parser import simple_semantic_parse

# 1. 加载环境变量 (KISS 配置管理)
load_dotenv()

# 2. 全局配置 (Global Settings)
# 适配您提供的 API 地址与模型
Settings.llm = OpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE")
)

Settings.embed_model = OpenAIEmbedding(
    model=os.getenv("EMBED_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE")
)

STORAGE_DIR = "stage_2_rag/storage"

def build_or_load_index():
    """
    语义索引构建器：
    - 支持本地持久化，避免重复调用 Embedding API 产生费用。
    - 自动适配 BGE-m3 模型。
    """
    if os.path.exists(STORAGE_DIR):
        print(f"[Indexer] 正在加载本地索引...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print("[Indexer] 正在解析文档并调用 Embedding (BGE-m3)...")
        nodes = simple_semantic_parse()
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"✅ 索引已持久化至 {STORAGE_DIR}")
        
    return index

if __name__ == "__main__":
    # 执行索引构建与冒烟测试
    idx = build_or_load_index()
    
    # 模拟真实检索 (启用 Top-K)
    query_engine = idx.as_query_engine(similarity_top_k=5)
    response = query_engine.query("研发中心环境配置中关于 Git 的步骤有哪些？")
    print(f"\n🔍 检索响应结果:\n{response}")
