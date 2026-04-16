import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings

# 导入自定义配置好的底层组件
from config import PrivateLLM, CustomQwenEmbedding, CustomReranker
from doc_parser import simple_semantic_parse

# 1. 环境准备 (Env Setup)
load_dotenv()

# 2. 全局组件注入 (Settings Injection)
# 这里决定了整个流程用什么模型来“思考”和“转化”
Settings.llm = PrivateLLM()
Settings.embed_model = CustomQwenEmbedding()

STORAGE_DIR = "stage_2_rag/storage"

def get_knowledge_index():
    """
    RAG 流程第一步 & 第二步：解析 (Parse) 与 索引 (Index)
    """
    if os.path.exists(STORAGE_DIR):
        # 如果有缓存，直接加载（跳过解析，节省时间）
        print(f"[RAG 流程] -> 加载持久化索引...")
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=STORAGE_DIR))
    else:
        # 如果没缓存：解析文档 -> 转化为向量 -> 存入索引
        print(f"[RAG 流程] -> 解析文档并构建新索引...")
        nodes = simple_semantic_parse()  # 解析 (Parse)
        index = VectorStoreIndex(nodes)  # 索引 (Index)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        return index

def run_rag_pipeline(query_str: str):
    """
    RAG 流程的核心运行环节
    """
    # [流程 A] 加载数据：获取之前构建好的知识库索引
    index = get_knowledge_index()
    
    # [流程 B] 准备插件：配置重排器，用于从候选文档中精选出最相关的
    reranker = CustomReranker()
    
    # [流程 C] 组装“一键式”查询引擎
    # 它将 索引(数据) + 重排器(筛选) + Settings.llm(大模型) 绑定在一起
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        node_postprocessors=[reranker]
    )
    
    # [流程 D] 执行生成 (这是最核心的一行代码)
    # 虽然看起来只有一句 query()，但框架在后台自动完成了三件事：
    # 1. 检索 (Retrieve)：去数据库里找 VPN 相关的文档片段
    # 2. 缝合 (Synthesize)：把找到的片段 + 你的问题，自动拼成一个超长的 Prompt
    # 3. 推理 (Generation)：【此处调用大模型】将 Prompt 发给 PrivateLLM 并获取答案
    print(f"[RAG 流程] -> 正在生成回答...\n")
    return query_engine.query(query_str)

if __name__ == "__main__":
    # 演示：查询 RAG 系统
    user_question = "研发中心新员工如何配置 VPN？"
    
    print(f"--- RAG 系统启动 ---\n问题: {user_question}")
    
    response = run_rag_pipeline(user_question)
    
    print(f"🔍 [最终结果]:\n{response}")
