import os
import httpx
from dotenv import load_dotenv
from typing import Any, List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from openai import OpenAI as OpenAIClient
from doc_parser import simple_semantic_parse

# 1. 加载环境变量
load_dotenv()

# --- 架构师注：终极方案 - 自定义 LLM 类，完全模仿您的原生调用方式，绕过模型名校验 ---
class PrivateLLM(CustomLLM):
    context_window: int = 32768
    num_output: int = 4096
    model_name: str = os.getenv("LLM_MODEL")

    @property
    def metadata(self) -> LLMMetadata:
        """获取 LLM 元数据。"""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # --- 架构师注：防御性编程，只允许 OpenAI 官方支持的参数通过 ---
        # 常见 OpenAI 参数白名单
        allowed_params = {
            "temperature", "top_p", "n", "stream", "stop", "max_tokens", 
            "presence_penalty", "frequency_penalty", "logit_bias", "user",
            "response_format", "seed", "tools", "tool_choice"
        }
        api_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        
        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            http_client=httpx.Client(proxy=None)
        )
        
        # 显式构造请求，避免 kwargs 冲突
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **api_kwargs
        )
        return CompletionResponse(text=response.choices[0].message.content)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("暂不支持流式输出")

# --- 架构师注：完全手写的 Embedding 实现 ---
class CustomQwenEmbedding(BaseEmbedding):
    """
    使用 httpx (proxy=None) 实现 Qwen3-Embedding，避开内置校验。
    """
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {os.getenv('QWEN_EMBEDDING_API_KEY')}"}
        payload = {"model": os.getenv("QWEN_EMBEDDING_MODEL_NAME"), "input": texts}
        with httpx.Client(proxy=None, timeout=60.0) as client:
            response = client.post(f"{os.getenv('QWEN_EMBEDDING_API_FULL_URL')}/embeddings", 
                                 headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return [d["embedding"] for d in data["data"]]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# --- 架构师注：Reranker 实现 ---
class CustomReranker(BaseNodePostprocessor):
    """
    禁用代理的 Reranker。
    """
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not query_bundle or not nodes: return nodes
        texts = [node.get_content() for node in nodes]
        headers = {"Authorization": f"Bearer {os.getenv('RERANK_API_KEY')}"}
        payload = {
            "model": os.getenv("RERANK_MODEL"), 
            "query": query_bundle.query_str, 
            "documents": texts
        }
        
        with httpx.Client(proxy=None, timeout=60.0) as client:
            response = client.post(os.getenv("RERANK_API_URL"), headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()["results"]
            
        new_nodes = []
        for res in results:
            idx = res.get("index")
            if idx is not None:
                orig_node = nodes[idx]
                orig_node.score = res.get("relevance_score") or res.get("score")
                new_nodes.append(orig_node)
        return sorted(new_nodes, key=lambda x: x.score or 0.0, reverse=True)

# 2. 全局配置 (Global Settings)
Settings.llm = PrivateLLM()
Settings.embed_model = CustomQwenEmbedding()

STORAGE_DIR = "stage_2_rag/storage"

def build_or_load_index():
    if os.path.exists(STORAGE_DIR):
        print(f"[Indexer] 正在加载本地索引缓存...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print(f"[Indexer] 正在解析文档并构建新索引...")
        nodes = simple_semantic_parse()
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"✅ 索引已持久化至 {STORAGE_DIR}")
    return index

if __name__ == "__main__":
    idx = build_or_load_index()
    reranker = CustomReranker()
    
    query_engine = idx.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[reranker]
    )
    
    response = query_engine.query("研发中心新员工如何配置 VPN？请根据文档详细回答步骤。")
    print(f"\n🔍 [RAG 检索+重排] 测试结果:\n{response}")
