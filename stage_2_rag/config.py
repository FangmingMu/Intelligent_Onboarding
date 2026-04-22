import os
import httpx
from typing import Any, List, Optional
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from openai import OpenAI as OpenAIClient

# --- 底层组件：自定义 LLM (适配私有 API) ---
class PrivateLLM(CustomLLM):
    context_window: int = 32768
    num_output: int = 4096
    # 增加默认值，防止环境变量缺失导致 Pydantic 报错
    model_name: str = os.getenv("LLM_MODEL") or "gpt-3.5-turbo"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window, 
            num_output=self.num_output, 
            model_name=self.model_name or "unknown-model"
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        allowed_params = {"temperature", "top_p", "max_tokens", "stream"}
        api_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            http_client=httpx.Client(proxy=None)
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **api_kwargs
        )
        return CompletionResponse(text=response.choices[0].message.content)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("暂不支持流式")

# --- 底层组件：自定义向量模型 (直连 API) ---
class CustomQwenEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]: return self._get_embeddings([query])[0]
    def _get_text_embedding(self, text: str) -> List[float]: return self._get_embeddings([text])[0]
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {os.getenv('QWEN_EMBEDDING_API_KEY')}"}
        payload = {"model": os.getenv("QWEN_EMBEDDING_MODEL_NAME"), "input": texts}
        with httpx.Client(proxy=None, timeout=60.0) as client:
            response = client.post(f"{os.getenv('QWEN_EMBEDDING_API_FULL_URL')}/embeddings", headers=headers, json=payload)
            response.raise_for_status()
            return [d["embedding"] for d in response.json()["data"]]

    async def _aget_query_embedding(self, query: str) -> List[float]: return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]: return self._get_text_embedding(text)

# --- 底层组件：自定义重排器 (提升准确率) ---
class CustomReranker(BaseNodePostprocessor):
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not query_bundle or not nodes: return nodes
        headers = {"Authorization": f"Bearer {os.getenv('RERANK_API_KEY')}"}
        payload = {"model": os.getenv("RERANK_MODEL"), "query": query_bundle.query_str, "documents": [n.get_content() for n in nodes]}
        with httpx.Client(proxy=None, timeout=60.0) as client:
            response = client.post(os.getenv("RERANK_API_URL"), headers=headers, json=payload)
            results = response.json()["results"]
        new_nodes = []
        for res in results:
            idx = res.get("index")
            if idx is not None:
                orig_node = nodes[idx]
                orig_node.score = res.get("relevance_score") or res.get("score")
                new_nodes.append(orig_node)
        return sorted(new_nodes, key=lambda x: x.score or 0.0, reverse=True)

# --- 混合检索工具 ---
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except ImportError:
    # 降级处理：如果没有装 BM25，我们将使用简单的关键词检索
    print("[警告] -> llama-index-retrievers-bm25 未正确安装，请检查依赖环境。")
    BM25Retriever = None

def get_bm25_retriever(nodes, similarity_top_k=5):
    if BM25Retriever:
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)
    return None
