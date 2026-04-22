import os
import httpx
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- 1. 定义路由结果模型 ---
class RouteResponse(BaseModel):
    """识别用户意图并分发到对应路径"""
    destination: Literal["RAG", "ACTION", "CHAT"] = Field(
        description="分发目的地。RAG: 知识查询；ACTION: 办理业务/提工单；CHAT: 闲聊或工作状态询问"
    )
    reason: str = Field(description="选择该路径的理由")

def get_router():
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        http_client=httpx.Client(proxy=None),
        temperature=0
    )
    # 使用 LangChain 的结构化输出功能
    return llm.with_structured_output(RouteResponse)

def route_request(query: str, history: list = None) -> str:
    router = get_router()
    
    # 构建包含上下文的 Prompt
    context_str = ""
    if history:
        # 只取最近 3 轮防止干扰
        recent_history = history[-3:]
        context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个企业助手的网关路由。请根据当前输入及历史对话判断意图。
        历史上下文：
        {context_str}
        
        判断规则：
        - 如果用户在回答之前关于 IT/业务的问题，或者提供办理业务所需的个人信息，选择 ACTION。
        - 如果用户询问关于公司政策、IT环境配置等知识，选择 RAG。
        - 如果用户单纯闲聊或询问工作状态，选择 CHAT。"""),
        ("human", "{query}")
    ])
    
    chain = prompt | router
    result = chain.invoke({"query": query})
    return result.destination

if __name__ == "__main__":
    # 测试路由
    test_queries = [
        "VPN 怎么配置？",
        "我是张三，帮我提一个工单重置密码。",
        "你今天心情怎么样？"
    ]
    for q in test_queries:
        dest = route_request(q)
        print(f"Query: {q} -> Destination: {dest}")
