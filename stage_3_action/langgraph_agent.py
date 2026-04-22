import os
import sqlite3
from typing import Annotated, List, Union, Dict, Any, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 兼容性导入：处理 LangGraph 0.2+ 的解耦
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    USE_SQLITE = True
except ImportError:
    try:
        # 尝试内存检查点作为回退
        from langgraph.checkpoint.memory import MemorySaver
        USE_SQLITE = False
    except ImportError:
        # 极旧版本可能在别处，或者干脆不支持
        USE_SQLITE = False

from langgraph.prebuilt import ToolNode
from stage_3_action.tools import AVAILABLE_TOOLS

# 1. 定义 Agent 状态 (State)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    approval_status: str  # "pending", "approved", "denied"
    pending_tool_call: Dict[str, Any]

# 2. 初始化模型与工具
def get_model():
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=0
    )
    return llm.bind_tools(AVAILABLE_TOOLS)

# --- 节点定义 ---

def call_model_node(state: AgentState):
    """AI 思考节点"""
    model = get_model()
    # 过滤掉空的或非消息类型的输入
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def route_after_model(state: AgentState) -> Literal["tools", "approval", "__end__"]:
    """条件路由：判断是结束、直接调工具、还是进审批流程"""
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return "__end__"
    
    # 检查是否有高危操作需要审批
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "submit_it_ticket":
            args = tool_call["args"]
            if args.get("priority") == "P0" or "root" in str(args.get("description")).lower():
                return "approval"
    
    return "tools"

def approval_node(state: AgentState):
    """审批节点"""
    last_message = state["messages"][-1]
    high_risk_call = next(tc for tc in last_message.tool_calls if tc["name"] == "submit_it_ticket")
    
    return {
        "approval_status": "pending",
        "pending_tool_call": high_risk_call
    }

# 3. 构建图 (Graph)
def create_agent_graph():
    # 根据导入情况选择检查点
    if USE_SQLITE:
        conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
        memory = SqliteSaver(conn)
    else:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model_node)
    workflow.add_node("tools", ToolNode(AVAILABLE_TOOLS))
    workflow.add_node("approval", approval_node)
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        route_after_model,
        {
            "tools": "tools",
            "approval": "approval",
            "__end__": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("approval", END)
    
    return workflow.compile(checkpointer=memory)

# 导出编译好的应用
graph_agent = create_agent_graph()
