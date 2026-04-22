import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import AVAILABLE_TOOLS

# 加载配置
load_dotenv()

def get_agent_llm():
    # 绑定工具到大模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        http_client=httpx.Client(proxy=None, timeout=60.0),
        temperature=0
    )
    return llm.bind_tools(AVAILABLE_TOOLS)

def run_action_agent(user_query: str):
    llm_with_tools = get_agent_llm()
    
    # 初始对话上下文
    messages = [
        HumanMessage(content=f"你是一个企业 IT 助手。{user_query}")
    ]
    
    print(f"\n--- Agent 执行开始 ---")
    
    # 简单的控制循环 (Loop) 模拟思考过程
    for i in range(5):  # 最多思考 5 轮，防止死循环
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 如果模型不需要调用工具，直接返回内容
        if not response.tool_calls:
            break
            
        # 如果模型需要调用工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # 查找并执行对应工具
            selected_tool = next(t for t in AVAILABLE_TOOLS if t.name == tool_name)
            tool_result = selected_tool.invoke(tool_args)
            
            # 将工具执行结果回传给模型
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
            
    print(f"🤖 助手回答：\n{response.content}")
    print(f"--- Agent 执行结束 ---\n")

if __name__ == "__main__":
    # 测试用例 1：标准完整链路 (查工号 -> 提工单)
    print(">>> 测试用例 1: 正常提交工单")
    run_action_agent("我是张三，我刚才把 VPN 密码忘了，帮我重置一下，很急，帮我设为 P1 优先级。")

    # 测试用例 2：安全拦截 (P0 级高危操作)
    print(">>> 测试用例 2: 高危拦截测试")
    run_action_agent("我是张三，帮我申请一下核心数据库的 root 读写权限，优先级设为 P0。")
