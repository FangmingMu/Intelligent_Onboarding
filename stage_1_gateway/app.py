import sys
import os
import streamlit as st
from dotenv import load_dotenv

# 确保能导入其他 stage 的代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_gateway.router import route_request
from stage_2_rag.indexer import build_or_load_db, rerank_documents, get_llm as get_rag_llm
from stage_3_action.agent import run_action_agent

load_dotenv()

st.set_page_config(page_title="智能入职小秘书", page_icon="🤖")

st.title("🤖 智能入职与 IT 助手")
st.markdown("---")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏：显示系统状态
with st.sidebar:
    st.header("系统状态")
    st.success("网关层: 运行中")
    st.success("RAG 引擎: 已就绪")
    st.success("Action 引擎: 已就绪")

# 展示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("有什么我可以帮您的吗？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 意图路由
        with st.status("🧠 正在思考意图...", expanded=False) as status:
            destination = route_request(prompt)
            status.update(label=f"意图识别完成: 分发至 {destination}", state="complete")
        
        response_content = ""

        # 2. 根据目的地执行不同逻辑
        if destination == "RAG":
            with st.spinner("📚 正在查阅公司手册..."):
                db = build_or_load_db()
                initial_docs = db.similarity_search(prompt, k=5)
                final_docs = rerank_documents(prompt, initial_docs)
                context = "\n\n".join([d.page_content for d in final_docs])
                
                llm = get_rag_llm()
                res = llm.invoke(f"根据以下信息回答：\n{context}\n问题：{prompt}")
                response_content = res.content

        elif destination == "ACTION":
            with st.spinner("🚀 正在调度执行引擎..."):
                # 这里简单重定向输出，实际生产建议重写 run_action_agent 返回字符串
                # 为了演示，我们在这里捕获它的逻辑
                from io import StringIO
                import sys as sys_orig
                
                # 临时捕获标准输出以展示思考过程
                old_stdout = sys_orig.stdout
                sys_orig.stdout = mystdout = StringIO()
                
                run_action_agent(prompt)
                
                sys_orig.stdout = old_stdout
                response_content = mystdout.getvalue()

        else:  # CHAT
            with st.spinner("☕ 小秘书正在与您闲聊..."):
                llm = get_rag_llm()
                res = llm.invoke(f"你是一个亲切的企业入职小秘书。请和用户进行简单的日常交流，关注用户的工作状态，并保持职业化的亲和力。用户说：{prompt}")
                response_content = res.content

        st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})
