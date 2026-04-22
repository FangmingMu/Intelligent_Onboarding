import sys
import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

# 确保能导入其他 stage 的代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_gateway.router import route_request
from stage_2_rag.indexer import build_or_load_db, rerank_documents, get_llm as get_rag_llm
from stage_3_action.agent import run_action_agent
from stage_4_obs.tracer import log_interaction, update_feedback, get_performance_stats

load_dotenv()

st.set_page_config(page_title="智能入职小秘书", page_icon="🤖", layout="wide")

# --- 侧边栏：系统状态与后台入口 ---
with st.sidebar:
    st.title("⚙️ 系统管理")
    mode = st.radio("选择视图", ["用户对话", "监控看板"])
    
    st.markdown("---")
    st.header("实时状态")
    st.success("网关层: 运行中")
    st.success("RAG 引擎: 已就绪")
    st.success("Action 引擎: 已就绪")

# --- 视图 1：监控看板 ---
if mode == "监控看板":
    st.title("📊 系统运行监控看板")
    df = get_performance_stats()
    
    if df is not None:
        # 指标概览
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总请求数", len(df))
        c2.metric("平均耗时 (ms)", f"{df['latency_ms'].mean():.2f}")
        c3.metric("总消耗 Tokens", f"{df['total_tokens'].sum():,}")
        
        # 计算满意度
        thumbs_up = len(df[df['feedback'] == 1])
        valid_feedback = len(df[df['feedback'].notnull()])
        satisfaction = (thumbs_up / valid_feedback) * 100 if valid_feedback > 0 else 0
        c4.metric("用户点赞率", f"{satisfaction:.1f}%")

        # 图表分析
        st.markdown("### 📈 流量与性能趋势")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("意图分发占比")
            dest_counts = df['destination'].value_counts()
            st.bar_chart(dest_counts)
            
        with chart_col2:
            st.write("响应延迟趋势 (ms)")
            st.line_chart(df['latency_ms'])

        st.markdown("### 📄 最近运行日志")
        st.dataframe(df[['timestamp', 'query', 'destination', 'latency_ms', 'total_tokens', 'feedback']].tail(10))
    else:
        st.info("暂称：暂无运行数据，请先开始对话。")
    st.stop()

# --- 视图 2：用户对话 (原逻辑) ---
st.title("🤖 智能入职与 IT 助手")
st.caption("基于 RAG (LlamaIndex) 与 Tool Calling (LangChain) 的企业级 Agent 架构")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None

# 展示对话历史
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("有什么我可以帮您的吗？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        
        with get_openai_callback() as cb:
            # 1. 意图路由
            with st.status("🧠 正在思考意图...", expanded=False) as status:
                destination = route_request(prompt, history=st.session_state.messages)
                status.update(label=f"意图识别完成: 分发至 {destination}", state="complete")
            
            response_content = ""

            # 2. 根据目的地执行不同逻辑
            if destination == "RAG":
                with st.spinner("📚 正在查阅公司手册..."):
                    # 现在 build_or_load_db() 返回的是一个混合检索器 (EnsembleRetriever)
                    retriever = build_or_load_db()
                    initial_docs = retriever.invoke(prompt)
                    final_docs = rerank_documents(prompt, initial_docs)
                    context = "\n\n".join([d.page_content for d in final_docs])
                    
                    llm = get_rag_llm()
                    res = llm.invoke(f"根据以下信息回答：\n{context}\n问题：{prompt}")
                    response_content = res.content

            elif destination == "ACTION":
                with st.spinner("🚀 正在调度执行引擎..."):
                    from io import StringIO
                    import sys as sys_orig
                    old_stdout = sys_orig.stdout
                    sys_orig.stdout = mystdout = StringIO()
                    run_action_agent(prompt, history=st.session_state.messages)
                    sys_orig.stdout = old_stdout
                    response_content = mystdout.getvalue()

            else:  # CHAT
                with st.spinner("☕ 小秘书正在与您闲聊..."):
                    llm = get_rag_llm()
                    res = llm.invoke(f"你是一个亲切的企业入职小秘书。请和用户进行简单的日常交流，关注用户的工作状态，并保持职业化的亲和力。用户说：{prompt}")
                    response_content = res.content

            latency = time.time() - start_time
            
            # 3. 记录日志 (Tracing)
            token_stats = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }
            request_id = log_interaction(prompt, response_content, destination, latency, token_stats)
            st.session_state.last_request_id = request_id

            # 展示结果
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            # 显示性能指标
            st.caption(f"⚡ 耗时: {latency*1000:.0f}ms | 🪙 Tokens: {cb.total_tokens}")

# 评价组件 (放在最后)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    feedback = st.feedback("thumbs", key=f"fb_{st.session_state.last_request_id}")
    if feedback is not None:
        score = 1 if feedback == 0 else -1 # streamlit thumbs returns 0 for up, 1 for down
        update_feedback(st.session_state.last_request_id, score)
        st.toast("感谢您的反馈！数据已记录至监控系统。")
