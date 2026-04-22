import sys
import os
import time
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

# 导入 Ragas 评测相关 (采用诊断脚本中验证成功的逻辑)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# 确保能导入其他 stage 的代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_gateway.router import route_request
from stage_2_rag.indexer import build_or_load_db, rerank_documents, get_llm as get_rag_llm, get_embeddings
from stage_3_action.agent import run_action_agent
from stage_4_obs.tracer import log_interaction, update_feedback, get_performance_stats

load_dotenv()

st.set_page_config(page_title="智能入职小秘书 - 实验版", page_icon="🧪", layout="wide")

# --- 侧边栏：系统状态与后台入口 ---
with st.sidebar:
    st.title("⚙️ 系统管理")
    mode = st.radio("选择视图", ["用户对话", "监控看板", "🧪 RAG 实验室"])
    
    st.markdown("---")
    st.header("RAG 策略配置")
    rag_mode = st.selectbox("当前检索模式", ["Hybrid (混合检索)", "Vector (纯向量检索)"])
    rag_k = st.slider("初始召回数量 (k)", 5, 20, 10)
    
    st.markdown("---")
    st.header("实时状态")
    st.success(f"模式: {rag_mode}")
    st.success("后端接口: 已连接")

backend_rag_mode = "Vector" if "Vector" in rag_mode else "Hybrid"

# --- 视图 1：监控看板 ---
if mode == "监控看板":
    st.title("📊 系统运行监控看板")
    df = get_performance_stats()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总请求数", len(df))
        c2.metric("平均耗时 (ms)", f"{df['latency_ms'].mean():.2f}")
        c3.metric("总消耗 Tokens", f"{df['total_tokens'].sum():,}")
        thumbs_up = len(df[df['feedback'] == 1])
        valid_feedback = len(df[df['feedback'].notnull()])
        satisfaction = (thumbs_up / valid_feedback) * 100 if valid_feedback > 0 else 0
        c4.metric("用户点赞率", f"{satisfaction:.1f}%")
        st.markdown("### 📄 最近运行日志")
        st.dataframe(df[['timestamp', 'query', 'destination', 'latency_ms', 'total_tokens', 'feedback']].tail(10), width="stretch")
    else:
        st.info("暂无数据。")
    st.stop()

# --- 视图 2：🧪 RAG 实验室 ---
if mode == "🧪 RAG 实验室":
    st.title("🧪 RAG 实验与自动化评测看板")
    st.info("在此模式下，您可以对比检索策略并利用 Ragas 自动为系统可靠性打分。")
    
    with st.expander("📊 批量数据集自动化评测 (Ragas 四维指标)", expanded=False):
        st.write("将使用 `eval_dataset.json` 进行全量自动化打分。注意：由于需要 LLM 充当裁判，此过程较慢。")
        if st.button("🚀 启动自动化评测流程"):
            if os.path.exists("stage_2_rag/eval_dataset.json"):
                with open("stage_2_rag/eval_dataset.json", "r", encoding="utf-8") as f:
                    test_data = json.load(f)
                
                # 移除 [:5] 限制，运行全量评测
                total_questions = len(test_data)
                
                results_data = {"question":[], "answer":[], "contexts":[], "ground_truth":[]}
                display_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, item in enumerate(test_data):
                    q = item["question"]
                    gt = item["ground_truth"]
                    status_text.text(f"⏳ 正在执行检索与生成 ({idx+1}/{len(test_data)}): {q}")
                    
                    # 1. 执行混合检索模式 (作为基准对比)
                    retriever = build_or_load_db(mode="Hybrid")
                    initial_docs = retriever.invoke(q)
                    final_docs = rerank_documents(q, initial_docs)
                    
                    context_list = [d.page_content for d in final_docs]
                    res = get_rag_llm().invoke(f"根据信息回答：\n{chr(10).join(context_list)}\n问题：{q}")
                    
                    # 收集数据供 Ragas 评估
                    results_data["question"].append(q)
                    results_data["answer"].append(res.content)
                    results_data["contexts"].append(context_list)
                    results_data["ground_truth"].append(gt)
                    
                    display_results.append({"问题": q, "模型回答预览": res.content[:50] + "..."})
                    progress_bar.progress((idx + 1) / (len(test_data) * 2)) # 检索占 50% 进度

                # 2. 调用 Ragas 自动化打分
                status_text.text("⚖️ 正在调用 LLM 裁判进行四维度打分...")
                dataset = Dataset.from_dict(results_data)
                eval_llm = get_rag_llm()
                eval_embed = get_embeddings()
                
                score_result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                    llm=eval_llm,
                    embeddings=eval_embed
                )
                
                progress_bar.progress(1.0)
                status_text.success("✅ 自动化评测完成！")
                
                # 将结果转为 DataFrame
                eval_df = score_result.to_pandas()
                
                # 展示总分汇总
                st.markdown("#### 🏆 评测总得分 (Average Scores)")
                c1, c2, c3, c4 = st.columns(4)
                
                # 动态获取指标列（过滤掉非数值列如 question, answer 等）
                metric_cols = [col for col in eval_df.columns if eval_df[col].dtype in ['float64', 'int64']]
                
                # 安全显示 metric（如果列不存在则显示 N/A）
                def get_mean(col_name):
                    return f"{eval_df[col_name].mean():.3f}" if col_name in eval_df.columns else "N/A"

                c1.metric("Faithfulness", get_mean('faithfulness'))
                c2.metric("Answer Relevancy", get_mean('answer_relevancy'))
                c3.metric("Context Recall", get_mean('context_recall'))
                c4.metric("Context Precision", get_mean('context_precision'))

                # 展示明细表 (显示所有列，避开写死列名导致的 KeyError)
                st.markdown("#### 📄 详细得分明细")
                st.dataframe(eval_df, width="stretch")
            else:
                st.error("未找到 eval_dataset.json")

    st.markdown("---")
    test_query = st.text_input("单题深度路径诊断:", placeholder="输入问题查看检索片段...")
    if st.button("查看诊断轨迹"):
        if test_query:
            start_time = time.time()
            retriever = build_or_load_db(mode=backend_rag_mode)
            docs = retriever.invoke(test_query)
            final_docs = rerank_documents(test_query, docs)
            
            st.markdown(f"### 🔍 检索路径: {backend_rag_mode}")
            for idx, doc in enumerate(final_docs):
                with st.expander(f"Top {idx+1} | 来源: {doc.metadata.get('source')}"):
                    st.write(doc.page_content)
    st.stop()

# --- 视图 3：用户对话 (主逻辑) ---
st.title("🤖 智能入职助手")
if "messages" not in st.session_state: st.session_state.messages = []
if "last_request_id" not in st.session_state: st.session_state.last_request_id = None

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("有什么我可以帮您的吗？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        with get_openai_callback() as cb:
            destination = route_request(prompt, history=st.session_state.messages)
            st.caption(f"🧠 路由决策: {destination}")
            
            if destination == "RAG":
                retriever = build_or_load_db(mode=backend_rag_mode)
                docs = retriever.invoke(prompt)
                final_docs = rerank_documents(prompt, docs)
                context = "\n\n".join([d.page_content for d in final_docs])
                res = get_rag_llm().invoke(f"根据信息回答：\n{context}\n问题：{prompt}")
                response_content = res.content
            elif destination == "ACTION":
                from io import StringIO
                import sys as sys_orig
                old_stdout = sys_orig.stdout
                sys_orig.stdout = mystdout = StringIO()
                run_action_agent(prompt, history=st.session_state.messages)
                sys_orig.stdout = old_stdout
                response_content = mystdout.getvalue()
            else:
                res = get_rag_llm().invoke(f"你是亲切的小秘书。用户说：{prompt}")
                response_content = res.content

            latency = time.time() - start_time
            request_id = log_interaction(prompt, response_content, f"{destination}({backend_rag_mode})", latency, 
                                        {"total_tokens": cb.total_tokens, "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens})
            st.session_state.last_request_id = request_id
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.caption(f"⚡ {latency*1000:.0f}ms | 🪙 {cb.total_tokens} tokens")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    fb = st.feedback("thumbs", key=f"fb_{st.session_state.last_request_id}")
    if fb is not None:
        update_feedback(st.session_state.last_request_id, 1 if fb == 0 else -1)
        st.toast("感谢反馈！")
