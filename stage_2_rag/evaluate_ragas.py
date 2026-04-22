import os
import json
import httpx
import pandas as pd
import warnings
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# 忽略 DeprecationWarning 警告，让输出更干净
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 导入我们的 RAG 引擎组件
from indexer import build_or_load_db, rerank_documents, get_llm

load_dotenv()

def init_ragas_models():
    eval_llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        http_client=httpx.Client(proxy=None),
        temperature=0
    )
    eval_embeddings = OpenAIEmbeddings(
        model=os.getenv("QWEN_EMBEDDING_MODEL_NAME"),
        openai_api_key=os.getenv("QWEN_EMBEDDING_API_KEY"),
        openai_api_base=os.getenv("QWEN_EMBEDDING_API_FULL_URL"),
        http_client=httpx.Client(proxy=None)
    )
    return eval_llm, eval_embeddings

def run_evaluation():
    print("🚀 开始 Ragas 深度诊断评估 (已优化检索策略)...")
    
    with open("stage_2_rag/eval_dataset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    test_data = test_data[:2] 
    
    db = build_or_load_db()
    llm = get_llm()
    eval_llm, eval_embeddings = init_ragas_models() 
    
    questions, answers, contexts, ground_truths = [], [], [], []
    
    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        # --- 策略优化：从 k=10 扩大到 k=15，并进行重排 ---
        initial_docs = db.similarity_search(q, k=15)
        # 如果重排器太严格，我们可以暂时调低阈值或增加重排后的保留数量
        final_docs = rerank_documents(q, initial_docs, top_n=5)
        
        context_list = [d.page_content for d in final_docs]
        
        print(f"\n[诊断] 问题 {i+1}: {q}")
        print(f"[诊断] 核心 Context 预览: {context_list[0][:150]}...")

        context_str = "\n\n".join(context_list)
        prompt = f"请严格根据以下已知信息回答问题。如果信息中没有提到，请说不知道。\n\n已知信息：\n{context_str}\n\n问题：{q}"
        response = llm.invoke(prompt).content
        
        questions.append(q)
        answers.append(response)
        contexts.append(context_list) 
        ground_truths.append(gt)
        
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print("\n📊 正在进行 Ragas 评分 (LLM 裁判评分中)...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n✅ 评估完成！汇总得分：")
    print(result)

if __name__ == "__main__":
    run_evaluation()
