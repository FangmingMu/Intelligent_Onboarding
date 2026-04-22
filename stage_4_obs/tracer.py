import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

LOG_FILE = "stage_4_obs/logs.jsonl"

def init_log_file():
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            pass

def log_interaction(
    query: str, 
    response: str, 
    destination: str, 
    latency: float, 
    tokens: Dict[str, int]
) -> str:
    """
    记录一次交互日志，返回唯一的 request_id 用于后续反馈更新。
    """
    init_log_file()
    request_id = str(uuid.uuid4())
    
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "destination": destination,
        "latency_ms": round(latency * 1000, 2),
        "total_tokens": tokens.get("total_tokens", 0),
        "prompt_tokens": tokens.get("prompt_tokens", 0),
        "completion_tokens": tokens.get("completion_tokens", 0),
        "feedback": None  # 默认为空，等待用户评价
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    return request_id

def update_feedback(request_id: str, score: int):
    """
    更新指定 request_id 的用户反馈评分。
    score: 1 表示赞，-1 表示踩 (或其他约定)
    """
    if not os.path.exists(LOG_FILE):
        return

    updated_lines = []
    found = False
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("request_id") == request_id:
                entry["feedback"] = score
                found = True
            updated_lines.append(json.dumps(entry, ensure_ascii=False))
    
    if found:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(updated_lines) + "\n")

def get_performance_stats():
    """
    为看板读取日志并返回分析指标。
    """
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return None
    
    df = pd.read_json(LOG_FILE, lines=True)
    return df
