import os
import random
import time
from typing import Dict, Union, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# --- 1. 定义工具参数的 Pydantic 模型 (实现 JSON 强制化) ---

class EmployeeInfoInput(BaseModel):
    employee_name: str = Field(description="用户的姓名，需从对话上下文中提取")

class TicketInput(BaseModel):
    emp_id: str = Field(description="员工唯一 ID，必须通过 get_employee_info 获取")
    issue_type: str = Field(
        description="工单类别: password_reset, permission_request, env_setup"
    )
    priority: str = Field(description="优先级: P0, P1, P2")
    description: str = Field(description="用户具体的问题描述")

# --- 2. 模拟后端接口 (含异常模拟) ---

def mock_get_employee_api(name: str) -> Dict:
    """模拟查询员工数据库"""
    db = {
        "张三": {"emp_id": "OP-202604", "name": "张三", "department": "AI应用部", "status": "active"},
        "李四": {"emp_id": "OP-202605", "name": "李四", "department": "研发中心", "status": "active"}
    }
    return db.get(name, {"error": "员工不存在"})

def mock_submit_ticket_api(data: Dict) -> Dict:
    """模拟提交工单，包含随机 500 错误测试重试机制"""
    # 模拟 20% 的概率出现服务器错误
    if random.random() < 0.2:
        raise Exception("Internal Server Error (503)")
    
    return {
        "status": "success",
        "ticket_id": f"TICKET-{random.randint(1000, 9999)}",
        "message": "工单已提交，等待IT处理"
    }

# --- 3. 封装为 LangChain Tool ---

@tool("get_employee_info", args_schema=EmployeeInfoInput)
def get_employee_info(employee_name: str) -> str:
    """通过姓名获取员工 ID。在提交工单前必须调用此工具。"""
    print(f"🔍 [API 调用] 正在查询员工信息: {employee_name}...")
    result = mock_get_employee_api(employee_name)
    return str(result)

@tool("submit_it_ticket", args_schema=TicketInput)
def submit_it_ticket(emp_id: str, issue_type: str, priority: str, description: str) -> str:
    """向 IT 系统提交工单。"""
    # --- 安全层：人机交互拦截 (Human-in-the-loop) ---
    if priority == "P0" or "root" in description.lower():
        return "ERROR: [SAFETY_INTERCEPT] 检测到高风险请求（P0级或涉及Root权限），需人工审批。请告知用户：'由于涉及高危操作，已为您挂起，请等待管理员二次确认'。"

    print(f"🚀 [API 调用] 正在提交工单: {issue_type} | 优先级: {priority}...")
    
    # --- 调度层：重试机制 (Retry Logic) ---
    max_retries = 3
    for i in range(max_retries):
        try:
            result = mock_submit_ticket_api({
                "emp_id": emp_id,
                "issue_type": issue_type,
                "priority": priority,
                "description": description
            })
            return str(result)
        except Exception as e:
            if i < max_retries - 1:
                print(f"⚠️ [Retry] API 响应失败，正在进行第 {i+1} 次重试...")
                time.sleep(2)
            else:
                return f"ERROR: API 调用在 {max_retries} 次尝试后依然失败: {str(e)}"

# 工具列表导出
AVAILABLE_TOOLS = [get_employee_info, submit_it_ticket]
