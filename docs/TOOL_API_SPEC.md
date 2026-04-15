# Tool Calling 接口定义规范 (V1.0)

## 1. 概述
本系统通过定义结构化的 API 工具，赋能 Agent 具备“查库”与“写工单”的能力。大模型需根据用户的自然语言，自主识别调用顺序并补全参数。

## 2. 工具定义 (Functions Definitions)

### 工具 2.1: `get_employee_info`
**功能描述**：
通过员工姓名（String）反查员工的结构化信息。
> **关键约束**：在执行 IT 工单提交前，模型**必须**先调用此工具获取 `emp_id`，禁止虚构工号。

**参数列表 (Input Schema)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `employee_name` | String | 是 | 用户的姓名（需从对话上下文中提取） |

**返回示例 (Output Mock)**:
```json
{
  "emp_id": "OP-202604",
  "name": "张三",
  "department": "AI应用部",
  "title": "初级研发",
  "status": "active"
}
```

---

### 工具 2.2: `submit_it_ticket`
**功能描述**：
向后台 IT 管理系统正式提交工单。

**参数列表 (Input Schema)**:
| 字段名 | 类型 | 必填 | 约束 (Enum/Pattern) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `emp_id` | String | 是 | 必须是真实的内部工号 | 员工唯一 ID |
| `issue_type` | Enum | 是 | `password_reset`, `permission_request`, `env_setup` | 工单类别 |
| `priority` | Enum | 是 | `P0`, `P1`, `P2` | 响应优先级 |
| `description` | String | 是 | - | 用户具体的问题描述 |

**返回示例 (Output Mock)**:
```json
{
  "status": "success",
  "ticket_id": "TICKET-9958",
  "message": "工单已提交，等待IT处理"
}
```

## 3. 典型调用链路示例 (Trace Analysis)

### 场景：用户说“我是张三，我刚才把 VPN 密码忘了，帮我重置一下，很急！”

1.  **意图路由**：模型识别出这是一项“业务办理”需求，触发 Tool Calling。
2.  **第一阶段 (Reasoning)**：模型识别到需要 `emp_id`，但用户未提供。
3.  **第一步调用**：执行 `get_employee_info(employee_name="张三")`。
4.  **数据注入**：拿到 `emp_id: "OP-202604"`。
5.  **第二阶段 (Reasoning)**：模型发现已具备提交工单的所有必要条件。
6.  **第二步调用**：执行 `submit_it_ticket(emp_id="OP-202604", issue_type="password_reset", priority="P1", ...)`。
7.  **结果反馈**：模型基于 API 返回的 `ticket_id` 礼貌地告知用户工单已创建。
