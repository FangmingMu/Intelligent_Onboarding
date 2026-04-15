# 3. 动作执行与安全层 (Action & Harness Layer)

## 职责描述
本层级负责将大模型的逻辑决策转化为具体的系统执行动作：
- **工具调用 (Tool Calling)**：封装企业级 API（如工单系统、LDAP 等）。
- **安全防线 (Harness)**：结构化强制校验与人机协同。