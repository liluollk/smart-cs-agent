# SmartCS — 智能客服多 Agent 系统

基于 LangGraph 编排的多 Agent 客服系统，具备知识库检索、意图路由、合规检查、工单处理等能力。

## 架构概览

```
用户输入 → 意图路由 → 合规前置检查 → Agent 并行调度 → 合规后置检查 → 结果合成
```

- **意图路由**：关键词预匹配 + LLM 两级分类
- **合规检查**：前后双层过滤，覆盖敏感词、注入攻击、PII 脱敏
- **Agent 调度**：LangGraph 状态图编排，支持知识检索 / 工单处理等子 Agent 并行执行
- **知识库**：PostgreSQL + pgvector 向量检索，支持语义搜索与文档管理

## 技术栈

- **语言**：Python 3.12+
- **框架**：FastAPI / LangGraph / FastMCP SDK
- **数据库**：PostgreSQL + pgvector / Redis
- **LLM**：百炼平台（qwen-plus / text-embedding-v4）

## 快速开始

```bash
# 启动基础服务
docker compose up -d postgres redis

# 安装依赖
cd backend && pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，设置 DASHSCOPE_API_KEY

# 启动
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/static/chat.html` 进入聊天界面。

## 主要功能

- **多轮对话**：支持上下文记忆，自动维护会话状态
- **知识库问答**：基于 RAG 的企业知识检索
- **工单处理**：创建工单、查询订单、发送通知
- **合规风控**：输入输出双层检测，覆盖安全场景
- **文件上传**：支持 PDF / DOCX / MD / TXT 文档入库
- **链路追踪**：全流程调用链可视化

## 项目结构

```
├── backend/          # Python 服务端
│   ├── agents/       # Agent 实现
│   ├── api/          # API 路由
│   ├── core/         # 核心逻辑（路由、合规、注册）
│   ├── knowledge/    # 知识库
│   ├── mcps/         # MCP 协议层
│   ├── memory/       # 记忆系统
│   ├── tools/        # 工具层
│   ├── tracing/      # 链路追踪
│   └── static/       # 前端界面
├── docs/             # 设计文档
└── docker-compose.yml
```

## License

MIT