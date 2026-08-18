# Agentic RAG

[![CI](https://github.com/sihuo-op/first_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/sihuo-op/first_rag/actions/workflows/ci.yml)

一个面向劳动法问答的 Agentic RAG 项目。系统支持多任务意图拆分、并行 RAG 工具调用、混合检索、调试信息展示，以及基于“滚动摘要 + 未压缩历史 + 长期用户记忆”的复杂会话记忆。

> 深入阅读：[架构设计与关键决策](./docs/ARCHITECTURE.md) · [量化评估报告](./docs/EVALUATION_REPORT.md)

## 核心能力

- **MainAgent 并行编排**：把复杂问题拆成多个子任务，并发调用 RAG 工具后汇总答案。
- **RAG 工具化**：每个子任务独立创建 `RAGGraph` 和 LLM 实例，避免共享状态互相污染。
- **混合检索**：Milvus dense 向量检索 + BM25 sparse 检索 + RRF 融合 + CrossEncoder 重排序。
- **迭代式 RAGGraph**：单问题检索链路支持 `retrieve -> evaluate -> rewrite/generate`。
- **长期记忆**：使用会话摘要、未压缩历史和用户级长期记忆辅助 query rewrite。
- **可视化调试**：前端展示拆分任务、检索过程、评估结果、最终生成信息。
- **线程安全加固**：对 embedding、reranker、BM25、Milvus collection cache、LLM invoke 等共享资源加锁或隔离。

## 当前架构

```text
Frontend
  -> Chat API
    -> ChatService
      -> MemoryService
        - ConversationSummary：较早历史滚动摘要
        - unsummarized messages：尚未压缩的近期原文
        - UserMemory：用户级长期记忆，Milvus 按 user_id 过滤召回
        - rewrite_query_with_memory：生成独立检索问题
      -> MainAgent
        - 判断/拆分多任务意图
        - 并行调用 RAGQATool
        - 汇总多个子任务答案
          -> RAGQATool
            -> RAGGraph
              -> RetrieveTool
              -> EvaluateTool
              -> RewriteTool / GenerateTool
                -> HybridRetriever
                  - Dense Milvus search
                  - Sparse BM25 search
                  - RRF fusion
                  - CrossEncoder rerank
```

## 长期记忆设计

本项目不是每轮都压缩摘要，而是采用类似 Claude/ChatGPT 的上下文组织方式：

```text
较早历史 -> ConversationSummary.summary
尚未压缩历史 -> 原文 messages tail
当前问题 -> query
长期用户记忆 -> UserMemory 向量召回
```

每次问答前：

1. 读取当前会话摘要。
2. 读取 `last_summarized_message_id` 之后的未压缩消息。
3. 估算 `summary + unsummarized messages + query` token 数。
4. 超过配置预算后，压缩较早的未压缩消息，保留最近 `MEMORY_RECENT_MESSAGE_LIMIT` 条原文。
5. 按 `user_id` 从长期记忆 collection 召回相关用户背景。
6. 用这些信息把当前问题改写成独立、适合检索的劳动法问题。

长期记忆只保存用户身份、偏好、目标、约束、案件事实等用户背景，不把法律条文或模型结论当作记忆。

## 项目结构

```text
backend/app/
├── api/                    # FastAPI 路由
│   ├── auth.py             # 登录、注册、当前用户
│   ├── chat.py             # 对话接口
│   ├── documents.py        # 文档上传和管理
│   └── admin.py            # 管理接口
├── agent/                  # Agent 编排层
│   ├── main_agent.py       # MainAgent：任务拆分、并行执行、答案汇总
│   └── rag_qa_tool.py      # RAGQATool：单问题 RAG 工具
├── core/                   # 配置、依赖、安全
│   ├── config.py
│   ├── dependencies.py
│   └── security.py
├── db/                     # 数据库初始化和 session
│   ├── init_db.py
│   └── session.py
├── entities/               # ORM 和 Pydantic schema
│   ├── database.py
│   └── schemas.py
├── llm/                    # LLM Provider 封装
│   └── providers.py
├── rag/                    # RAG 核心
│   ├── graph.py            # RAGGraph 状态机
│   ├── steps.py            # Retrieve/Rewrite/Generate/Evaluate tools
│   ├── retriever.py        # HybridRetriever、BM25、Reranker
│   ├── splitter.py         # 三层文档切分
│   └── vector_store.py     # Milvus 封装，含 memory collection
└── services/               # 业务服务
    ├── auth_service.py
    ├── chat_service.py
    ├── document_service.py
    ├── memory_service.py   # 会话摘要、长期记忆、memory-aware rewrite
    └── token_budget.py     # token 估算

frontend/src/
├── api/                    # Axios API 封装
├── router/                 # 路由守卫
├── store/                  # Pinia store
├── utils/                  # request/auth 工具
└── views/                  # Login、Chat、Admin 页面
```

## 快速启动

### 1. 启动 Milvus

```bash
docker-compose -f docker-compose.db.yml up -d
```

### 2. 初始化数据库

```bash
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/init_db_script.py
```

默认管理员：

```text
username: admin
password: admin123
```

### 3. 启动后端

推荐使用 `--app-dir`，避免从项目根目录误加载其他 `main` 模块：

```bash
D:/tln/code/first_rag/.venv/Scripts/python.exe -m uvicorn main:app --app-dir D:/tln/code/first_rag/backend --host 0.0.0.0 --port 8000
```

访问 API 文档：

```text
http://127.0.0.1:8000/docs
```

### 4. 启动前端

```bash
npm --prefix D:/tln/code/first_rag/frontend run dev -- --host 0.0.0.0
```

访问前端：

```text
http://localhost:5173/
```

前端开发代理使用：

```js
target: 'http://127.0.0.1:8000'
```

不要改回 `localhost`，在部分 Windows 环境下可能导致 Vite 代理登录接口返回 500。

### 5. 一键导入演示语料

```bash
D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/scripts/demo_setup.py
```

脚本会自动等待后端就绪、登录、上传内置的劳动法语料（`backend/scripts/demo_corpus/`）、等待处理完成，并打印可直接试问的示例问题。

## 开发与质量

- **CI**：GitHub Actions（lint + 后端单测含覆盖率 + KG 集成测试（testcontainers Neo4j）+ 前端构建），见 [ci.yml](.github/workflows/ci.yml)。
- **Lint**：`ruff check backend/app backend/main.py`（配置在根 `pyproject.toml`）。
- **Pre-commit**：`pre-commit install` 后自动执行 ruff 与空白符检查（配置见 [.pre-commit-config.yaml](.pre-commit-config.yaml)）。
- **测试**：`pytest tests/unit`（无外部依赖）；`pytest tests/test_knowledge_graph --ignore=tests/test_knowledge_graph/test_e2e.py`（需本机 Docker）。

## 登录接口验证

后端直连：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

前端代理：

```bash
curl -X POST http://127.0.0.1:5173/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

成功时返回：

```json
{
  "access_token": "...",
  "token_type": "bearer",
  "refresh_token": "..."
}
```

## 主要接口

### 认证

```text
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/refresh
GET  /api/v1/auth/me
```

登录请求体：

```json
{
  "username": "admin",
  "password": "admin123"
}
```

### 对话

```text
POST /api/v1/chat/conversations
GET  /api/v1/chat/conversations
GET  /api/v1/chat/conversations/{conversation_id}/messages
POST /api/v1/chat/chat
```

对话请求示例：

```json
{
  "query": "试用期被辞退能拿补偿吗？",
  "conversation_id": 1,
  "use_rag": true,
  "stream": false
}
```

响应中的 `debug_info.memory_info` 会包含：

```json
{
  "standalone_query": "改写后的独立问题",
  "summary_used": true,
  "unsummarized_message_count": 8,
  "long_term_memories": [],
  "token_budget": {
    "used": 1200,
    "budget": 22400,
    "ratio": 0.8,
    "compressed": false
  }
}
```

## 核心配置

`.env` 需提供 embedding、LLM、reranker 等配置。常用项：

```env
APP_ENV=development
APP_DEBUG=true

DATABASE_URL=sqlite:///./data/sqlite/app.db
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_PREFIX=rag_

EMBEDDING_USE_LOCAL=true
EMBEDDING_DIMENSION=1024
SENTENCE_TRANSFORMER_MODEL=BAAI/bge-m3

CHAT_API_KEY=your-api-key
CHAT_API_BASE=https://your-openai-compatible-endpoint/v1
CHAT_MODEL=your-chat-model

GENERATION_LLM_MODEL=
GENERATION_LLM_TEMPERATURE=0.1
GENERATION_LLM_MAX_TOKENS=2000
REWRITE_LLM_MODEL=
REWRITE_LLM_TEMPERATURE=0.3
REWRITE_LLM_MAX_TOKENS=500

RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_TOP_N=10

MEMORY_ENABLED=true
MEMORY_COLLECTION_NAME=memories
MEMORY_RETRIEVAL_TOP_K=5
MEMORY_RECENT_MESSAGE_LIMIT=10
MEMORY_SUMMARY_TARGET_TOKENS=1200
MEMORY_EXTRACTION_MAX_ITEMS=5
CHAT_MODEL_CONTEXT_WINDOW=128000
REWRITE_MODEL_CONTEXT_WINDOW=32000
MEMORY_CONTEXT_RATIO=0.8
MEMORY_RESERVED_OUTPUT_TOKENS=4000
MEMORY_RESERVED_RAG_TOKENS=20000
MEMORY_MAX_COMPRESS_ROUNDS=2
```

## 验证命令

编译后端：

```bash
D:/tln/code/first_rag/.venv/Scripts/python.exe -m compileall D:/tln/code/first_rag/backend/app
```

Smoke import：

```bash
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe - <<'PY'
from app.services.memory_service import MemoryService
from app.services.token_budget import TokenBudget
from app.entities.database import ConversationSummary, UserMemory
from app.services.chat_service import ChatService
print('ok')
PY
```

## 常见问题

### 登录失败

先确认后端直连是否正常：

```text
http://127.0.0.1:8000/api/v1/auth/login
```

如果后端直连正常，但前端页面登录失败，检查 `frontend/vite.config.js` 代理目标是否是：

```js
target: 'http://127.0.0.1:8000'
```

### 后端启动提示找不到 `app`

不要在项目根目录直接运行 `uvicorn main:app`，推荐：

```bash
D:/tln/code/first_rag/.venv/Scripts/python.exe -m uvicorn main:app --app-dir D:/tln/code/first_rag/backend --host 0.0.0.0 --port 8000
```

### Milvus 连接失败

确认 Docker 服务和 Milvus 已启动：

```bash
docker-compose -f docker-compose.db.yml up -d
```

### 模型下载慢

可在环境中配置 HuggingFace 镜像，例如：

```env
HF_ENDPOINT=https://hf-mirror.com
```

## 许可证

MIT License
