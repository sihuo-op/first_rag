# OpenTelemetry 接入设计

**日期**：2026-08-04
**状态**：待审阅
**范围**：`backend/` 后端服务 + `docker-compose.db.yml`

## 1. 目标

为 first_rag 后端接入 OpenTelemetry（OTel）可观测性，提供请求级别的链路追踪能力。

**本期目标**：
- 自动埋点覆盖 FastAPI / httpx / SQLAlchemy 三层
- Trace 数据通过 OTLP gRPC 上报到本地 Jaeger 容器
- 在 Jaeger UI 中能查看完整请求链路及各段耗时

**本期不做**：
- 手动 span（只在 `observability.py` 预留 `get_tracer()` 接口）
- Metrics / Logs 采集
- 前端改动
- 自动化测试

## 2. 架构

### 2.1 数据流

```
浏览器请求
  -> FastAPI 自动 span（GET /api/chat/...）
    -> SQLAlchemy 自动 span（读写 sqlite）
    -> httpx 自动 span（出站调用 LLM API）
  -> OTLP gRPC 上报 localhost:4317
    -> Jaeger all-in-one 容器
      -> Jaeger UI localhost:16686 查看
```

### 2.2 组件角色

| 组件 | 角色 |
|---|---|
| OTel SDK（Python） | 采集 span，挂载到 FastAPI/httpx/SQLAlchemy |
| OTLP exporter | 把 span 通过 gRPC 推送到 4317 端口 |
| Jaeger all-in-one | 接收 + 存储（内存）+ UI 展示 |
| Jaeger UI | `localhost:16686`，按 service 名筛选 trace |

## 3. 文件改动

### 3.1 新增文件

#### `backend/app/core/observability.py`

OTel 初始化与 instrumentor 注册的统一入口。

职责：
- `setup_otel(settings)`：配置 TracerProvider + OTLP gRPC exporter + BatchSpanProcessor + TraceIdRatioBasedSampler
- `instrument_app(app)`：注册 FastAPI / httpx / SQLAlchemy 三个 instrumentor；必须在 `main.py` 中所有 middleware 和 router 注册之后调用，使 OTel middleware 成为最外层
- `get_tracer(name)`：导出 tracer，供后期手动 span 使用

关键实现要点：
- `setup_otel` 在 `instrument_app` 之前调用（先有 TracerProvider，再注册 instrumentor）
- 整个 `setup_otel` 包 try/except，失败时只 print 警告，不抛异常（埋点不能拖垮应用）
- `OTEL_ENABLED=False` 时直接 return，零开销

### 3.2 修改文件

#### `backend/app/core/config.py`

`Settings` 类新增 4 个字段：

```python
OTEL_ENABLED: bool = True
OTEL_SERVICE_NAME: str = "first-rag-backend"
OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
OTEL_SAMPLING_RATE: float = 1.0
```

走现有的 `.env` + pydantic-settings 加载机制，与项目其他配置风格一致。

#### `backend/main.py`

在 `app = FastAPI(...)` 之后、所有 `app.add_middleware(...)` / `@app.middleware(...)` / `app.include_router(...)` **全部注册完之后**，再加两行：

```python
from app.core.observability import setup_otel, instrument_app
setup_otel(settings)
instrument_app(app)
```

**调用顺序说明**：OTel FastAPI instrumentor 内部用 `app.add_middleware()` 注册，而 Starlette 的 `add_middleware` 是"后加的在外层"。把 `instrument_app(app)` 放在所有现有 middleware（CORS、`add_process_time_header`）和 router 之后，确保 OTel middleware 作为最外层，捕获完整请求生命周期。

具体位置：放在 `app.include_router(admin.router)` 这行之后、`@app.on_event("startup")` 之前。其余代码不动。`RAGGraph.run` 里现有的 `time.time()` 计时逻辑保留，与 OTel span 并行不冲突（前者是业务数据给前端展示，后者是观测数据）。

#### `docker-compose.db.yml`

在 `services:` 下新增 `jaeger` 服务：

```yaml
jaeger:
  image: jaegertracing/all-in-one:1.54
  environment:
    COLLECTOR_OTLP_ENABLED: true
  ports:
    - "16686:16686"  # UI
    - "4317:4317"    # OTLP gRPC
    - "4318:4318"    # OTLP HTTP（预留）
  networks:
    - rag-network
```

挂到现有的 `rag-network`，与其他服务同网段。内存存储，重启丢数据（开发环境可接受）。

#### `pyproject.toml`

`dependencies` 字段新增：

```
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-fastapi
opentelemetry-instrumentation-httpx
opentelemetry-instrumentation-sqlalchemy
```

不指定版本号，让 pip 自动解析最新兼容版本。

## 4. 错误处理与边界

### 4.1 初始化失败不致命

`setup_otel()` 内部 try/except，失败时 print 警告并跳过埋点。FastAPI 正常启动，业务不受影响。

### 4.2 Jaeger 不可达

`BatchSpanProcessor` 异步批量上报，Jaeger 宕时不阻塞业务请求，span 在本地队列暂存。开发环境足够。

### 4.3 已知边界（本期不修）

- **异步任务断链**：`main.py` 的 `asyncio.create_task(_load_bm25_index(...))` 和 `chat_service.py` 的 `asyncio.create_task(run_extraction())` 默认不传播 OTel context，span 会断链。这些任务不在请求主链路，影响可接受。后期做手动 span 时用 `context.attach()` 修复。
- **流式接口**：`agentic_chat_stream` 的 SSE 长连接会被 FastAPI instrumentor 包成单个 span，看不到流式 token 细节。本期靠 HTTP 级耗时足够。
- **LangChain/LangGraph 内部 HTTP 调用**：如果走 httpx 会被自动埋点抓到，走其他客户端则抓不到。本期不验证，跑通后看实际 span 情况。

### 4.4 与现有计时代码的关系

`RAGGraph.run` 的 `step_timings`、`generation_time` 等 `time.time()` 计时**保留不动**。它们是返回给前端的业务数据，OTel span 是观测数据，两套并行。后期手动 span 可复用这些计时点作为 span 边界。

## 5. 验证步骤

1. **起 Jaeger**：`docker compose -f docker-compose.db.yml up -d jaeger`，访问 `http://localhost:16686` 能看到 UI
2. **装依赖**：`pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-sqlalchemy`
3. **启动后端**：`uvicorn main:app --reload`（与平时启动方式一致），启动日志无 `[OTel] init failed`
4. **发一个请求**：登录后调用 `/api/chat/agentic` 或 `/api/chat/stream` 提问
5. **查看 Jaeger**：UI 中 Service 下拉选 `first-rag-backend`，Find Traces，应看到一条 trace，包含：
   - 1 个 FastAPI span
   - 多个 SQLAlchemy span（SELECT/INSERT chat_messages、conversations 等）
   - 1~N 个 httpx span（`POST {CHAT_API_BASE}/chat/completions`）
6. **断链检查**：展开 trace，SQLAlchemy 和 httpx 子 span 应挂在 FastAPI 父 span 下

### 5.1 失败排查

| 现象 | 排查方向 |
|---|---|
| Jaeger UI 看不到 Service | `OTEL_EXPORTER_OTLP_ENDPOINT` 是否指向 `http://localhost:4317`；容器 4317 端口是否映射 |
| 启动日志有 `[OTel] init failed` | 看具体报错，多半是包未装齐或端口不通 |
| Trace 有但只有 FastAPI span | httpx/SQLAlchemy instrumentor 未注册成功，检查 `instrument_app` 调用 |
| Trace 完全没有 | `OTEL_ENABLED` 是否为 False；采样率是否为 0 |

## 6. 后续扩展接口

`observability.py` 导出 `get_tracer(name)`，供后期手动 span 使用。用法示例（**本期不实现**）：

```python
from app.core.observability import get_tracer
tracer = get_tracer("rag.graph")

with tracer.start_as_current_span("rag.run") as span:
    span.set_attribute("rag.question", question)
    with tracer.start_as_current_span("rag.retrieve"):
        span.set_attribute("rag.docs_found", len(docs))
```

候选 span 命名（后期参考）：
- `rag.run`（顶层）
- `rag.retrieve` / `rag.retrieve.dense` / `rag.retrieve.sparse` / `rag.retrieve.rerank`
- `rag.rewrite` / `rag.evaluate` / `rag.generate`
- `agent.run_parallel`

## 7. 范围边界

本期明确不做：
- 手动 span（仅预留 `get_tracer` 接口）
- Metrics 采集（MeterProvider 配置，YAGNI）
- Logs 采集（OTel logs 仍处实验阶段）
- 前端改动
- `docker-compose.db.yml` 之外的部署文件修改
- 自动化测试（OTel 集成测试在开发环境手测即可）
