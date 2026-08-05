# OpenTelemetry 接入实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 first_rag 后端接入 OpenTelemetry 自动埋点（FastAPI/httpx/SQLAlchemy），trace 上报到本地 Jaeger 容器，在 Jaeger UI 中查看请求链路。

**Architecture:** 在 `backend/app/core/observability.py` 新建 OTel 初始化模块，通过 `setup_otel()` 配置 TracerProvider + OTLP gRPC exporter，通过 `instrument_app(app)` 注册三个 instrumentor。`main.py` 在所有 middleware 和 router 注册之后调用这两个函数，使 OTel middleware 成为最外层。Jaeger all-in-one 容器加到现有 `docker-compose.db.yml`。

**Tech Stack:** Python 3.11+ / FastAPI / OpenTelemetry Python SDK / opentelemetry-instrumentation-{fastapi,httpx,sqlalchemy} / Jaeger all-in-one 1.54 / Docker Compose

## Global Constraints

- Python >= 3.11（`pyproject.toml` 已声明）
- 启动方式：`uvicorn main:app`（不使用 `opentelemetry-instrument` CLI 前缀）
- 配置走 `.env` + pydantic-settings，与现有 `Settings` 类风格一致
- `setup_otel()` 失败不能拖垮应用（try/except + 警告打印）
- `OTEL_ENABLED=False` 时零开销（直接 return）
- `instrument_app(app)` 必须在所有 middleware 和 router 注册之后调用
- 本期不做手动 span、不做 metrics/logs、不动前端、不写集成自动化测试
- 环境：Windows + bash（命令用 Unix 语法，路径用正斜杠）

---

### Task 1: 安装 OTel 依赖并更新 pyproject.toml

**Files:**
- Modify: `pyproject.toml:5`

**Interfaces:**
- Produces: `pyproject.toml` 中声明 6 个 OTel 包，供后续 task 导入

- [ ] **Step 1: 修改 pyproject.toml 的 dependencies 字段**

把 `pyproject.toml` 第 5 行的 `dependencies = []` 改为：

```toml
dependencies = [
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-instrumentation-httpx",
    "opentelemetry-instrumentation-sqlalchemy",
]
```

- [ ] **Step 2: 安装依赖**

Run:
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-sqlalchemy
```

Expected: 6 个包及其依赖安装成功，无报错。

- [ ] **Step 3: 验证导入可用**

Run:
```bash
python -c "from opentelemetry import trace; from opentelemetry.sdk.trace import TracerProvider; from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter; from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor; from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor; from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor; print('all imports OK')"
```

Expected: 输出 `all imports OK`，无 ImportError。

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml
git commit -m "chore: add OpenTelemetry dependencies to pyproject.toml"
```

---

### Task 2: 在 Settings 中添加 OTEL_* 配置字段

**Files:**
- Modify: `backend/app/core/config.py:119`（在 `FIRST_ADMIN_EMAIL` 字段之后、`@property` 之前插入）
- Modify: `.env.example:57`（文件末尾追加）
- Create: `backend/tests/unit/test_config_otel.py`

**Interfaces:**
- Produces: `Settings` 类新增 4 个字段（`OTEL_ENABLED: bool = True`、`OTEL_SERVICE_NAME: str = "first-rag-backend"`、`OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"`、`OTEL_SAMPLING_RATE: float = 1.0`），供 Task 3 的 `setup_otel(settings)` 消费

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/__init__.py`（空文件）和 `backend/tests/unit/test_config_otel.py`：

```python
"""验证 Settings 类的 OTEL_* 字段存在且有正确默认值。

依赖项目根目录的 .env 文件存在（与现有 tests/layers/ 测试风格一致）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import get_settings


def test_otel_settings_fields_exist_with_defaults():
    """OTEL_* 字段应存在；当 .env 未设置时使用默认值。"""
    settings = get_settings()

    assert hasattr(settings, "OTEL_ENABLED")
    assert hasattr(settings, "OTEL_SERVICE_NAME")
    assert hasattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT")
    assert hasattr(settings, "OTEL_SAMPLING_RATE")

    # 默认值检查（如果 .env 中未设置 OTEL_*，则使用 Settings 类中的默认值）
    assert settings.OTEL_ENABLED is True
    assert settings.OTEL_SERVICE_NAME == "first-rag-backend"
    assert settings.OTEL_EXPORTER_OTLP_ENDPOINT == "http://localhost:4317"
    assert settings.OTEL_SAMPLING_RATE == 1.0
```

注意：此测试依赖 `.env` 文件存在且包含项目原有必填项（`EMBEDDING_USE_LOCAL` 等）。如果 `.env` 缺失，`get_settings()` 会抛 pydantic 校验错误，这是预期行为（说明项目未配置，应先解决 .env 问题）。

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd backend && python -m pytest tests/unit/test_config_otel.py -v
```

Expected: FAIL，报错类似 `AttributeError: 'Settings' object has no attribute 'OTEL_ENABLED'` 或 pydantic 校验错误。

- [ ] **Step 3: 在 Settings 类中添加 OTEL_* 字段**

在 `backend/app/core/config.py` 的 `FIRST_ADMIN_EMAIL: str = "admin@example.com"` 这行之后、`@property` 之前，插入：

```python
    # OpenTelemetry 配置
    OTEL_ENABLED: bool = True
    OTEL_SERVICE_NAME: str = "first-rag-backend"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SAMPLING_RATE: float = 1.0
```

- [ ] **Step 4: 在 .env.example 末尾追加 OTEL 配置示例**

在 `.env.example` 文件末尾追加：

```
# OpenTelemetry 配置
OTEL_ENABLED=true
OTEL_SERVICE_NAME=first-rag-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SAMPLING_RATE=1.0
```

- [ ] **Step 5: 运行测试验证通过**

Run:
```bash
cd backend && python -m pytest tests/unit/test_config_otel.py -v
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add backend/app/core/config.py .env.example backend/tests/unit/__init__.py backend/tests/unit/test_config_otel.py
git commit -m "feat(config): add OTEL_* settings fields with defaults"
```

---

### Task 3: 创建 observability.py 模块

**Files:**
- Create: `backend/app/core/observability.py`
- Create: `backend/tests/unit/test_observability.py`

**Interfaces:**
- Consumes: `Settings` 对象（来自 Task 2），需要 `OTEL_ENABLED`、`OTEL_SERVICE_NAME`、`OTEL_EXPORTER_OTLP_ENDPOINT`、`OTEL_SAMPLING_RATE` 四个字段
- Produces:
  - `setup_otel(settings) -> None`：配置 TracerProvider，失败时 print 警告不抛异常
  - `instrument_app(app: FastAPI) -> None`：注册 FastAPI/httpx/SQLAlchemy instrumentor
  - `get_tracer(name: str) -> trace.Tracer`：返回 tracer，供后期手动 span 使用

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_observability.py`：

```python
"""验证 observability.py 的基本行为（不依赖 Jaeger 运行）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_setup_otel_disabled_is_noop():
    """OTEL_ENABLED=False 时 setup_otel 不应抛异常，也不应设置 provider。"""
    from app.core import observability
    observability._initialized = False  # 重置模块状态

    settings = SimpleNamespace(
        OTEL_ENABLED=False,
        OTEL_SERVICE_NAME="test",
        OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317",
        OTEL_SAMPLING_RATE=1.0,
    )

    observability.setup_otel(settings)

    assert observability._initialized is False


def test_get_tracer_returns_tracer():
    """get_tracer 应返回非 None 的 tracer 对象。"""
    from app.core import observability
    from opentelemetry import trace

    tracer = observability.get_tracer("test")
    assert tracer is not None
    assert hasattr(tracer, "start_as_current_span")


def test_instrument_app_without_init_is_noop():
    """未初始化时 instrument_app 应直接返回，不抛异常。"""
    from app.core import observability
    observability._initialized = False

    # 传入 None 作为 app，不应抛异常（应直接 return）
    observability.instrument_app(None)
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd backend && python -m pytest tests/unit/test_observability.py -v
```

Expected: FAIL，报错 `ModuleNotFoundError: No module named 'app.core.observability'`。

- [ ] **Step 3: 创建 observability.py**

创建 `backend/app/core/observability.py`：

```python
"""OpenTelemetry 初始化与 instrumentor 注册。

初始化失败不拖垮应用：setup_otel 内部 try/except，失败时只 print 警告。
OTEL_ENABLED=False 时零开销，直接 return。
"""
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

_initialized = False


def setup_otel(settings) -> None:
    """配置 TracerProvider + OTLP gRPC exporter。失败时只 print 警告，不抛异常。"""
    global _initialized
    if _initialized:
        return
    if not settings.OTEL_ENABLED:
        print("[OTel] disabled by OTEL_ENABLED=False")
        return
    try:
        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(settings.OTEL_SAMPLING_RATE),
        )
        exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
        print(
            f"[OTel] tracing enabled: service={settings.OTEL_SERVICE_NAME}, "
            f"endpoint={settings.OTEL_EXPORTER_OTLP_ENDPOINT}, "
            f"sampling_rate={settings.OTEL_SAMPLING_RATE}"
        )
    except Exception as e:
        print(f"[OTel] init failed, tracing disabled: {e}")


def instrument_app(app: FastAPI) -> None:
    """注册 FastAPI / httpx / SQLAlchemy 自动埋点。必须在所有 middleware 和 router 注册之后调用。"""
    if not _initialized:
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        print("[OTel] instrumentors registered: fastapi, httpx, sqlalchemy")
    except Exception as e:
        print(f"[OTel] instrument_app failed: {e}")


def get_tracer(name: str):
    """获取 tracer，供手动 span 使用。"""
    return trace.get_tracer(name)
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd backend && python -m pytest tests/unit/test_observability.py -v
```

Expected: 3 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/observability.py backend/tests/unit/test_observability.py
git commit -m "feat(observability): add OTel setup module with setup_otel/instrument_app/get_tracer"
```

---

### Task 4: 在 main.py 中接入 OTel

**Files:**
- Modify: `backend/main.py:49`（在 `app.include_router(admin.router)` 之后、`@app.on_event("startup")` 之前插入）

**Interfaces:**
- Consumes: `setup_otel`、`instrument_app`（来自 Task 3）；`settings`（已在 `main.py:19` 定义）
- Produces: 后端启动时自动初始化 OTel，自动埋点 FastAPI/httpx/SQLAlchemy

- [ ] **Step 1: 在 main.py 中插入 OTel 初始化代码**

在 `backend/main.py` 的 `app.include_router(admin.router)` 这行之后、`@app.on_event("startup")` 之前，插入：

```python
# OTel 初始化（必须在所有 middleware 和 router 注册之后，使 OTel 成为最外层 middleware）
from app.core.observability import setup_otel, instrument_app
setup_otel(settings)
instrument_app(app)
```

注意：`settings` 变量已在 `main.py:19` 通过 `settings = get_settings()` 定义，直接复用。

- [ ] **Step 2: 启动后端验证无导入错误**

Run（在 backend 目录下）:
```bash
cd backend && python -c "import main; print('import OK')"
```

Expected: 输出 `[OTel] tracing enabled: service=first-rag-backend, endpoint=http://localhost:4317, sampling_rate=1.0` 和 `[OTel] instrumentors registered: fastapi, httpx, sqlalchemy` 和 `import OK`，无异常。

注意：此时 Jaeger 未启动，但 `setup_otel` 不会失败（OTLP exporter 是懒连接的，只在导出 span 时才连）。如果看到 `[OTel] init failed`，说明包没装齐或导入路径有问题。

- [ ] **Step 3: 提交**

```bash
git add backend/main.py
git commit -m "feat(main): wire OTel setup and instrumentation into FastAPI app"
```

---

### Task 5: 在 docker-compose.db.yml 中添加 Jaeger 服务

**Files:**
- Modify: `docker-compose.db.yml:54`（在 `attu` 服务块之后、`networks:` 之前插入 `jaeger` 服务）

**Interfaces:**
- Produces: `jaeger` 容器，暴露 16686（UI）、4317（OTLP gRPC）、4318（OTLP HTTP）端口

- [ ] **Step 1: 修改 docker-compose.db.yml**

在 `docker-compose.db.yml` 的 `attu:` 服务块结束之后、顶格的 `networks:` 之前，插入 `jaeger` 服务块（注意 YAML 缩进，与 `attu`/`milvus` 同级）：

```yaml
  jaeger:
    image: jaegertracing/all-in-one:1.54
    environment:
      COLLECTOR_OTLP_ENABLED: true
    ports:
      - "16686:16686"
      - "4317:4317"
      - "4318:4318"
    networks:
      - rag-network
```

完整上下文参考（修改后 `attu` 块结尾到 `networks` 之间应该是这样）：

```yaml
  attu:
    image: zilliz/attu:v2.2.8
    ports:
      - "8001:3000"
    environment:
      MILVUS_URL: milvus:19530
    depends_on:
      - milvus
    networks:
      - rag-network

  jaeger:
    image: jaegertracing/all-in-one:1.54
    environment:
      COLLECTOR_OTLP_ENABLED: true
    ports:
      - "16686:16686"
      - "4317:4317"
      - "4318:4318"
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
```

- [ ] **Step 2: 启动 Jaeger 容器**

Run:
```bash
docker compose -f docker-compose.db.yml up -d jaeger
```

Expected: 容器启动，无报错。

- [ ] **Step 3: 验证 Jaeger UI 可访问**

Run:
```bash
docker compose -f docker-compose.db.yml ps jaeger
```

Expected: `jaeger` 容器状态为 `running`（或 `Up`）。

打开浏览器访问 `http://localhost:16686`，应看到 Jaeger UI 主页（顶部有 Service 下拉框，可能为空）。

- [ ] **Step 4: 提交**

```bash
git add docker-compose.db.yml
git commit -m "infra: add Jaeger all-in-one service to docker-compose"
```

---

### Task 6: 端到端手动验证

**Files:**
- 无文件改动，纯验证步骤

**Interfaces:**
- Consumes: Task 1-5 的全部产出

**前置条件:**
- Jaeger 容器已启动（Task 5）
- 后端依赖已安装（Task 1）
- `.env` 文件存在且配置了 `CHAT_API_KEY` 等 LLM 必填项（项目原有要求）

- [ ] **Step 1: 启动后端**

在 backend 目录下执行：
```bash
cd backend && uvicorn main:app --reload
```

Expected 启动日志包含:
```
[OTel] tracing enabled: service=first-rag-backend, endpoint=http://localhost:4317, sampling_rate=1.0
[OTel] instrumentors registered: fastapi, httpx, sqlalchemy
Starting application...
Application startup complete!
```

如果出现 `[OTel] init failed` 或 `[OTel] instrument_app failed`，按 spec 5.1 节的失败排查表处理。

- [ ] **Step 2: 发起一个 chat 请求**

通过前端或 curl 发起一次问答。如果用前端：登录后发一条消息。如果用 curl：

```bash
# 先登录拿 token（替换为实际的 admin 账号）
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))")

# 发起 chat 请求
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"劳动合同终止的情形有哪些"}'
```

Expected: 后端返回包含 `answer`、`retrieved_chunks`、`debug_info` 的 JSON，无 500 错误。

注意：端口以实际 uvicorn 启动端口为准（默认 8000）。如果项目用的是其他端口，相应调整。

- [ ] **Step 3: 在 Jaeger UI 查看 trace**

打开 `http://localhost:16686`：
1. 左上角 Service 下拉框选 `first-rag-backend`
2. Operation 下拉框可留空或选 `POST /api/v1/chat`
3. 点 "Find Traces"
4. 应看到至少一条 trace

点开 trace，应看到层级结构：
```
POST /api/v1/chat                  （FastAPI span，顶层）
├── SELECT conversations ...        （SQLAlchemy span）
├── INSERT chat_messages ...        （SQLAlchemy span）
├── POST .../chat/completions       （httpx span，LLM 调用）
└── SELECT/INSERT ...               （其他 SQLAlchemy span）
```

- [ ] **Step 4: 验证 span 父子关系正确**

在 trace 详情页，确认:
- SQLAlchemy 和 httpx 子 span 挂在 FastAPI 父 span 下（有缩进）
- 每个 span 有 `duration` 显示
- FastAPI span 有 `http.method`、`http.url`、`http.status_code` 等 attribute

如果子 span 散落不挂在父 span 下，说明 context 传播有问题，需检查 instrumentor 是否注册成功（看启动日志的 `[OTel] instrumentors registered` 是否打印）。

- [ ] **Step 5: 失败排查（仅在有问题时执行）**

| 现象 | 排查 |
|---|---|
| Jaeger UI 看不到 `first-rag-backend` service | 确认 `.env` 中 `OTEL_ENABLED=true`；确认容器 4317 端口映射：`docker compose -f docker-compose.db.yml port jaeger 4317` |
| 启动日志有 `[OTel] init failed` | 看具体报错；常见：包未装齐（重新 `pip install` 6 个包）、endpoint URL 格式错误 |
| Trace 有但只有 FastAPI span | httpx/SQLAlchemy instrumentor 未注册成功；检查 `instrument_app` 是否被调用、`_initialized` 是否为 True |
| Trace 完全没有 | `OTEL_ENABLED` 是否为 False；`OTEL_SAMPLING_RATE` 是否为 0；Jaeger 容器是否 running |

- [ ] **Step 6: 验证完成后停止后端和容器**

后端: `Ctrl+C` 停止 uvicorn。
Jaeger 容器（可选，保留以便后续调试）:
```bash
docker compose -f docker-compose.db.yml stop jaeger
```

- [ ] **Step 7: 无代码改动，无需提交**

如果验证过程发现 bug 并修复了，则按修复内容单独提交。否则本 task 无 commit。
