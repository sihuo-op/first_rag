# Manual Spans Refactor Plan

**Date:** 2026-08-05
**Goal:** 为 RAG 链路添加手动 span，让 Jaeger trace 能区分 `rag.retrieve` / `rag.rewrite` / `rag.evaluate` / `rag.generate` / `agent.decompose` / `agent.merge` 等步骤，并能看到每个 LLM 调用（403/404）的归属。
**Approach:** BaseTool 模板方法 + MainAgent 方法包装

## 改动清单

### 文件 1: `backend/app/rag/steps.py`（核心改动）

**目标**：4 个 Tool 自动有 `rag.{name}` span，子类不写 span 代码。

1. 顶部新增：
   ```python
   from opentelemetry import trace
   from app.core.observability import get_tracer
   tracer = get_tracer("rag.tools")
   ```

2. `BaseTool` 改造为模板方法：
   - `execute()` 不再 abstract，改为：开 span -> 调 `_execute_impl` -> 设 attribute（`tool.name`、`tool.success`） -> 失败时设 span status ERROR -> 返回 result
   - 新增 abstract `_execute_impl(state) -> ToolResult`

3. 4 个 Tool 子类：
   - `RetrieveTool`: `execute` -> `_execute_impl`，内部用 `trace.get_current_span().set_attribute("tool.docs_found", ...)` 加属性
   - `RewriteTool`: `execute` -> `_execute_impl`，加 `tool.queries_count`、`tool.rewrite_type`
   - `GenerateTool`: `execute` -> `_execute_impl`，加 `tool.docs_used`
   - `EvaluateTool`: `execute` -> `_execute_impl`，加 `tool.grade`、`tool.confidence`

   保留各自的 try/except（返回 ToolResult 让基类读取成功/失败）。

### 文件 2: `backend/app/agent/main_agent.py`

1. 顶部新增：
   ```python
   from app.core.observability import get_tracer
   tracer = get_tracer("agent")
   ```

2. 4 个方法用 `with tracer.start_as_current_span(...)` 包裹：
   - `run_parallel` -> `agent.run_parallel`
   - `_decompose_question` -> `agent.decompose`
   - `_merge_tool_results` -> `agent.merge`
   - `_direct_answer` -> `agent.direct_answer`

### 文件 3: `backend/app/rag/graph.py`（bonus，2 行）

- `RAGGraph.run` 用 `rag.run` span 包裹整个方法体
- 不动现有 `step_timings` 计时逻辑（业务数据给前端，span 是观测数据，并行）

## 不改的

- 现有 `step_timings` / `time.time()` 计时逻辑保留
- Tool 内部的 try/except 保留（返回 ToolResult，基类读取）
- 不改测试（除非签名变化导致测试挂）
- 不动 `RAGQATool`（它委托给 `RAGGraph.run`，已有 span）
- 不动 `ChatService`（FastAPI 自动埋点已覆盖）

## 期望的 trace 结构

```
POST /api/v1/chat                          (FastAPI auto, root)
└── agent.run_parallel                     (manual)
    ├── agent.decompose                    (manual, if decomposing)
    ├── rag.run                            (manual, RAGGraph)
    │   ├── rag.retrieve                   (manual, RetrieveTool)
    │   │   └── (无 LLM 子 span)
    │   ├── rag.evaluate                   (manual, EvaluateTool)
    │   │   └── POST .../chat/completions  (httpx auto, 404)
    │   ├── rag.rewrite                    (manual, RewriteTool, if rewriting)
    │   │   └── POST .../chat/completions  (httpx auto, 403)
    │   └── rag.generate                   (manual, GenerateTool)
    │       └── POST .../chat/completions  (httpx auto, 403)
    └── agent.merge                        (manual, if multiple subtasks)
        └── POST .../chat/completions      (httpx auto, 403)
```

## 验证步骤

1. 重启后端
2. 发 chat 请求（"劳动合同终止的情形有哪些"）
3. Jaeger 看 trace，确认：
   - 有 `agent.run_parallel`、`rag.run`、`rag.retrieve`、`rag.evaluate`、`rag.generate` 等 span
   - 403/404 的 httpx span 挂在对应的 `rag.*` 或 `agent.*` 下
   - 各 span 有 attribute（`tool.name`、`tool.grade` 等）
