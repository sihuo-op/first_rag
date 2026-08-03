# Agentic RAG 改造指南

## 一、概念对比

### 1.1 传统 RAG vs Agentic RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                      传统 RAG (当前架构)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   用户问题 ──► 检索 ──► 重排 ──► 组装 Prompt ──► LLM 生成 ──► 答案  │
│                 │        │                                      │
│              固定参数   固定流程                                   │
│                                                                 │
│   特点：线性流程，无反馈循环                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Agentic RAG (目标架构)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌──────────────┐                              │
│                    │   用户问题    │                              │
│                    └──────┬───────┘                              │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│         ┌─────────│    Agent     │─────────┐                     │
│         │         │   (大脑)      │         │                     │
│         │         └──────┬───────┘         │                     │
│         │                │                 │                     │
│    ┌────▼────┐    ┌─────▼─────┐    ┌──────▼──────┐               │
│    │Query改写 │    │ 评估结果   │    │  生成答案   │               │
│    └─────┬───┘    └─────┬─────┘    └──────┬──────┘               │
│          │              │                 │                      │
│          ▼              ▼                 ▼                      │
│    ┌─────────┐   ┌────────────┐    ┌────────────┐                │
│    │ 检索工具 │   │网页搜索工具 │    │ 数据查询工具│                │
│    └─────────┘   └────────────┘    └────────────┘                │
│          │              │                 │                      │
│          └──────────────┴─────────────────┘                      │
│                         │                                        │
│                         ▼                                        │
│                  ┌──────────────┐                               │
│                  │   结果足够？   │──否──► 循环回 Agent           │
│                  └──────┬───────┘                               │
│                         │是                                     │
│                         ▼                                        │
│                    最终答案                                       │
│                                                                 │
│   特点：Agent 自主决策，可迭代，可调用多工具                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心差异

| 维度 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| **流程** | 固定线性流程 | Agent 动态决策流程 |
| **检索策略** | 单次检索，固定 top_k | 多次检索，动态调整策略 |
| **查询理解** | 直接用原始问题检索 | 先分析问题，改写/分解查询 |
| **结果评估** | 无反馈机制 | Agent 评估结果质量，决定是否重试 |
| **工具调用** | 仅检索文档库 | 可调用多种工具（检索、网页、API等） |
| **可解释性** | 黑盒 | Agent 决策过程可追溯 |

---

## 二、当前项目架构分析

### 2.1 现有组件

```
backend/app/
├── rag/
│   ├── embeddings/          # 向量化
│   ├── retriever/
│   │   ├── hybrid_retriever.py   # 混合检索【核心】
│   │   ├── sparse_retriever.py   # BM25 稀疏检索
│   │   └── reranker.py           # 重排序
│   ├── splitter/             # 文档切分
│   ├── vector_store/        # Milvus 向量库
│   └── parsers/             # 文档解析
└── services/
    └── chat_service.py      # 聊天服务【调用 RAG 的入口】
```

### 2.2 当前流程（chat_service.py）

```python
# 简化的当前流程
async def chat():
    # 1. 直接用原始问题检索
    chunks = await retriever.retrieve(query, top_k=10)

    # 2. 重排序
    reranked = await reranker.rerank(query, chunks)

    # 3. 组装 prompt，调用 LLM
    context = build_context(reranked[:5])
    answer = await llm.generate(query, context)

    # 4. 返回答案
    return answer
```

**问题**：
- 单次检索，无法处理复杂问题
- 无查询改写，检索效果依赖用户表述
- 无结果评估，无法自我纠错

---

## 三、Agentic RAG 架构设计

### 3.1 目标架构

```
backend/app/
├── rag/                        # 现有组件（保留）
├── agent/                      # 新增：Agent 模块
│   ├── __init__.py
│   ├── agent_orchestrator.py   # Agent 编排器（核心）
│   ├── tools/                  # 工具集
│   │   ├── __init__.py
│   │   ├── retrieve_tool.py     # 文档检索工具
│   │   ├── web_search_tool.py  # 网页搜索工具（可选）
│   │   └── query_rewrite_tool.py  # 查询改写工具
│   ├── prompts/                # Agent Prompt 模板
│   │   └── system_prompt.py
│   └── evaluators/             # 结果评估器
│       └── answer_evaluator.py
└── services/
    └── chat_service.py         # 修改：调用 Agent
```

### 3.2 核心组件说明

#### Agent Orchestrator（编排器）
- 负责 Agent 的整体流程控制
- 管理 Tool 的调用
- 维护对话状态和检索历史

#### Tools（工具）
- **RetrieveTool**：封装现有的 HybridRetriever
- **QueryRewriteTool**：改写查询、分解复杂问题
- **WebSearchTool**（可选）：联网搜索补充信息

#### Evaluator（评估器）
- 评估检索结果的相关性
- 评估生成答案的置信度
- 决定是否需要继续检索

---

## 四、详细改造步骤

### 4.1 步骤一：定义 Tool 接口

创建 `backend/app/agent/tools/base.py`：

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    data: Any
    message: str = ""


class BaseTool(ABC):
    """工具基类"""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    def get_schema(self) -> Dict:
        """返回工具的 JSON Schema，供 LLM 理解"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self._get_parameters_schema(),
                "required": self._get_required_parameters()
            }
        }

    def _get_parameters_schema(self) -> Dict:
        return {}

    def _get_required_parameters(self) -> List[str]:
        return []
```

### 4.2 步骤二：实现检索工具

创建 `backend/app/agent/tools/retrieve_tool.py`：

```python
from .base import BaseTool, ToolResult
from app.rag.retriever.hybrid_retriever import HybridRetriever


class RetrieveTool(BaseTool):
    """文档检索工具"""

    name = "retrieve_documents"
    description = "从文档库中检索相关内容。当用户问题需要查询已有文档时使用。"

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def _get_parameters_schema(self) -> dict:
        return {
            "query": {
                "type": "string",
                "description": "用于检索的查询语句，可以是改写后的查询"
            },
            "top_k": {
                "type": "integer",
                "description": "返回的文档数量，默认为10",
                "default": 10
            },
            "doc_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "限定检索的文档ID列表，不指定则搜索全部文档"
            }
        }

    async def execute(self, query: str, top_k: int = 10,
                      doc_ids: list[int] = None, **kwargs) -> ToolResult:
        try:
            chunks = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                doc_ids=doc_ids
            )
            return ToolResult(
                success=True,
                data=chunks,
                message=f"检索到 {len(chunks)} 个相关片段"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"检索失败: {str(e)}"
            )
```

### 4.3 步骤三：实现查询改写工具

创建 `backend/app/agent/tools/query_rewrite_tool.py`：

```python
from .base import BaseTool, ToolResult
from app.core.config import settings
import httpx


class QueryRewriteTool(BaseTool):
    """查询改写工具 - 使用 LLM 改写或分解查询"""

    name = "rewrite_query"
    description = "改写查询语句使其更适合检索，或将复杂问题分解为多个子问题"

    def _get_parameters_schema(self) -> dict:
        return {
            "original_query": {
                "type": "string",
                "description": "原始用户问题"
            },
            "rewrite_type": {
                "type": "string",
                "enum": ["improve", "decompose", "expand"],
                "description": "改写类型：improve=优化表述, decompose=分解子问题, expand=扩展关键词"
            }
        }

    async def execute(self, original_query: str,
                      rewrite_type: str = "improve", **kwargs) -> ToolResult:
        prompts = {
            "improve": f"""请将以下问题改写为更适合文档检索的查询语句。
要求：
- 保留核心意图
- 使用更准确的关键词
- 去除口语化表达

原问题：{original_query}

改写后的查询：""",

            "decompose": f"""请将以下复杂问题分解为多个简单的子问题。
要求：
- 每个子问题独立可检索
- 子问题之间有逻辑关系
- 用换行分隔每个子问题

原问题：{original_query}

分解后的子问题：""",

            "expand": f"""请扩展以下查询的关键词，生成多个相关的检索查询。
要求：
- 使用同义词
- 考虑上下位概念
- 每行一个查询

原查询：{original_query}

扩展后的查询："""
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OPENAI_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    json={
                        "model": settings.OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompts[rewrite_type]}],
                        "temperature": 0.3
                    },
                    timeout=30.0
                )

            result = response.json()["choices"][0]["message"]["content"]

            # 如果是分解类型，解析为列表
            if rewrite_type == "decompose":
                queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
                return ToolResult(success=True, data=queries)
            elif rewrite_type == "expand":
                queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
                return ToolResult(success=True, data=queries)
            else:
                return ToolResult(success=True, data=[result.strip()])

        except Exception as e:
            return ToolResult(success=False, data=None, message=str(e))
```

### 4.4 步骤四：实现答案评估器

创建 `backend/app/agent/evaluators/answer_evaluator.py`：

```python
from typing import List, Dict, Any
from pydantic import BaseModel
from app.core.config import settings
import httpx


class EvaluationResult(BaseModel):
    """评估结果"""
    is_sufficient: bool       # 答案是否充分
    confidence: float          # 置信度 0-1
    missing_info: List[str]    # 缺失的信息点
    suggestion: str            # 改进建议


class AnswerEvaluator:
    """答案质量评估器"""

    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        评估生成的答案是否充分回答了问题
        """

        context = "\n".join([
            f"[文档{i+1}]: {chunk['content'][:200]}..."
            for i, chunk in enumerate(retrieved_chunks[:5])
        ])

        prompt = f"""请评估以下答案是否充分回答了用户问题。

用户问题：{query}

检索到的文档片段：
{context}

生成的答案：{answer}

请以 JSON 格式返回评估结果：
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "missing_info": ["缺失的信息点1", "缺失的信息点2"],
    "suggestion": "如果答案不充分，建议如何改进（如改写查询、增加检索等）"
}}
"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OPENAI_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    json={
                        "model": settings.OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=30.0
                )

            import json
            result_text = response.json()["choices"][0]["message"]["content"]

            # 尝试从回答中提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                return EvaluationResult(**result)

        except Exception as e:
            pass

        # 默认返回
        return EvaluationResult(
            is_sufficient=True,
            confidence=0.5,
            missing_info=[],
            suggestion=""
        )
```

### 4.5 步骤五：实现 Agent 编排器（核心）

创建 `backend/app/agent/agent_orchestrator.py`：

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from app.agent.tools.base import BaseTool, ToolResult
from app.agent.tools.retrieve_tool import RetrieveTool
from app.agent.tools.query_rewrite_tool import QueryRewriteTool
from app.agent.evaluators.answer_evaluator import AnswerEvaluator, EvaluationResult
from app.rag.retriever.hybrid_retriever import HybridRetriever
from app.core.config import settings
import httpx


class AgentAction(Enum):
    """Agent 可执行的动作"""
    RETRIEVE = "retrieve"           # 检索文档
    REWRITE_QUERY = "rewrite_query" # 改写查询
    GENERATE = "generate"           # 生成答案
    FINISH = "finish"               # 完成


@dataclass
class AgentState:
    """Agent 运行状态"""
    original_query: str                    # 原始问题
    current_queries: List[str] = field(default_factory=list)  # 当前查询列表
    retrieved_chunks: List[Dict] = field(default_factory=list)  # 已检索的内容
    iteration: int = 0                     # 迭代次数
    max_iterations: int = 3                # 最大迭代次数
    history: List[Dict] = field(default_factory=list)  # 决策历史


class AgentOrchestrator:
    """
    Agentic RAG 编排器

    核心流程：
    1. 接收用户问题，Agent 分析是否需要改写
    2. 执行检索
    3. Agent 评估结果是否足够
    4. 若不够，改写查询重新检索
    5. 生成答案
    6. 评估答案质量，决定是否继续迭代
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.evaluator = AnswerEvaluator()

        # 初始化工具
        self.tools: Dict[str, BaseTool] = {
            "retrieve": RetrieveTool(retriever),
            "rewrite_query": QueryRewriteTool()
        }

    async def run(self, query: str, doc_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        执行 Agentic RAG 流程

        Args:
            query: 用户问题
            doc_ids: 限定的文档ID列表

        Returns:
            包含答案、检索过程、决策历史的完整结果
        """
        state = AgentState(original_query=query, current_queries=[query])

        while state.iteration < state.max_iterations:
            state.iteration += 1

            # Step 1: Agent 决定下一步动作
            action = await self._decide_action(state)

            state.history.append({
                "iteration": state.iteration,
                "action": action.value,
                "queries": state.current_queries.copy()
            })

            # Step 2: 执行动作
            if action == AgentAction.FINISH:
                break

            elif action == AgentAction.REWRITE_QUERY:
                # 改写查询
                rewrite_result = await self.tools["rewrite_query"].execute(
                    original_query=query,
                    rewrite_type="improve"
                )
                if rewrite_result.success:
                    state.current_queries = rewrite_result.data

            elif action == AgentAction.RETRIEVE:
                # 执行检索
                for q in state.current_queries:
                    result = await self.tools["retrieve"].execute(
                        query=q,
                        top_k=10,
                        doc_ids=doc_ids
                    )
                    if result.success:
                        # 合并新检索的结果（去重）
                        existing_ids = {c.get("id") for c in state.retrieved_chunks}
                        for chunk in result.data:
                            if chunk.get("id") not in existing_ids:
                                state.retrieved_chunks.append(chunk)

            elif action == AgentAction.GENERATE:
                # 生成答案
                answer = await self._generate_answer(state)

                # 评估答案
                eval_result = await self.evaluator.evaluate(
                    query=query,
                    answer=answer,
                    retrieved_chunks=state.retrieved_chunks
                )

                if eval_result.is_sufficient or state.iteration >= state.max_iterations:
                    return {
                        "answer": answer,
                        "retrieved_chunks": state.retrieved_chunks[:10],
                        "iterations": state.iteration,
                        "history": state.history,
                        "evaluation": eval_result.model_dump()
                    }

                # 答案不充分，根据建议改写查询
                if eval_result.missing_info:
                    state.current_queries = eval_result.missing_info

        # 达到最大迭代次数，返回最后生成的答案
        final_answer = await self._generate_answer(state)
        return {
            "answer": final_answer,
            "retrieved_chunks": state.retrieved_chunks[:10],
            "iterations": state.iteration,
            "history": state.history,
            "evaluation": {"is_sufficient": False, "confidence": 0.5}
        }

    async def _decide_action(self, state: AgentState) -> AgentAction:
        """
        Agent 决定下一步动作

        简化版规则：
        - 第一次迭代：先检索
        - 检索后：生成答案
        - 答案不充分：改写查询重新检索

        完整版应该让 LLM 来决策
        """
        if state.iteration == 1:
            return AgentAction.RETRIEVE
        elif not state.retrieved_chunks:
            return AgentAction.REWRITE_QUERY
        else:
            return AgentAction.GENERATE

    async def _generate_answer(self, state: AgentState) -> str:
        """根据检索结果生成答案"""

        # 限制上下文长度
        context_chunks = state.retrieved_chunks[:8]
        context = "\n\n---\n\n".join([
            f"【文档片段 {i+1}】\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"""你是一个智能助手，请根据以下文档内容回答用户问题。
如果文档中没有相关信息，请明确说明。

文档内容：
{context}

用户问题：{state.original_query}

请给出详细、准确的回答："""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={
                    "model": settings.OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=60.0
            )

        return response.json()["choices"][0]["message"]["content"]
```

### 4.6 步骤六：修改 ChatService

修改 `backend/app/services/chat_service.py`：

```python
# 在现有代码基础上添加 Agent 模式支持

from app.agent.agent_orchestrator import AgentOrchestrator


class ChatService:
    def __init__(self, ...):
        # 现有初始化代码...
        self.agent_orchestrator = None  # 延迟初始化

    async def chat_with_agent(self, query: str, doc_ids: List[int] = None) -> dict:
        """
        使用 Agentic RAG 模式对话
        """
        if self.agent_orchestrator is None:
            self.agent_orchestrator = AgentOrchestrator(
                retriever=self.retriever
            )

        result = await self.agent_orchestrator.run(query, doc_ids)

        return {
            "answer": result["answer"],
            "sources": result["retrieved_chunks"],
            "iterations": result["iterations"],
            "history": result["history"],
            "evaluation": result["evaluation"]
        }
```

### 4.7 步骤七：添加 API 端点

修改 `backend/app/api/chat.py`：

```python
@router.post("/chat/agent")
async def agent_chat(
    request: AgentChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Agentic RAG 对话接口

    与普通 chat 接口的区别：
    - 支持多轮检索迭代
    - 返回决策过程
    - 返回答案质量评估
    """
    result = await chat_service.chat_with_agent(
        query=request.query,
        doc_ids=request.doc_ids
    )
    return result
```

---

## 五、进阶优化

### 5.1 让 LLM 真正做决策

上面的简化版 `_decide_action` 使用的是硬编码规则。完整版应该让 LLM 来决策：

```python
async def _decide_action(self, state: AgentState) -> AgentAction:
    """让 LLM 来决定下一步动作"""

    # 构造工具描述
    tools_desc = "\n".join([
        f"- {name}: {tool.description}"
        for name, tool in self.tools.items()
    ])

    prompt = f"""你是一个 RAG 智能代理，请分析当前状态并决定下一步动作。

用户原始问题：{state.original_query}
当前迭代次数：{state.iteration}/{state.max_iterations}
已检索文档数：{len(state.retrieved_chunks)}

可选动作：
- RETRIEVE: 执行文档检索
- REWRITE_QUERY: 改写查询词
- GENERATE: 生成答案
- FINISH: 已经有足够信息，结束流程

请只返回动作名称（RETRIEVE/REWRITE_QUERY/GENERATE/FINISH）："""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.OPENAI_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": settings.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            },
            timeout=30.0
        )

    action_text = response.json()["choices"][0]["message"]["content"].strip().upper()

    # 映射到动作
    action_map = {
        "RETRIEVE": AgentAction.RETRIEVE,
        "REWRITE_QUERY": AgentAction.REWRITE_QUERY,
        "GENERATE": AgentAction.GENERATE,
        "FINISH": AgentAction.FINISH
    }

    return action_map.get(action_text, AgentAction.GENERATE)
```

### 5.2 支持多工具协作

可以添加更多工具：

```python
# web_search_tool.py
class WebSearchTool(BaseTool):
    """联网搜索工具"""

    name = "web_search"
    description = "当文档库中没有相关信息时，从互联网搜索"

    async def execute(self, query: str, **kwargs) -> ToolResult:
        # 调用搜索 API（如 Serper、Tavily 等）
        pass


# database_query_tool.py
class DatabaseQueryTool(BaseTool):
    """数据库查询工具"""

    name = "query_database"
    description = "查询结构化数据，如统计数据、报表等"

    async def execute(self, sql: str, **kwargs) -> ToolResult:
        # 执行 SQL 查询
        pass
```

### 5.3 使用 LangChain 简化实现

如果不想自己实现 Agent 框架，可以使用 LangChain：

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI


def create_rag_agent(retriever):
    """使用 LangChain 创建 Agent"""

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY
    )

    # 定义工具
    tools = [
        Tool(
            name="retrieve_documents",
            func=lambda q: retriever.retrieve(q),
            description="从文档库检索相关内容"
        ),
        Tool(
            name="rewrite_query",
            func=rewrite_query_func,
            description="改写查询以提高检索效果"
        )
    ]

    agent = create_openai_functions_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return agent_executor
```

---

## 六、改造优先级建议

| 优先级 | 改动内容 | 工作量 | 价值 |
|--------|---------|--------|------|
| P0 | AgentOrchestrator + RetrieveTool | 中 | 核心：实现迭代检索 |
| P1 | QueryRewriteTool | 低 | 提升：查询质量 |
| P1 | AnswerEvaluator | 中 | 提升：答案质量控制 |
| P2 | LLM 决策（替换硬编码规则） | 低 | 提升：真正的 Agent |
| P3 | 多工具支持（网页搜索等） | 中 | 扩展：能力增强 |
| P3 | LangChain 集成（替代自建） | 高 | 可选：框架替代 |

---

## 七、测试验证

改造完成后，对比测试：

```python
# 测试用例
test_queries = [
    # 简单问题：传统 RAG 可能就够了
    "什么是向量数据库？",

    # 复杂问题：需要 Agentic RAG
    "比较 Milvus 和 Pinecone 的优缺点，并给出选型建议",

    # 多跳问题：需要分解
    "文档中提到的 RAG 框架有哪些？它们各自的特点是什么？"
]
```

预期效果：
- 简单问题：两种方案差不多
- 复杂问题：Agentic RAG 答案更完整
- 多跳问题：Agentic RAG 能分解问题逐步解决
