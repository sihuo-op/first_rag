from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import SessionLocal
from app.entities.database import Conversation, ChatMessage, MessageRole
from app.entities.schemas import ConversationCreate, ConversationUpdate
from app.rag.retriever import HybridRetriever
from app.core.config import get_settings
from app.agent.main_agent import MainAgent
from app.llm.providers import get_generation_llm, get_rewrite_llm
from app.services.memory_service import MemoryService
import httpx
import json
import time

settings = get_settings()


class ChatService:
    """
    RAG 对话服务 - 核心问答编排器

    职责：
    - 管理用户会话（创建、查询、更新、删除）
    - 编排完整的 RAG 检索增强生成流程
    - 保存对话历史和检索结果用于追溯

    RAG 流程：
    1. 接收用户问题
    2. LLM 改写查询（扩展同义词、结合上下文补全语义）
    3. 调用 HybridRetriever 检索相关文档片段
    4. 构建包含检索上下文的 Prompt
    5. 调用 LLM 生成回答
    6. 保存问答记录到数据库

    依赖：
    - HybridRetriever: 负责多粒度向量检索和重排序
    - LLM API: 用于查询改写和答案生成（OpenAI 兼容接口，统一配置）
    - SQLite: 存储会话和消息记录
    """

    def __init__(self, db: Session, retriever: HybridRetriever):
        """
        初始化对话服务

        Args:
            db: 数据库会话
            retriever: 混合检索器实例
        """
        self.db = db
        self.retriever = retriever
        self._async_client = httpx.AsyncClient(timeout=120.0)

    def get_conversation_by_id(self, conv_id: int, user_id: Optional[int] = None) -> Optional[Conversation]:
        """
        根据ID获取会话

        Args:
            conv_id: 会话ID
            user_id: 用户ID（可选，用于权限验证）

        Returns:
            会话对象，不存在则返回None
        """
        query = self.db.query(Conversation).filter(Conversation.id == conv_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        return query.first()

    def get_conversations(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """
        获取用户的会话列表

        Args:
            user_id: 用户ID
            skip: 跳过的记录数
            limit: 返回的最大记录数

        Returns:
            会话列表，按最后消息时间倒序排列（最新的在前）
        """
        # 使用子查询获取每个会话的最后消息时间
        subquery = self.db.query(
            ChatMessage.conversation_id,
            func.max(ChatMessage.created_at).label('last_time')
        ).group_by(ChatMessage.conversation_id).subquery()

        # 关联查询并按时间降序
        return self.db.query(Conversation).outerjoin(
            subquery, Conversation.id == subquery.c.conversation_id
        ).filter(
            Conversation.user_id == user_id
        ).order_by(
            subquery.c.last_time.desc().nullslast(),
            Conversation.created_at.desc()
        ).offset(skip).limit(limit).all()

    def create_conversation(self, user_id: int, data: ConversationCreate) -> Conversation:
        """
        创建新会话

        Args:
            user_id: 用户ID
            data: 会话创建数据

        Returns:
            创建的会话对象
        """
        conversation = Conversation(
            user_id=user_id,
            title=data.title or "New Conversation"
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_conversation(
        self,
        conv_id: int,
        data: ConversationUpdate,
        user_id: Optional[int] = None
    ) -> Optional[Conversation]:
        """
        更新会话信息

        Args:
            conv_id: 会话ID
            data: 更新数据
            user_id: 用户ID（可选，用于权限验证）

        Returns:
            更新后的会话对象
        """
        conversation = self.get_conversation_by_id(conv_id, user_id)
        if not conversation:
            return None

        if data.title:
            conversation.title = data.title

        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def delete_conversation(self, conv_id: int, user_id: Optional[int] = None) -> bool:
        """
        删除会话

        Args:
            conv_id: 会话ID
            user_id: 用户ID（可选，用于权限验证）

        Returns:
            是否删除成功
        """
        conversation = self.get_conversation_by_id(conv_id, user_id)
        if not conversation:
            return False

        self.db.delete(conversation)
        self.db.commit()
        return True

    def get_messages(self, conv_id: int, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[ChatMessage]:
        """
        获取会话的消息列表

        Args:
            conv_id: 会话ID
            user_id: 用户ID（可选，用于权限验证）
            skip: 跳过的记录数
            limit: 返回的最大记录数

        Returns:
            消息列表，按时间正序排列
        """
        conversation = self.get_conversation_by_id(conv_id, user_id)
        if not conversation:
            return []
        return self.db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conv_id
        ).order_by(ChatMessage.created_at.asc()).offset(skip).limit(limit).all()

    async def agentic_chat(
        self,
        user_id: int,
        query: str,
        conv_id: Optional[int] = None,
        max_attempts: int = 2,
        background_tasks=None
    ) -> Dict[str, Any]:
        """
        Agentic RAG 对话（ReAct Commander 模式）

        使用 ReAct Agent 作为顶层指挥，决定是否调用检索工具
        将 RAGGraph 封装为工具，由 ReAct Agent 决定何时调用

        Args:
            user_id: 用户ID
            query: 用户问题
            conv_id: 会话ID（可选）
            max_attempts: RAGGraph 最大迭代次数
            background_tasks: FastAPI BackgroundTasks，用于 fire-and-forget 写 chunk 统计

        Returns:
            包含回答和执行过程的字典
        """
        start_time = time.time()

        # 创建或获取会话
        if conv_id is None:
            conversation = self.create_conversation(user_id, ConversationCreate(title=query[:50]))
            conv_id = conversation.id
        else:
            conversation = self.get_conversation_by_id(conv_id, user_id)
            if not conversation:
                raise ValueError("Conversation not found")

        # 保存用户消息
        user_message = ChatMessage(
            conversation_id=conv_id,
            role=MessageRole.USER,
            content=query
        )
        self.db.add(user_message)
        self.db.commit()

        result = {}
        memory_service = None
        memory_context = {"summary": "", "unsummarized_messages": [], "long_term_memories": [], "token_budget": {}}
        standalone_query = query
        created_memories = []
        # 运行 MainAgent 编排器
        try:
            memory_service = MemoryService(self.db, self.retriever.vector_store)
            memory_context = memory_service.prepare_memory_context(user_id, conv_id, query)
            standalone_query = memory_service.rewrite_query_with_memory(query, memory_context)

            # 获取 LLM 实例
            generation_llm = get_generation_llm()
            rewrite_llm = get_rewrite_llm()

            # 创建 MainAgent（并行调用 RAG QA 工具）
            agent = MainAgent(
                llm=generation_llm,
                retriever=self.retriever,
                rewrite_llm=rewrite_llm,
                max_attempts=max_attempts
            )

            # 运行并行 RAG 工具编排器
            result = await agent.run_parallel(standalone_query)

            answer = result.get("answer", "")
            tool_calls = result.get("tool_calls", [])
            retrieved = result.get("retrieved", False)
            elapsed_time = result.get("elapsed_time", 0)
            
            # 从 RAGGraph 获取详细执行信息
            attempt_count = result.get("attempt_count", 0)
            query_history = result.get("query_history", [])
            confidence = result.get("confidence", 0.0)
            evaluation_reason = result.get("evaluation_reason", "")
            execution_log = result.get("execution_log", [])
            documents = result.get("documents", [])
            
            # 传统 RAG 格式的调试信息
            retrieval_steps = result.get("retrieval_steps", [])
            chunks_by_type = result.get("chunks_by_type", {"small": 0, "medium": 0, "large": 0})
            rerank_used = result.get("rerank_used", False)
            detail = result.get("detail", {})
            llm_messages_count = result.get("llm_messages_count", 0)
            all_iterations = result.get("all_iterations", result.get("iterations", []))
            step_timings = result.get("step_timings", [])
            generation_time = result.get("generation_time", 0)
            sub_tasks = result.get("sub_tasks", [])
            decomposed_tasks = result.get("decomposed_tasks", [])
            mode = result.get("mode", "parallel_rag_tools")

        except Exception as e:
            print(f"Agentic RAG error: {e}")
            answer = "抱歉，处理您的问题时出现了错误。"
            tool_calls = []
            retrieved = False
            elapsed_time = 0
            attempt_count = 0
            query_history = []
            confidence = 0.0
            evaluation_reason = ""
            execution_log = []
            documents = []
            retrieval_steps = []
            chunks_by_type = {"small": 0, "medium": 0, "large": 0}
            rerank_used = False
            detail = {}
            llm_messages_count = 0
            all_iterations = []
            step_timings = []
            generation_time = 0
            sub_tasks = []
            decomposed_tasks = []
            mode = "parallel_rag_tools"

        # 注册 chunk 统计更新（fire-and-forget）
        # 检索在 agent 内部完成（可能跨 asyncio.to_thread），不便透传 BackgroundTasks，
        # 因此在 agent 完成后用最终 documents 触发统计更新。
        self._schedule_stats_update(background_tasks, documents)

        # 计算处理时间
        process_time = time.time() - start_time

        # 构建 Agentic 调试信息（ReAct Commander 模式）
        agentic_info = {
            "mode": mode,
            "tool_calls": tool_calls,
            "retrieved": retrieved,
            "commander_elapsed_time": elapsed_time,
            "attempt_count": attempt_count,
            "query_history": query_history,
            "confidence": confidence,
            "evaluation_grade": result.get("grade", "unknown") if result else "unknown",
            "evaluation_reason": evaluation_reason,
            "execution_log": execution_log,
            "all_iterations": all_iterations,
            "step_timings": step_timings,
            "generation_time": generation_time,
            "decomposed_tasks": decomposed_tasks,
            "sub_tasks": sub_tasks,
        }

        memory_info = {
            "standalone_query": standalone_query,
            "summary_used": bool(memory_context.get("summary")),
            "unsummarized_message_count": len(memory_context.get("unsummarized_messages", [])),
            "long_term_memories": memory_context.get("long_term_memories", []),
            "token_budget": memory_context.get("token_budget", {}),
        }

        # 构建 debug_info（包含 agentic_info，用于持久化）
        debug_info = {
            "mode": mode,  # 标记当前编排模式
            "original_query": query,
            "rewritten_query": standalone_query if standalone_query != query else (query_history[-1] if query_history else None),
            "retrieval_steps": retrieval_steps,
            "total_chunks_retrieved": len(documents),
            "chunks_by_type": chunks_by_type,
            "rerank_used": rerank_used,
            "final_context": "\n\n".join([doc.get("content", "") for doc in documents]) if documents else None,
            "llm_messages_count": llm_messages_count,
            "detail": detail,
            # 性能调试信息
            "step_timings": step_timings,
            "generation_time": generation_time,
            # ReAct Commander 特有信息
            "agentic_info": agentic_info,
            "memory_info": memory_info
        }

        # 提取检索到的文档片段
        retrieved_chunks = []
        for doc in documents:
            retrieved_chunks.append({
                "id": doc.get("id", ""),
                "content": doc.get("content", ""),
                "score": doc.get("score", 0.0),
                "source": doc.get("source", "")
            })

        # 保存助手回答
        assistant_message = ChatMessage(
            conversation_id=conv_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            retrieved_chunks=retrieved_chunks,
            debug_info=debug_info,
            process_time=int(process_time * 1000)
        )
        self.db.add(assistant_message)

        # 更新会话时间
        conversation.updated_at = datetime.now()
        self.db.commit()
        self.db.refresh(conversation)

        if memory_service:
            self._schedule_memory_extraction(user_id, conv_id, query, answer)

        return {
            "answer": answer,
            "conversation_id": conv_id,
            "retrieved_chunks": retrieved_chunks,  # 返回检索到的文档片段
            "process_time": process_time,
            "debug_info": debug_info,
            "agentic_info": agentic_info
        }

    def _schedule_memory_extraction(self, user_id: int, conv_id: int, query: str, answer: str) -> None:
        async def run_extraction():
            try:
                await asyncio.to_thread(self._extract_memory_with_new_session, user_id, conv_id, query, answer)
            except Exception as e:
                print(f"Memory extraction error: {e}")

        try:
            asyncio.create_task(run_extraction())
        except RuntimeError:
            self._extract_memory_with_new_session(user_id, conv_id, query, answer)

    def _schedule_stats_update(self, background_tasks, documents: List[Dict[str, Any]]) -> None:
        """
        注册 chunk 统计更新（fire-and-forget）。

        优先使用 FastAPI BackgroundTasks（响应发送后执行）。
        若 background_tasks 不可用（非 API 调用），回退到 asyncio.create_task
        在独立线程中执行；若无事件循环则同步执行。

        documents 保留完整字段（rerank_score/rrf_score/dense_score），
        由 _update_stats_wrapper 归一化为单一 score。
        """
        if not documents:
            return
        # 只更新有 id 的文档（id 即 Milvus chunk id），保留 score 字段供归一化
        hits = [d for d in documents if d.get("id")]
        if not hits:
            return

        if background_tasks is not None:
            background_tasks.add_task(HybridRetriever._update_stats_wrapper, hits)
            return

        # 回退：无 BackgroundTasks 时用 asyncio.to_thread 异步执行
        async def _run_stats():
            try:
                await asyncio.to_thread(HybridRetriever._update_stats_wrapper, hits)
            except Exception as e:
                print(f"Chunk stats update error: {e}")

        try:
            asyncio.create_task(_run_stats())
        except RuntimeError:
            # 无事件循环，同步执行
            try:
                HybridRetriever._update_stats_wrapper(hits)
            except Exception as e:
                print(f"Chunk stats update (sync) error: {e}")

    def _extract_memory_with_new_session(self, user_id: int, conv_id: int, query: str, answer: str) -> None:
        db = SessionLocal()
        try:
            memory_service = MemoryService(db, self.retriever.vector_store)
            memory_service.extract_long_term_memories_after_turn(user_id, conv_id, query, answer)
        finally:
            db.close()

    def _sse_event(self, payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def agentic_chat_stream(
        self,
        user_id: int,
        query: str,
        conv_id: Optional[int] = None,
        max_attempts: int = 2,
        background_tasks=None
    ) -> AsyncGenerator[str, None]:
        """
        Agentic RAG 对话（流式输出版本）

        使用 MainAgent 作为顶层编排器，流式返回 LLM 生成的回答

        Args:
            user_id: 用户ID
            query: 用户问题
            conv_id: 会话ID（可选）
            max_attempts: RAGGraph 最大迭代次数
            background_tasks: FastAPI BackgroundTasks，用于 fire-and-forget 写 chunk 统计

        Yields:
            SSE 格式的数据块
        """
        start_time = time.time()
        retrieved_chunks_for_db = []

        yield self._sse_event({"type": "status", "stage": "conversation", "message": "正在保存问题..."})
        conv_id = await self._setup_conversation(user_id, query, conv_id)

        result = {}
        memory_service = None
        memory_context = {"summary": "", "unsummarized_messages": [], "long_term_memories": [], "token_budget": {}}
        standalone_query = query
        try:
            yield self._sse_event({"type": "status", "stage": "memory", "message": "正在结合会话记忆改写问题..."})
            memory_service = MemoryService(self.db, self.retriever.vector_store)
            memory_context = memory_service.prepare_memory_context(user_id, conv_id, query)
            standalone_query = memory_service.rewrite_query_with_memory(query, memory_context)

            yield self._sse_event({"type": "status", "stage": "rag", "message": "正在检索资料并评估相关性..."})
            generation_llm = get_generation_llm()
            rewrite_llm = get_rewrite_llm()

            agent = MainAgent(
                llm=generation_llm,
                retriever=self.retriever,
                rewrite_llm=rewrite_llm,
                max_attempts=max_attempts
            )

            result = await agent.run_parallel(standalone_query, generate_answer=False)

            documents = result.get("documents", [])
            candidate_documents = result.get("candidate_documents", documents)
            retrieval_steps = result.get("retrieval_steps", [])
            chunks_by_type = result.get("chunks_by_type", {"small": 0, "medium": 0, "large": 0})
            rerank_used = result.get("rerank_used", False)
            detail = result.get("detail", {})
            query_history = result.get("query_history", [])
            step_timings = result.get("step_timings", [])
            execution_log = result.get("execution_log", [])
            all_iterations = result.get("all_iterations", result.get("iterations", []))
            sub_tasks = result.get("sub_tasks", [])
            decomposed_tasks = result.get("decomposed_tasks", [])
            mode = result.get("mode", "parallel_rag_tools")

            for doc in documents or candidate_documents:
                retrieved_chunks_for_db.append({
                    "id": doc.get("id", ""),
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "source": doc.get("source", "")
                })

            context_parts = []
            for i, doc in enumerate((documents or candidate_documents)[:5]):
                context_parts.append(f"[{i+1}] {doc.get('content', '')}")
            context = "\n\n".join(context_parts)

            messages = self._build_rag_answer_messages(standalone_query, documents)

        except Exception as e:
            print(f"Agentic RAG stream setup error: {e}")
            messages = self._build_messages(query, "", conv_id)
            documents = []
            candidate_documents = []
            retrieval_steps = []
            chunks_by_type = {"small": 0, "medium": 0, "large": 0}
            rerank_used = False
            detail = {}
            query_history = [query]
            step_timings = []
            execution_log = []
            all_iterations = []
            sub_tasks = []
            decomposed_tasks = []
            mode = "parallel_rag_tools"
            context = ""
            memory_context = memory_context if 'memory_context' in locals() else {"summary": "", "unsummarized_messages": [], "long_term_memories": [], "token_budget": {}}
            standalone_query = standalone_query if 'standalone_query' in locals() else query
            result = {"answer": "抱歉，处理您的问题时出现了错误。"}
            yield self._sse_event({"type": "error", "message": str(e)})

        # 注册 chunk 统计更新（fire-and-forget）
        # 检索在 agent 内部完成（跨 asyncio.to_thread），在 agent 完成后用最终 documents 触发。
        self._schedule_stats_update(background_tasks, documents)

        full_answer = ""
        generation_time = 0
        if documents:
            yield self._sse_event({"type": "status", "stage": "answer", "message": "正在流式生成答案..."})
            answer_parts = []
            generate_start = time.time()
            try:
                async for token in self._call_llm_stream(messages):
                    answer_parts.append(token)
                    yield self._sse_event({'type': 'content', 'content': token})
                full_answer = "".join(answer_parts).strip()
                generation_time = round(time.time() - generate_start, 3)
            except Exception as e:
                print(f"LLM stream generation error: {e}")
                full_answer = "抱歉，生成答案时出现了错误。"
                generation_time = round(time.time() - generate_start, 3)
                yield self._sse_event({"type": "error", "message": str(e)})
                yield self._sse_event({'type': 'content', 'content': full_answer})
        else:
            full_answer = "未找到可用于回答的相关资料。"
            yield self._sse_event({'type': 'content', 'content': full_answer})

        process_time = time.time() - start_time

        memory_info = {
            "standalone_query": standalone_query,
            "summary_used": bool(memory_context.get("summary")),
            "unsummarized_message_count": len(memory_context.get("unsummarized_messages", [])),
            "long_term_memories": memory_context.get("long_term_memories", []),
            "token_budget": memory_context.get("token_budget", {}),
        }

        agentic_info = {
            "mode": mode,
            "decomposed_tasks": decomposed_tasks,
            "sub_tasks": sub_tasks,
            "candidate_documents": candidate_documents,
            "execution_log": execution_log,
            "all_iterations": all_iterations,
            "step_timings": step_timings,
            "generation_time": generation_time,
            "attempt_count": result.get("attempt_count", 0) if result else 0,
            "query_history": query_history,
            "confidence": result.get("confidence", 0.0) if result else 0.0,
            "evaluation_grade": result.get("grade", "unknown") if result else "unknown",
            "evaluation_reason": result.get("evaluation_reason", "") if result else "",
        }

        debug_info = {
            "mode": mode,
            "original_query": query,
            "rewritten_query": standalone_query if standalone_query != query else (query_history[-1] if len(query_history) > 1 else None),
            "retrieval_steps": retrieval_steps,
            "total_chunks_retrieved": len(documents),
            "candidate_chunks_retrieved": len(candidate_documents),
            "chunks_by_type": chunks_by_type,
            "rerank_used": rerank_used,
            "final_context": context if context else None,
            "llm_messages_count": len(messages),
            "detail": detail,
            "step_timings": step_timings,
            "generation_time": generation_time,
            "agentic_info": agentic_info,
            "memory_info": memory_info,
        }

        yield self._sse_event({"type": "status", "stage": "saving", "message": "正在保存回答和调试信息..."})
        assistant_message = ChatMessage(
            conversation_id=conv_id,
            role=MessageRole.ASSISTANT,
            content=full_answer,
            retrieved_chunks=retrieved_chunks_for_db,
            debug_info=debug_info,
            process_time=int(process_time * 1000)
        )
        self.db.add(assistant_message)
        self.db.flush()

        conversation = self.db.query(Conversation).filter(Conversation.id == conv_id).first()
        if conversation:
            conversation.updated_at = datetime.now()
        self.db.commit()

        if memory_service:
            self._schedule_memory_extraction(user_id, conv_id, query, full_answer)

        yield self._sse_event({
            'type': 'done',
            'conversation_id': conv_id,
            'process_time': process_time,
            'step_timings': step_timings,
            'generation_time': generation_time,
            'retrieved_chunks': retrieved_chunks_for_db,
            'debug_info': debug_info,
            'agentic_info': agentic_info,
        })


    async def _setup_conversation(self, user_id: int, query: str, conv_id: Optional[int] = None) -> int:
        """创建或获取会话，返回 conv_id"""
        if conv_id is None:
            conversation = self.create_conversation(user_id, ConversationCreate(title=query[:50]))
            conv_id = conversation.id
        else:
            conversation = self.get_conversation_by_id(conv_id, user_id)
            if not conversation:
                raise ValueError("Conversation not found")

        user_message = ChatMessage(
            conversation_id=conv_id,
            role=MessageRole.USER,
            content=query
        )
        self.db.add(user_message)
        self.db.commit()
        return conv_id

    async def _rewrite_query(self, query: str, conv_id: int) -> str:
        """
        使用LLM改写查询，结合对话上下文扩展同义词和相关概念，提高召回率
        """
        # 获取最近对话历史，让改写理解上下文
        history = self.get_messages(conv_id, limit=10)
        history_text = ""
        if len(history) > 1:
            history_parts = []
            for msg in history[:-1]:  # 排除刚插入的当前用户消息
                role = "用户" if msg.role.value == "user" else "助手"
                history_parts.append(f"{role}: {msg.content}")
            history_text = "\n".join(history_parts[-6:])  # 最近3轮对话

        context_hint = ""
        if history_text:
            context_hint = f"""
对话历史：
{history_text}

请结合对话历史理解用户当前问题的含义，确保改写后的查询是完整、独立的。例如如果对话历史在讨论"劳动合同终止"，用户追问"具体的情形"，改写时应当包含"劳动合同终止的具体情形"。

"""

        rewrite_prompt = f"""请将以下用户问题改写为更适合检索的查询。
{context_hint}要求：
1. 保持原意不变
2. 补充同义词和相关概念（如"终止"和"解除"）
3. 如果是短查询或追问，结合对话历史扩展为完整独立的表述
4. 只输出改写后的查询，不要解释

用户问题：{query}

改写后的查询："""

        messages = [{"role": "user", "content": rewrite_prompt}]

        try:
            rewritten = await self._call_llm(messages)
            print(f"Query rewritten: '{query}' -> '{rewritten.strip()}'")
            return rewritten.strip()
        except Exception as e:
            print(f"Query rewrite failed: {e}, using original query")
            return query

    def _build_rag_answer_messages(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        context_docs = sorted_docs[:8]
        context = "\n\n---\n\n".join([
            f"【文档片段 {i+1}】(相关度: {doc.get('score', 0):.2f})\n{doc.get('content', '')}"
            for i, doc in enumerate(context_docs)
        ])
        prompt = f"""你是一个专业的劳动法知识助手。请严格根据以下提供的参考文档回答用户问题。

参考文档：
{context}

用户问题：{query}

回答要求：
1. 严格基于参考文档内容回答，不得添加文档中未提及的任何信息
2. 直接引用文档中的原文条款，引用时标注来源（如"第X条"或"文档片段N"）
3. 如果文档内容足以回答问题，简洁准确地给出答案，不需要额外解读或分析
4. 如果文档内容仅部分相关，只回答文档能支撑的部分，对文档未覆盖的内容明确说明"参考文档未涉及"
5. 只有当所有文档内容都与问题完全无关时，才说明"未找到相关信息"
6. 禁止对法律条款进行延伸解读、推理或补充说明，只陈述文档中已有的内容

请给出回答："""
        return [{"role": "user", "content": prompt}]

    def _build_messages(self, query: str, context: str, conv_id: int) -> List[Dict[str, str]]:
        """
        构建发送给LLM的消息列表

        Args:
            query: 用户问题
            context: 检索到的上下文
            conv_id: 会话ID

        Returns:
            消息列表，包含系统提示、历史对话和当前问题
        """
        messages = []

        # 系统提示词
        system_prompt = "你是一个专业的知识问答助手。请用中文回答问题。"

        messages.append({"role": "system", "content": system_prompt})

        # 加入历史对话（最近10条）
        history = self.get_messages(conv_id, limit=20)
        for msg in history[-10:]:
            messages.append({"role": msg.role.value, "content": msg.content})

        # 加入当前问题（带 context 如果有）
        if context:
            user_content = f"""请根据下面的参考资料回答用户的问题。

参考资料：
{context}

用户问题：{query}

回答要求：
1. 必须基于参考资料回答，不要回答"未找到"或"暂无相关信息"
2. 直接给出答案要点，使用简洁的条目式回答
3. 引用具体法律条款编号（如有）"""
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})

        return messages

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 LLM API（OpenAI 兼容接口）

        统一使用 OpenAI 兼容接口调用 LLM，支持豆包、OpenAI、智谱等服务商。
        只需配置 CHAT_API_KEY、CHAT_API_BASE、CHAT_MODEL 即可切换不同服务。

        Args:
            messages: 消息列表

        Returns:
            LLM 的回答

        Raises:
            ValueError: 未配置 API Key
        """
        if not settings.CHAT_API_KEY:
            return self._mock_response(messages)

        url = f"{settings.CHAT_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.CHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": settings.CHAT_MODEL,
            "messages": messages,
            "temperature": 0.3  # 较低温度使输出更稳定，减少检索结果波动
        }

        try:
            response = await self._async_client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.ConnectError as e:
            print(f"LLM API connection error: {e}")
            raise
        except Exception as e:
            print(f"LLM API error: {e}")
            raise

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """
        模拟回答（当 API 不可用时使用）

        Args:
            messages: 消息列表

        Returns:
            模拟的回答
        """
        last_msg = messages[-1]["content"] if messages else "Hello"
        return f"This is a mock response to: '{last_msg}'. Please configure CHAT_API_KEY in .env file to get real responses."

    async def _call_llm_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        调用 LLM API 并流式返回（OpenAI 兼容接口）

        Args:
            messages: 消息列表

        Yields:
            LLM 回答的文本块

        Raises:
            ValueError: 未配置 API Key
        """
        if not settings.CHAT_API_KEY:
            full_response = self._mock_response(messages)
            for char in full_response:
                yield char
            return

        url = f"{settings.CHAT_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.CHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": settings.CHAT_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "stream": True
        }

        try:
            async with self._async_client.stream("POST", url, headers=headers, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        import json
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            print(f"LLM API connection error: {e}")
            raise
        except Exception as e:
            print(f"LLM API error: {e}")
            raise
