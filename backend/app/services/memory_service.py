import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from opentelemetry import trace
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import ChatMessage, ConversationSummary, MemoryType, MessageRole, UserMemory
from app.llm.providers import get_rewrite_llm, invoke_llm_threadsafe
from app.rag.vector_store import MilvusStore
from app.services.token_budget import TokenBudget

tracer = get_tracer("memory")


class MemoryService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()
        self.rewrite_llm = get_rewrite_llm()
        model_name = self.settings.REWRITE_LLM_MODEL or self.settings.CHAT_MODEL
        self.token_budget = TokenBudget(model_name)

    def prepare_memory_context(self, user_id: int, conversation_id: int, query: str) -> Dict[str, Any]:
        if not self.settings.MEMORY_ENABLED:
            return self._empty_context(query)

        summary = self._get_or_create_summary(conversation_id)
        compressed = self._compress_if_needed(summary, query)
        self.db.refresh(summary)
        unsummarized_messages = self._get_unsummarized_messages(conversation_id, summary.last_summarized_message_id)
        long_term_memories = self.retrieve_long_term_memories(user_id, query, self.settings.MEMORY_RETRIEVAL_TOP_K)
        budget_info = self._build_budget_info(summary.summary, unsummarized_messages, query, compressed)

        return {
            "summary": summary.summary or "",
            "unsummarized_messages": self._serialize_messages(unsummarized_messages),
            "long_term_memories": long_term_memories,
            "token_budget": budget_info,
        }

    def rewrite_query_with_memory(self, query: str, memory_context: Dict[str, Any]) -> str:
        with tracer.start_as_current_span("memory.rewrite_query") as span:
            span.set_attribute("memory.query", query)
            span.set_attribute("memory.has_summary", bool(memory_context.get("summary")))
            span.set_attribute("memory.unsummarized_count", len(memory_context.get("unsummarized_messages", [])))
            span.set_attribute("memory.long_term_count", len(memory_context.get("long_term_memories", [])))
            return self._rewrite_query_with_memory_impl(query, memory_context)

    def _rewrite_query_with_memory_impl(self, query: str, memory_context: Dict[str, Any]) -> str:
        if not self.settings.MEMORY_ENABLED:
            return query

        summary = memory_context.get("summary", "")
        unsummarized_messages = memory_context.get("unsummarized_messages", [])
        long_term_memories = memory_context.get("long_term_memories", [])

        if not summary and not unsummarized_messages and not long_term_memories:
            return query

        recent_text = "\n".join(
            f"{self._role_label(item.get('role'))}: {item.get('content', '')}"
            for item in unsummarized_messages
        )
        memory_text = "\n".join(
            f"- [{item.get('memory_type')}] {item.get('content')}"
            for item in long_term_memories
        )

        prompt = f"""请根据会话记忆将用户当前问题改写成一个完整、独立、适合检索的劳动法问题。

会话摘要（较早历史）：
{summary or "无"}

未压缩的近期对话原文：
{recent_text or "无"}

长期用户记忆（只作为用户背景，不作为法律依据）：
{memory_text or "无"}

当前用户问题：{query}

要求：
1. 如果当前问题有代词、省略或承接上文，请补全背景
2. 保留用户真实意图，不添加未出现的事实
3. 不要直接回答问题
4. 只输出改写后的独立问题

独立问题："""
        try:
            response = invoke_llm_threadsafe(self.rewrite_llm, [HumanMessage(content=prompt)])
            rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return rewritten or query
        except Exception as e:
            print(f"[MemoryService] 记忆改写失败，使用原问题: {e}")
            return query

    def extract_long_term_memories_after_turn(self, user_id: int, conversation_id: int, user_query: str, answer: str) -> List[Dict[str, Any]]:
        with tracer.start_as_current_span("memory.extract") as span:
            span.set_attribute("memory.user_id", user_id)
            span.set_attribute("memory.conversation_id", conversation_id)
            span.set_attribute("memory.query_len", len(user_query))
            span.set_attribute("memory.answer_len", len(answer))
            try:
                result = self._extract_long_term_memories_impl(user_id, conversation_id, user_query, answer)
                span.set_attribute("memory.extracted_count", len(result))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def _extract_long_term_memories_impl(self, user_id: int, conversation_id: int, user_query: str, answer: str) -> List[Dict[str, Any]]:
        if not self.settings.MEMORY_ENABLED:
            return []

        prompt = f"""请从本轮对话中抽取值得长期记住的用户信息。

用户问题：{user_query}
助手回答：{answer}

只抽取以下类型：
- profile: 用户身份、角色、稳定背景
- preference: 用户偏好
- goal: 用户长期目标
- constraint: 用户约束条件
- case_fact: 用户案件事实或持续咨询背景

不要抽取：
- 一次性寒暄
- 助手给出的法律条文或法律结论
- 没有用户背景价值的普通问题

返回 JSON 数组，最多 {self.settings.MEMORY_EXTRACTION_MAX_ITEMS} 条：
[
  {{"memory_type": "case_fact", "content": "用户处于试用期，公司通知下周不用来了", "importance": 0.8}}
]
如果没有可记忆信息，返回 []。"""
        try:
            response = invoke_llm_threadsafe(self.rewrite_llm, [HumanMessage(content=prompt)])
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()
            json_match = re.search(r'\[[\s\S]*\]', content)
            raw_items = json.loads(json_match.group() if json_match else content)
        except Exception as e:
            print(f"[MemoryService] 长期记忆抽取失败: {e}")
            return []

        created = []
        source_ids = self._latest_message_ids(conversation_id, limit=2)
        for item in raw_items[:self.settings.MEMORY_EXTRACTION_MAX_ITEMS]:
            if not isinstance(item, dict):
                continue
            memory_type = self._parse_memory_type(item.get("memory_type"))
            content = str(item.get("content", "")).strip()
            if not memory_type or not content or self._memory_exists(user_id, memory_type, content):
                continue
            memory = UserMemory(
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=memory_type,
                content=content,
                source_message_ids=source_ids,
                importance=self._normalize_importance(item.get("importance", 0.5)),
                status="active",
            )
            self.db.add(memory)
            self.db.commit()
            self.db.refresh(memory)
            self._add_memory_vector(memory)
            created.append(self._serialize_memory(memory))
        return created

    def retrieve_long_term_memories(self, user_id: int, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store.has_collection(self.settings.MEMORY_COLLECTION_NAME):
                return []
            results = self.vector_store.search_memories(self.settings.MEMORY_COLLECTION_NAME, query, user_id, top_k)
        except Exception as e:
            print(f"[MemoryService] 长期记忆检索失败: {e}")
            return []

        memory_ids = [item.get("memory_id") for item in results if item.get("memory_id")]
        if not memory_ids:
            return []

        memories = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.id.in_(memory_ids),
            UserMemory.status == "active",
        ).all()
        score_by_id = {item.get("memory_id"): item.get("score", 0.0) for item in results}
        serialized = []
        for memory in memories:
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.now()
            item = self._serialize_memory(memory)
            item["score"] = score_by_id.get(memory.id, 0.0)
            serialized.append(item)
        self.db.commit()
        return sorted(serialized, key=lambda item: item.get("score", 0), reverse=True)

    def _compress_if_needed(self, summary: ConversationSummary, query: str) -> bool:
        compressed = False
        for _ in range(self.settings.MEMORY_MAX_COMPRESS_ROUNDS):
            unsummarized = self._get_unsummarized_messages(summary.conversation_id, summary.last_summarized_message_id)
            if not self._over_budget(summary.summary or "", unsummarized, query):
                break
            keep = self.settings.MEMORY_RECENT_MESSAGE_LIMIT
            compressible = unsummarized[:-keep] if len(unsummarized) > keep else []
            if not compressible:
                break
            selected = self._select_messages_to_compress(summary.summary or "", compressible, query)
            if not selected:
                break
            new_summary = self._summarize_messages(summary.summary or "", selected)
            summary.summary = new_summary
            summary.last_summarized_message_id = selected[-1].id
            summary.message_count = (summary.message_count or 0) + len(selected)
            summary.summary_token_count = self.token_budget.count_text(new_summary)
            self.db.commit()
            compressed = True
        return compressed

    def _over_budget(self, summary: str, messages: List[ChatMessage], query: str) -> bool:
        info = self._build_budget_info(summary, messages, query, False)
        return info["used"] > info["budget"]

    def _build_budget_info(self, summary: str, messages: List[Any], query: str, compressed: bool) -> Dict[str, Any]:
        model_window = self.settings.REWRITE_MODEL_CONTEXT_WINDOW
        reserved = self.settings.MEMORY_RESERVED_OUTPUT_TOKENS
        budget = int((model_window - reserved) * self.settings.MEMORY_CONTEXT_RATIO)
        if messages and isinstance(messages[0], dict):
            message_dicts = messages
        else:
            message_dicts = self._serialize_messages(messages)
        used = self.token_budget.count_text(summary or "") + self.token_budget.count_messages(message_dicts) + self.token_budget.count_text(query)
        return {"used": used, "budget": max(1, budget), "ratio": self.settings.MEMORY_CONTEXT_RATIO, "compressed": compressed}

    def _select_messages_to_compress(self, summary: str, messages: List[ChatMessage], query: str) -> List[ChatMessage]:
        selected = []
        for message in messages:
            selected.append(message)
            remaining = messages[len(selected):]
            if not self._over_budget(summary, remaining, query):
                break
        return selected

    def _summarize_messages(self, old_summary: str, messages: List[ChatMessage]) -> str:
        with tracer.start_as_current_span("memory.summarize") as span:
            span.set_attribute("memory.message_count", len(messages))
            span.set_attribute("memory.old_summary_len", len(old_summary or ""))
            return self._summarize_messages_impl(old_summary, messages)

    def _summarize_messages_impl(self, old_summary: str, messages: List[ChatMessage]) -> str:
        history = "\n".join(f"{self._role_label(message.role.value)}: {message.content}" for message in messages)
        prompt = f"""请更新会话滚动摘要，用于后续多轮问答理解上下文。

旧摘要：
{old_summary or "无"}

需要压缩进摘要的新消息：
{history}

要求：
1. 保留用户案件事实、约束、目标、已讨论的问题
2. 不要保留无意义寒暄
3. 不要添加新事实
4. 控制在约 {self.settings.MEMORY_SUMMARY_TARGET_TOKENS} tokens 内

新摘要："""
        response = invoke_llm_threadsafe(self.rewrite_llm, [HumanMessage(content=prompt)])
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def _get_or_create_summary(self, conversation_id: int) -> ConversationSummary:
        summary = self.db.query(ConversationSummary).filter(ConversationSummary.conversation_id == conversation_id).first()
        if summary:
            return summary
        summary = ConversationSummary(conversation_id=conversation_id, summary="", last_summarized_message_id=None, message_count=0, summary_token_count=0)
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        return summary

    def _get_unsummarized_messages(self, conversation_id: int, last_message_id: Optional[int]) -> List[ChatMessage]:
        query = self.db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation_id)
        if last_message_id:
            query = query.filter(ChatMessage.id > last_message_id)
        return query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()

    def _add_memory_vector(self, memory: UserMemory) -> None:
        try:
            if not self.vector_store.has_collection(self.settings.MEMORY_COLLECTION_NAME):
                self.vector_store.create_memory_collection(self.settings.MEMORY_COLLECTION_NAME)
            self.vector_store.add_memory_texts(self.settings.MEMORY_COLLECTION_NAME, [memory.content], [{
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "conversation_id": memory.conversation_id or 0,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance or 0.5,
            }])
        except Exception as e:
            print(f"[MemoryService] 写入记忆向量失败: {e}")

    def _memory_exists(self, user_id: int, memory_type: MemoryType, content: str) -> bool:
        return self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.memory_type == memory_type,
            UserMemory.content == content,
            UserMemory.status == "active",
        ).first() is not None

    def _latest_message_ids(self, conversation_id: int, limit: int) -> List[int]:
        messages = self.db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation_id).order_by(ChatMessage.id.desc()).limit(limit).all()
        return [message.id for message in reversed(messages)]

    def _serialize_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        return [{"id": message.id, "role": message.role.value, "content": message.content} for message in messages]

    def _serialize_memory(self, memory: UserMemory) -> Dict[str, Any]:
        return {
            "id": memory.id,
            "memory_type": memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
            "content": memory.content,
            "importance": memory.importance,
            "conversation_id": memory.conversation_id,
        }

    def _parse_memory_type(self, value) -> Optional[MemoryType]:
        try:
            return MemoryType(str(value).lower())
        except Exception:
            return None

    def _normalize_importance(self, value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.5

    def _role_label(self, role) -> str:
        value = role.value if hasattr(role, "value") else str(role)
        return "用户" if value == "user" else "助手" if value == "assistant" else "系统"

    def _empty_context(self, query: str) -> Dict[str, Any]:
        return {"summary": "", "unsummarized_messages": [], "long_term_memories": [], "token_budget": self._build_budget_info("", [], query, False)}
