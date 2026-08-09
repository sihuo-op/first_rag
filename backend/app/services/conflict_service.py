"""
冲突检测服务

新文档入库后，对每个新 chunk 检索相似旧 chunk，用 LLM 判定是否语义冲突。
高置信度自动作废，低置信度转人工审核。
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import DocumentChunk
from app.llm.providers import get_rewrite_llm, invoke_llm_threadsafe
from app.rag.vector_store import MilvusStore

tracer = get_tracer("conflict")


class ConflictService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()
        self.llm = get_rewrite_llm()

    def judge_conflicts(self, new_content: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        用 LLM 判断 new_content 与每个 candidate 是否冲突。

        Args:
            new_content: 新 chunk 的文本内容
            candidates: 候选旧 chunk 列表，每项含 id 和 content

        Returns:
            判定结果列表，每项含 old_id / conflict / confidence / reason
        """
        if not candidates:
            return []

        prompt = self._build_prompt(new_content, candidates)
        try:
            response = self._invoke_llm(prompt)
            return self._parse_response(response, candidates)
        except Exception as e:
            print(f"[ConflictService] LLM 判冲突失败: {e}")
            return []

    def _invoke_llm(self, prompt: str):
        return invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])

    def _build_prompt(self, new_content: str, candidates: List[Dict]) -> str:
        cand_text = "\n".join(
            f"[{i}] id={c['id']}\n内容: {c['content']}"
            for i, c in enumerate(candidates)
        )
        return f"""判断新内容与下列每条旧内容是否语义冲突（讲同一件事但结论不同/已过期）。

新内容：{new_content}

旧内容列表：
{cand_text}

要求：
1. 对每条旧内容判断是否与新内容冲突
2. conflict=true 表示讲同一事但结论矛盾，新内容应取代旧内容
3. conflict=false 表示讲不同事、互补关系、或无关
4. confidence 范围 0-1

只输出 JSON 数组，不要其他文字：
[{{"old_id": "<id>", "conflict": true/false, "confidence": 0.0, "reason": "<简短原因>"}}]"""

    def _parse_response(self, response, candidates: List[Dict]) -> List[Dict]:
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON 数组（容错：LLM 可能加额外文字）
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group(0))
            # 校验结构
            valid_ids = {c["id"] for c in candidates}
            return [
                {
                    "old_id": item.get("old_id"),
                    "conflict": bool(item.get("conflict", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "reason": item.get("reason", "")
                }
                for item in result
                if item.get("old_id") in valid_ids
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[ConflictService] 解析 LLM 响应失败: {e}")
            return []

    def detect_for_document(self, doc_id: int) -> None:
        """
        对指定文档的所有 chunk 跑冲突检测管道。
        作为后台任务调用，不抛异常（失败只记日志）。
        """
        from app.entities.database import Document
        with tracer.start_as_current_span("conflict.detect_for_document") as span:
            span.set_attribute("conflict.doc_id", doc_id)
            try:
                document = self.db.query(Document).filter_by(id=doc_id).first()
                if not document:
                    print(f"[ConflictService] document {doc_id} not found")
                    return

                # 标记检测中
                document.conflict_check_status = "in_progress"
                document.conflict_check_started_at = datetime.utcnow()
                self.db.commit()

                new_chunks = self.db.query(DocumentChunk).filter_by(
                    document_id=doc_id, status="active"
                ).all()
                total = len(new_chunks)
                print(f"[ConflictService] doc {doc_id}: detecting conflicts for {total} chunks")

                processed = 0
                for chunk in new_chunks:
                    try:
                        self._detect_for_single_chunk(chunk, doc_id)
                    except Exception as e:
                        print(f"[ConflictService] chunk {chunk.id} detect failed: {e}")
                    processed += 1
                    document.conflict_check_progress = f"{processed}/{total}"
                    self.db.commit()

                document.conflict_check_status = "completed"
                document.conflict_check_completed_at = datetime.utcnow()
                self.db.commit()
                print(f"[ConflictService] doc {doc_id}: detection complete")

            except Exception as e:
                print(f"[ConflictService] detect_for_document failed: {e}")
                try:
                    document = self.db.query(Document).filter_by(id=doc_id).first()
                    if document:
                        document.conflict_check_status = "failed"
                        self.db.commit()
                except Exception:
                    pass

    def _detect_for_single_chunk(self, new_chunk: DocumentChunk, new_doc_id: int) -> None:
        """对单个新 chunk 检测冲突"""
        if not new_chunk.milvus_id or not new_chunk.content:
            return

        # Step 1: Milvus 检索相似旧 chunk（排除本文档）
        query_vector = self.vector_store.embed_query(new_chunk.content)
        candidates = self.vector_store.search_vectors(
            "chunks", query_vector, top_k=5,
            filter_expr=f"status == 'active' && document_id != {new_doc_id}"
        )
        if not candidates:
            return

        # Step 2: LLM 判冲突
        cand_for_llm = [{"id": c["id"], "content": c["content"]} for c in candidates]
        judgments = self.judge_conflicts(new_chunk.content, cand_for_llm)

        # Step 3: 按置信度分流
        high_threshold = self.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE
        low_threshold = self.settings.CONFLICT_DETECTION_LOW_CONFIDENCE

        for j in judgments:
            if not j["conflict"]:
                continue
            old_milvus_id = j["old_id"]
            old_chunk = self.db.query(DocumentChunk).filter_by(milvus_id=old_milvus_id).first()
            if not old_chunk or old_chunk.status != "active":
                continue

            confidence = j["confidence"]
            now = datetime.utcnow()

            if confidence >= high_threshold:
                # 自动作废
                old_chunk.status = "superseded"
                old_chunk.superseded_at = now
                old_chunk.conflict_with_chunk_id = new_chunk.milvus_id
                old_chunk.conflict_detected_at = now
                old_chunk.confidence = confidence
                old_chunk.review_reason = f"auto:conflict_with:{new_chunk.milvus_id}:{j['reason']}"
                self.db.commit()
                # 同步 Milvus（失败仅记日志，PG 为准，可后续 re-sync）
                try:
                    self.vector_store.upsert_status("chunks", old_milvus_id, "superseded")
                except Exception as e:
                    print(f"[ConflictService] Milvus sync failed for chunk {old_chunk.id} (superseded): {e}")
                print(f"[ConflictService] auto-superseded chunk {old_chunk.id} (confidence={confidence:.2f})")
            elif confidence >= low_threshold:
                # 转人工
                old_chunk.status = "pending_review"
                old_chunk.conflict_with_chunk_id = new_chunk.milvus_id
                old_chunk.conflict_detected_at = now
                old_chunk.confidence = confidence
                old_chunk.review_reason = f"review:conflict_with:{new_chunk.milvus_id}:{j['reason']}"
                self.db.commit()
                try:
                    self.vector_store.upsert_status("chunks", old_milvus_id, "pending_review")
                except Exception as e:
                    print(f"[ConflictService] Milvus sync failed for chunk {old_chunk.id} (pending_review): {e}")
                print(f"[ConflictService] pending_review chunk {old_chunk.id} (confidence={confidence:.2f})")
