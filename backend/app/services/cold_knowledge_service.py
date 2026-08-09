"""
冷知识识别与清理服务

定期扫描 active chunk，按四类信号判定是否归档：
- timeout: last_accessed_at > 90 天
- low_freq: 上传 > 30 天且 access_count < 2
- low_quality: hit_count >= 5 且 avg_score < 0.3
- manual: admin 手动归档（走 PATCH 接口，不走本服务）

归档后 90 天由 hard_delete_sweep 物理删除。
"""
from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import DocumentChunk
from app.rag.vector_store import MilvusStore

tracer = get_tracer("cold_knowledge")


class ColdKnowledgeService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()

    def sweep(self) -> Dict[str, int]:
        """扫描所有 active chunk，按规则归档。返回各类归档数量。"""
        with tracer.start_as_current_span("cold_knowledge.sweep") as span:
            stats = {"timeout": 0, "low_freq": 0, "low_quality": 0}
            now = datetime.utcnow()

            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.status == "active"
            ).all()

            print(f"[ColdKnowledge] scanning {len(chunks)} active chunks")
            for chunk in chunks:
                reason = self._classify(chunk, now)
                if reason:
                    self._archive(chunk, reason, now)
                    stats[reason] += 1

            self.db.commit()
            span.set_attribute("cold_knowledge.archived_total", sum(stats.values()))
            for k, v in stats.items():
                span.set_attribute(f"cold_knowledge.archived.{k}", v)
            print(f"[ColdKnowledge] done: {stats}")
            return stats

    def _classify(self, chunk: DocumentChunk, now: datetime) -> str:
        """判定 chunk 应归档的原因，返回 None 表示不归档"""
        # 规则 1: timeout
        if chunk.last_accessed_at:
            days_since_access = (now - chunk.last_accessed_at).days
            if days_since_access > self.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS:
                return "timeout"
        else:
            # 从未访问过，看 created_at
            if chunk.created_at:
                days_since_create = (now - chunk.created_at).days
                if days_since_create > self.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS:
                    return "timeout"

        # 规则 2: low_freq
        if chunk.created_at:
            days_since_create = (now - chunk.created_at).days
            if (days_since_create > self.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS
                    and chunk.access_count < self.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD):
                return "low_freq"

        # 规则 3: low_quality
        if (chunk.hit_count >= self.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS
                and chunk.avg_score < self.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE):
            return "low_quality"

        return None

    def _archive(self, chunk: DocumentChunk, reason: str, now: datetime) -> None:
        """归档单个 chunk（PG + Milvus）"""
        chunk.status = "archived"
        chunk.archived_reason = reason
        chunk.archived_at = now
        if chunk.milvus_id:
            try:
                self.vector_store.upsert_status("chunks", chunk.milvus_id, "archived")
            except Exception as e:
                print(f"[ColdKnowledge] upsert_status failed for chunk {chunk.id}: {e}")

    def hard_delete_sweep(self) -> int:
        """扫描归档超过保留期的 chunk，物理删除。返回删除数量。"""
        with tracer.start_as_current_span("cold_knowledge.hard_delete") as span:
            retention_days = self.settings.COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS
            cutoff = datetime.utcnow() - timedelta(days=retention_days)

            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.status == "archived",
                DocumentChunk.archived_at < cutoff
            ).all()

            print(f"[ColdKnowledge] hard deleting {len(chunks)} chunks archived before {cutoff}")
            for chunk in chunks:
                if chunk.milvus_id:
                    try:
                        self.vector_store.delete_vectors("chunks", ids=[chunk.milvus_id])
                    except Exception as e:
                        print(f"[ColdKnowledge] milvus delete failed for chunk {chunk.id}: {e}")
                self.db.delete(chunk)

            self.db.commit()
            span.set_attribute("cold_knowledge.hard_deleted", len(chunks))
            return len(chunks)
