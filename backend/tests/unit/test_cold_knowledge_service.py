"""测试冷知识扫描规则。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.cold_knowledge_service import ColdKnowledgeService


def _make_chunk(status="active", last_accessed_at=None, access_count=0, hit_count=0, avg_score=0.0, created_at=None, archived_reason=None):
    c = MagicMock()
    c.status = status
    c.last_accessed_at = last_accessed_at
    c.access_count = access_count
    c.hit_count = hit_count
    c.avg_score = avg_score
    c.created_at = created_at or datetime.utcnow()
    c.archived_reason = archived_reason
    c.milvus_id = "milvus_x"
    return c


def test_timeout_rule_archives_old_unaccessed_chunk():
    """90 天未被访问的 chunk 归档为 timeout"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    old = _make_chunk(last_accessed_at=datetime.utcnow() - timedelta(days=100))
    svc.db.query.return_value.filter.return_value.all.return_value = [old]

    stats = svc.sweep()
    assert stats["timeout"] == 1
    assert old.status == "archived"
    assert old.archived_reason == "timeout"


def test_low_freq_rule_archives_unpopular_old_chunk():
    """上传 30+ 天且命中 < 2 次的归档为 low_freq"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=10),  # 不触发 timeout
        access_count=1,
        created_at=datetime.utcnow() - timedelta(days=40)  # 上传 40 天
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["low_freq"] == 1
    assert chunk.archived_reason == "low_freq"


def test_low_quality_rule_archives_low_score_chunk():
    """hit_count >= 5 且 avg_score < 0.3 的归档为 low_quality"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=1),
        access_count=10,
        hit_count=6,
        avg_score=0.2,
        created_at=datetime.utcnow() - timedelta(days=10)
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["low_quality"] == 1
    assert chunk.archived_reason == "low_quality"


def test_no_archive_for_fresh_chunk():
    """新 chunk（上传 < 30 天）即使命中少也不归档"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=1),
        access_count=1,
        hit_count=1,
        avg_score=0.5,
        created_at=datetime.utcnow() - timedelta(days=5)
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["timeout"] + stats["low_freq"] + stats["low_quality"] == 0
    assert chunk.status == "active"
