"""测试 ConflictService 的 LLM 冲突判定逻辑（mock LLM）。"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.conflict_service import ConflictService


def test_judge_conflicts_returns_empty_when_no_candidates():
    """没有候选时返回空列表"""
    svc = ConflictService.__new__(ConflictService)
    result = svc.judge_conflicts("新内容", [])
    assert result == []


def test_judge_conflicts_parses_llm_response():
    """能正确解析 LLM 返回的 JSON 判定"""
    svc = ConflictService.__new__(ConflictService)
    candidates = [
        {"id": "old1", "content": "电池不可退货退款"},
        {"id": "old2", "content": "退款到原支付账户"},
    ]
    fake_llm_response = MagicMock()
    fake_llm_response.content = '''[
        {"old_id": "old1", "conflict": true, "confidence": 0.9, "reason": "结论矛盾"},
        {"old_id": "old2", "conflict": false, "confidence": 0.1, "reason": "讲不同事"}
    ]'''
    with patch.object(svc, "_invoke_llm", return_value=fake_llm_response):
        result = svc.judge_conflicts("所有商品都可7天无理由退款", candidates)

    assert len(result) == 2
    assert result[0]["old_id"] == "old1"
    assert result[0]["conflict"] is True
    assert result[0]["confidence"] == 0.9
    assert result[1]["conflict"] is False


def test_judge_conflicts_returns_empty_on_llm_failure():
    """LLM 异常时返回空列表（不阻塞）"""
    svc = ConflictService.__new__(ConflictService)
    with patch.object(svc, "_invoke_llm", side_effect=Exception("LLM down")):
        result = svc.judge_conflicts("新内容", [{"id": "x", "content": "旧内容"}])
    assert result == []


def test_detect_for_single_chunk_auto_supersede_on_high_confidence():
    """高置信度时自动作废旧 chunk"""
    from unittest.mock import MagicMock
    from datetime import datetime

    svc = ConflictService.__new__(ConflictService)
    svc.settings = MagicMock()
    svc.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE = 0.85
    svc.settings.CONFLICT_DETECTION_LOW_CONFIDENCE = 0.5

    new_chunk = MagicMock()
    new_chunk.milvus_id = "new1"
    new_chunk.content = "所有商品都可7天无理由退款"

    old_chunk = MagicMock()
    old_chunk.id = 100
    old_chunk.milvus_id = "old1"
    old_chunk.status = "active"

    # mock db query chain
    svc.db = MagicMock()
    svc.db.query.return_value.filter_by.return_value.first.return_value = old_chunk

    # mock vector_store
    svc.vector_store = MagicMock()
    svc.vector_store.embed_query.return_value = [0.1] * 1024
    svc.vector_store.search_vectors.return_value = [
        {"id": "old1", "content": "电池不可退货退款"}
    ]

    # mock judge_conflicts
    svc.judge_conflicts = MagicMock(return_value=[
        {"old_id": "old1", "conflict": True, "confidence": 0.92, "reason": "结论矛盾"}
    ])

    svc._detect_for_single_chunk(new_chunk, new_doc_id=999)

    assert old_chunk.status == "superseded"
    assert old_chunk.confidence == 0.92
    svc.vector_store.upsert_status.assert_called_once_with("chunks", "old1", "superseded")


def test_detect_for_single_chunk_pending_review_on_medium_confidence():
    """中置信度转人工审核"""
    from unittest.mock import MagicMock
    from app.services.conflict_service import ConflictService

    svc = ConflictService.__new__(ConflictService)
    svc.settings = MagicMock()
    svc.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE = 0.85
    svc.settings.CONFLICT_DETECTION_LOW_CONFIDENCE = 0.5

    new_chunk = MagicMock()
    new_chunk.milvus_id = "new1"
    new_chunk.content = "新内容"

    old_chunk = MagicMock()
    old_chunk.id = 100
    old_chunk.milvus_id = "old1"
    old_chunk.status = "active"

    svc.db = MagicMock()
    svc.db.query.return_value.filter_by.return_value.first.return_value = old_chunk
    svc.vector_store = MagicMock()
    svc.vector_store.embed_query.return_value = [0.1] * 1024
    svc.vector_store.search_vectors.return_value = [{"id": "old1", "content": "旧内容"}]
    svc.judge_conflicts = MagicMock(return_value=[
        {"old_id": "old1", "conflict": True, "confidence": 0.6, "reason": "可能冲突"}
    ])

    svc._detect_for_single_chunk(new_chunk, new_doc_id=999)

    assert old_chunk.status == "pending_review"
    svc.vector_store.upsert_status.assert_called_once_with("chunks", "old1", "pending_review")
