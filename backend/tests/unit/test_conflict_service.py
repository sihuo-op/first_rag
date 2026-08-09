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
