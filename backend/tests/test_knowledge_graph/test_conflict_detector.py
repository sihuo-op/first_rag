"""ConflictDetector 单测：Neo4j/LLM 全部 mock，不依赖真实数据库与真实 LLM。

mock 约定：
- ``store.session.return_value.__enter__.return_value.run.return_value`` 为
  ``_find_related_articles`` 的查询结果行列表（dict 形状，与真实 Record 的
  ``dict(r)`` 行为一致）。
- LLM 通过 patch ``app.knowledge_graph.conflict_detector.invoke_llm_threadsafe`` mock。
"""
import json
from unittest.mock import MagicMock, patch

from app.knowledge_graph.conflict_detector import ConflictDetector
from app.knowledge_graph.schema import ConflictStatus


def _related_row(
    existing_id="art-2",
    new_content="第十九条 劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
    existing_content="第十九条 试用期最长不得超过六个月。",
):
    # new_content/existing_content 与生产 Cypher 投影对齐：条款原文（n.content），
    # 而非 content_hash（sha256 hex 摘要，对 LLM 判定毫无意义）。
    return {
        "existing_id": existing_id,
        "concept_name": "试用期",
        "concept_id": "c-1",
        "new_content": new_content,
        "existing_content": existing_content,
    }


def _llm_response(payload: dict):
    resp = MagicMock()
    resp.content = json.dumps(payload)
    return resp


def _make_store(rows):
    store = MagicMock()
    store.session.return_value.__enter__.return_value.run.return_value = rows
    return store


# ---------- 基础行为 ----------

def test_detect_no_existing_articles_returns_zero():
    store = _make_store([])
    detector = ConflictDetector(store, llm=MagicMock())
    assert detector.detect_for_article("art-1") == 0


def test_detect_finds_conflict_writes_edge():
    store = _make_store([_related_row(new_content="最长6个月", existing_content="最长3个月")])
    fake_response = _llm_response({
        "is_conflict": True, "reason": "6个月 vs 3个月 互斥", "confidence": 0.9
    })

    with patch("app.knowledge_graph.conflict_detector.invoke_llm_threadsafe", return_value=fake_response):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 1


def test_detect_no_conflict_skips_edge():
    store = _make_store([_related_row(new_content="包含在合同期内", existing_content="最长6个月")])
    fake_response = _llm_response({
        "is_conflict": False, "reason": "互补", "confidence": 0.8
    })

    with patch("app.knowledge_graph.conflict_detector.invoke_llm_threadsafe", return_value=fake_response):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 0


# ---------- 返回值契约：永远是 int（Task 9 extractor 用 += 累加） ----------

def test_detect_for_article_always_returns_int():
    store = _make_store([])
    detector = ConflictDetector(store, llm=MagicMock())
    result = detector.detect_for_article("art-1")
    assert isinstance(result, int)

    store2 = _make_store([_related_row()])
    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": True, "reason": "x", "confidence": 0.9}),
    ):
        detector2 = ConflictDetector(store2, llm=MagicMock())
        result2 = detector2.detect_for_article("art-1")
    assert isinstance(result2, int)


# ---------- 边写入：CONFLICTS_WITH + status=pending_review ----------

def test_conflict_edge_written_with_pending_review_status():
    store = _make_store([_related_row()])
    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": True, "reason": "互斥", "confidence": 0.9}),
    ):
        detector = ConflictDetector(store, llm=MagicMock())
        detector.detect_for_article("art-1")

    store.merge_relation.assert_called_once()
    args = store.merge_relation.call_args.args
    assert args[0] == "art-1"
    assert args[1] == "art-2"
    assert args[2] == "CONFLICTS_WITH"
    props = args[3]
    assert props["status"] == ConflictStatus.PENDING_REVIEW.value
    assert props["reason"] == "互斥"
    assert props["confidence"] == 0.9
    assert "detected_at" in props


def test_no_conflict_writes_no_edge():
    store = _make_store([_related_row()])
    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": False, "reason": "互补", "confidence": 0.8}),
    ):
        detector = ConflictDetector(store, llm=MagicMock())
        detector.detect_for_article("art-1")

    store.merge_relation.assert_not_called()


def test_multiple_conflicts_counted():
    store = _make_store([_related_row("art-2"), _related_row("art-3")])
    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": True, "reason": "互斥", "confidence": 0.9}),
    ):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 2
    assert store.merge_relation.call_count == 2


# ---------- 候选查找：共享 Concept + EXPLAINS，且排除已有 CONFLICTS_WITH ----------

def test_candidate_query_uses_explains_and_excludes_existing_conflicts():
    """Cypher 必须经 (Article)-[:EXPLAINS]->(Concept)<-[:EXPLAINS]-(Article) 找候选，
    并排除已存在 CONFLICTS_WITH 边的文章对（避免重复检测 / 覆盖人工审核状态）。"""
    store = _make_store([])
    detector = ConflictDetector(store, llm=MagicMock())
    detector.detect_for_article("art-1")

    run_mock = store.session.return_value.__enter__.return_value.run
    query = run_mock.call_args.args[0]
    assert "EXPLAINS" in query
    assert "Concept" in query
    assert "NOT (new)-[:CONFLICTS_WITH]-(existing)" in query
    assert run_mock.call_args.kwargs == {"article_id": "art-1"}


def test_candidate_query_projects_article_content_not_hash():
    """候选查询必须投影 n.content（条款原文）给 LLM，绝不能用 content_hash。"""
    store = _make_store([])
    detector = ConflictDetector(store, llm=MagicMock())
    detector.detect_for_article("art-1")

    query = store.session.return_value.__enter__.return_value.run.call_args.args[0]
    assert "new.content AS new_content" in query
    assert "existing.content AS existing_content" in query
    assert "content_hash" not in query


# ---------- LLM 重试：失败重试 2 次（共 3 次尝试） ----------

def test_llm_failure_retries_then_succeeds():
    store = _make_store([_related_row()])
    good = _llm_response({"is_conflict": True, "reason": "互斥", "confidence": 0.9})

    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        side_effect=[RuntimeError("provider down"), RuntimeError("timeout"), good],
    ) as mock_invoke:
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert mock_invoke.call_count == 3  # 1 次初始 + 2 次重试
    assert count == 1


def test_llm_all_retries_exhausted_skips_conflict():
    store = _make_store([_related_row()])

    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        side_effect=RuntimeError("provider down"),
    ) as mock_invoke:
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert mock_invoke.call_count == 3
    assert count == 0
    store.merge_relation.assert_not_called()


# ---------- LLM 输出容错 ----------

def test_malformed_json_returns_no_conflict():
    store = _make_store([_related_row()])
    bad = MagicMock()
    bad.content = "这不是 JSON"

    with patch("app.knowledge_graph.conflict_detector.invoke_llm_threadsafe", return_value=bad):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 0
    store.merge_relation.assert_not_called()


def test_json_in_markdown_code_block_extracted():
    store = _make_store([_related_row()])
    wrapped = MagicMock()
    wrapped.content = '```json\n{"is_conflict": true, "reason": "互斥", "confidence": 0.9}\n```'

    with patch("app.knowledge_graph.conflict_detector.invoke_llm_threadsafe", return_value=wrapped):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 1


def test_non_numeric_confidence_defaults_to_zero():
    store = _make_store([_related_row()])

    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": True, "reason": "互斥", "confidence": "高"}),
    ):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 1
    props = store.merge_relation.call_args.args[3]
    assert props["confidence"] == 0.0


def test_null_content_treated_as_parse_failure():
    store = _make_store([_related_row()])
    null_resp = MagicMock()
    null_resp.content = None

    with patch("app.knowledge_graph.conflict_detector.invoke_llm_threadsafe", return_value=null_resp):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 0


def test_string_false_is_conflict_treated_as_false():
    """LLM 偶尔回字符串 "false"：朴素 bool("false") 是 True，必须宽松解析为 False。"""
    store = _make_store([_related_row()])

    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": "false", "reason": "互补", "confidence": 0.8}),
    ):
        detector = ConflictDetector(store, llm=MagicMock())
        count = detector.detect_for_article("art-1")

    assert count == 0
    store.merge_relation.assert_not_called()


# ---------- LLM 提示词包含双方内容 ----------

def test_llm_prompt_contains_article_text_not_hashes():
    """提示词必须携带条款原文（n.content）：含具体条文文本，且不含 hash 摘要。"""
    store = _make_store([_related_row(
        new_content="第十九条 劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
        existing_content="第十九条 试用期最长不得超过六个月。",
    )])

    with patch(
        "app.knowledge_graph.conflict_detector.invoke_llm_threadsafe",
        return_value=_llm_response({"is_conflict": False, "reason": "-", "confidence": 0.5}),
    ) as mock_invoke:
        detector = ConflictDetector(store, llm=MagicMock())
        detector.detect_for_article("art-1")

    prompt = mock_invoke.call_args.args[1][0].content
    assert "试用期" in prompt
    assert "不得超过一个月" in prompt
    assert "不得超过六个月" in prompt
    # 回归：不得再出现 hex hash 摘要（旧 bug 把 content_hash 当"条文内容"喂给 LLM）
    import re as _re
    assert not _re.search(r"\b[0-9a-f]{32,}\b", prompt)
