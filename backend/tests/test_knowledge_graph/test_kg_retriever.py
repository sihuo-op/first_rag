"""KGRetriever 单测：全部 mock Neo4j/Milvus，不依赖真实数据库。

mock 约定：
- Concept 向量检索走 ``store.session()`` -> ``run(...)``，结果可迭代（模拟 neo4j Result）。
- Article 查询走 Task 8 的 ``store.find_articles_by_concept``，直接 mock 其返回值。
- chunk 反查走 ``vector_store.get_chunks_by_ids``。

全局约束：任何 KG 异常都必须内部吞掉并返回 []（RRF 自动降级为两路），
绝不能让 KG 检索拖垮主检索链路。
"""
from unittest.mock import MagicMock

from app.knowledge_graph.exceptions import KGQueryError
from app.knowledge_graph.kg_retriever import KGRetriever


def _make_concept_result(records):
    """模拟 neo4j Result：可迭代，产出 dict 记录。"""
    return MagicMock(__iter__=lambda self: iter(records))


def _make_store(concept_records, articles=None):
    """构造 mock Neo4jStore：session().run() 返回 concept_records，
    find_articles_by_concept 返回 articles。"""
    store = MagicMock()
    session_mock = MagicMock()
    session_mock.run.return_value = _make_concept_result(concept_records)
    store.session.return_value.__enter__.return_value = session_mock
    if articles is not None:
        store.find_articles_by_concept.return_value = articles
    return store


# ---------- 无匹配 -> 空结果 ----------

def test_retrieve_returns_empty_when_no_concepts_matched():
    store = MagicMock()
    # Concept vector search returns no results
    store.session.return_value.__enter__.return_value.run.return_value = []
    vector_store = MagicMock()
    retriever = KGRetriever(store=store, vector_store=vector_store)
    results = retriever.retrieve(query="无关问题", query_embedding=[0.1] * 1024)
    assert results == []
    # 无 concept 时不应触发 Article 查询
    store.find_articles_by_concept.assert_not_called()


# ---------- 正常链路：concept -> article -> chunks ----------

def test_retrieve_finds_articles_and_fetches_chunks():
    store = _make_store(
        concept_records=[{"id": "c-1", "name": "试用期"}],
        articles=[{
            "article_id": "art-1",
            "chunk_ids": ["chunk-1", "chunk-2"],
            "matched_concepts": ["试用期"],
            "concept_hit_count": 1,
        }],
    )

    vector_store = MagicMock()
    vector_store.get_chunks_by_ids.return_value = [
        {"id": "chunk-1", "content": "第一条...", "chunk_type": "small", "document_id": 1},
        {"id": "chunk-2", "content": "第二条...", "chunk_type": "small", "document_id": 1},
    ]

    retriever = KGRetriever(store=store, vector_store=vector_store)
    results = retriever.retrieve(query="试用期多长", query_embedding=[0.1] * 1024)

    assert len(results) == 2
    assert all("kg_score" in r for r in results)
    assert all(r["chunk_type"] == "small" for r in results)
    assert {r["chunk_id"] for r in results} == {"chunk-1", "chunk-2"}
    assert all(r["content"] for r in results)
    # Article 查询复用 Task 8 的 find_articles_by_concept，深度用默认 2
    store.find_articles_by_concept.assert_called_once_with(["c-1"], max_depth=2)


# ---------- kg_score 归一化：hit_count / max_hit ----------

def test_kg_score_normalized_by_concept_hit_count():
    store = _make_store(
        concept_records=[{"id": "c-1"}],
        articles=[
            {"article_id": "art-1", "chunk_ids": ["chunk-a"], "concept_hit_count": 2, "matched_concepts": ["试用期", "经济补偿"]},
            {"article_id": "art-2", "chunk_ids": ["chunk-b"], "concept_hit_count": 1, "matched_concepts": ["试用期"]},
            # 无 chunk_ids 的 Article 直接跳过
            {"article_id": "art-3", "chunk_ids": [], "concept_hit_count": 1, "matched_concepts": ["试用期"]},
        ],
    )
    vector_store = MagicMock()
    vector_store.get_chunks_by_ids.side_effect = [
        [{"id": "chunk-a", "content": "甲", "chunk_type": "small"}],
        [{"id": "chunk-b", "content": "乙", "chunk_type": "small"}],
    ]

    retriever = KGRetriever(store=store, vector_store=vector_store)
    results = retriever.retrieve(query="试用期 经济补偿", query_embedding=[0.1] * 1024)

    scores = {r["chunk_id"]: r["kg_score"] for r in results}
    assert scores == {"chunk-a": 1.0, "chunk-b": 0.5}


# ---------- 同一 chunk 被多个 Article 引用 -> 去重取最高分 ----------

def test_retrieve_dedupes_chunks_across_articles_keeping_max_score():
    store = _make_store(
        concept_records=[{"id": "c-1"}],
        articles=[
            {"article_id": "art-1", "chunk_ids": ["chunk-1"], "concept_hit_count": 2, "matched_concepts": ["试用期", "解除"]},
            {"article_id": "art-2", "chunk_ids": ["chunk-1"], "concept_hit_count": 1, "matched_concepts": ["试用期"]},
        ],
    )
    vector_store = MagicMock()
    vector_store.get_chunks_by_ids.side_effect = [
        [{"id": "chunk-1", "content": "第一条", "chunk_type": "small"}],
        [{"id": "chunk-1", "content": "第一条", "chunk_type": "small"}],
    ]

    retriever = KGRetriever(store=store, vector_store=vector_store)
    results = retriever.retrieve(query="试用期 解除", query_embedding=[0.1] * 1024)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["kg_score"] == 1.0


# ---------- Party/Region 词表抽取 ----------

def test_party_region_extracted_via_wordlist():
    store = MagicMock()
    session_mock = MagicMock()
    session_mock.run.return_value = MagicMock(__iter__=lambda self: iter([]))
    store.session.return_value.__enter__.return_value = session_mock

    retriever = KGRetriever(store=store, vector_store=MagicMock())
    party, region = retriever._extract_party_region("北京用人单位")
    assert party == ["用人单位"]
    assert region == ["北京"]


# ---------- 失败回退：任何异常 -> []，绝不抛出 ----------

def test_retrieve_returns_empty_on_kgerror():
    store = _make_store(concept_records=[{"id": "c-1"}])
    store.find_articles_by_concept.side_effect = KGQueryError("Cypher 查询失败")
    retriever = KGRetriever(store=store, vector_store=MagicMock())
    assert retriever.retrieve(query="试用期多长", query_embedding=[0.1] * 1024) == []


def test_retrieve_returns_empty_on_unexpected_error():
    store = MagicMock()
    store.session.side_effect = RuntimeError("neo4j connection reset")
    retriever = KGRetriever(store=store, vector_store=MagicMock())
    assert retriever.retrieve(query="试用期多长", query_embedding=[0.1] * 1024) == []


def test_retrieve_returns_empty_when_vector_fetch_fails():
    store = _make_store(
        concept_records=[{"id": "c-1"}],
        articles=[{"article_id": "art-1", "chunk_ids": ["chunk-1"], "concept_hit_count": 1, "matched_concepts": ["试用期"]}],
    )
    vector_store = MagicMock()
    vector_store.get_chunks_by_ids.side_effect = TimeoutError("milvus timeout")
    retriever = KGRetriever(store=store, vector_store=vector_store)
    assert retriever.retrieve(query="试用期多长", query_embedding=[0.1] * 1024) == []


# ---------- MilvusStore.get_chunks_by_ids ----------

def test_milvus_get_chunks_by_ids_queries_by_id_list():
    from app.rag.vector_store import MilvusStore

    vs = MilvusStore(host="localhost", port=19530)
    vs.connect = MagicMock()
    collection = MagicMock()
    collection.query.return_value = [
        {"id": "chunk-1", "document_id": 1, "chunk_type": "small",
         "content": "第一条...", "content_hash": "h1", "status": "active"},
    ]
    vs._get_collection = MagicMock(return_value=collection)

    result = vs.get_chunks_by_ids("chunks", ["chunk-1", "chunk-2"])

    assert len(result) == 1
    assert result[0]["id"] == "chunk-1"
    assert result[0]["chunk_type"] == "small"
    assert result[0]["content"] == "第一条..."
    # expr 按 id 列表过滤，字符串 id 需要合法引号
    expr = collection.query.call_args.kwargs["expr"]
    assert "chunk-1" in expr and "chunk-2" in expr


def test_milvus_get_chunks_by_ids_empty_ids_returns_empty():
    from app.rag.vector_store import MilvusStore

    vs = MilvusStore(host="localhost", port=19530)
    assert vs.get_chunks_by_ids("chunks", []) == []
