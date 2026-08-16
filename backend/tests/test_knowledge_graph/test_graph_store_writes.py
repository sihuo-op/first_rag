"""graph_store 写方法测试。

- 集成测试使用 conftest.py 的 testcontainers `graph_store` fixture。
- 重试逻辑用 fake session 做 mock 测试（不起容器）。

关键回归（Task 7 handoff 硬性要求）：Level-2 别名合并会反复写入已存在的别名，
upsert_concept / upsert_party 必须对 aliases 做去重合并，绝不能朴素追加。
"""
import time

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.knowledge_graph.schema import ArticleNode, DocumentNode, LawNode


# ---------------------------------------------------------------------------
# upsert_law
# ---------------------------------------------------------------------------

def test_upsert_law_returns_id_and_merges(graph_store):
    law = LawNode(id="law-1", name="劳动合同法", level="法律", effective_date="2008-01-01")
    returned_id = graph_store.upsert_law(law)
    assert returned_id == "law-1"
    with graph_store.session() as s:
        result = s.run("MATCH (n:Law {id: $id}) RETURN n.name AS name", id="law-1")
        assert result.single()["name"] == "劳动合同法"


def test_upsert_law_idempotent(graph_store):
    law = LawNode(id="law-1", name="劳动合同法", level="法律")
    graph_store.upsert_law(law)
    graph_store.upsert_law(law)
    with graph_store.session() as s:
        result = s.run("MATCH (n:Law {id: $id}) RETURN count(*) AS c", id="law-1")
        assert result.single()["c"] == 1


# ---------------------------------------------------------------------------
# upsert_article
# ---------------------------------------------------------------------------

def test_upsert_article_idempotent(graph_store):
    article = ArticleNode(
        id="art-1", law_id="law-1", article_no=19, content_hash="abc",
        chunk_ids=["c1"], status="active", char_start=0, char_end=100,
    )
    assert graph_store.upsert_article(article) == "art-1"
    assert graph_store.upsert_article(article) == "art-1"
    with graph_store.session() as s:
        result = s.run("MATCH (n:Article {id: $id}) RETURN count(*) AS c", id="art-1")
        assert result.single()["c"] == 1


def test_upsert_article_persists_content(graph_store):
    """ArticleNode.content 必须落库：conflict_detector 的候选查询投影 new/existing.content。"""
    article = ArticleNode(
        id="art-c", law_id="law-1", article_no=19, content_hash="abc",
        content="第十九条 劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
        chunk_ids=[], status="active", char_start=0, char_end=100,
    )
    graph_store.upsert_article(article)
    with graph_store.session() as s:
        result = s.run(
            "MATCH (n:Article {id: $id}) RETURN n.content AS content", id="art-c",
        )
        assert result.single()["content"] == (
            "第十九条 劳动合同期限三个月以上不满一年的，试用期不得超过一个月。"
        )


# ---------------------------------------------------------------------------
# upsert_concept（含 Task 7 回归：别名去重）
# ---------------------------------------------------------------------------

def test_upsert_concept_with_embedding(graph_store):
    node_id = graph_store.upsert_concept(
        name="试用期", aliases=["试用期间"],
        embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    assert node_id
    with graph_store.session() as s:
        result = s.run(
            "MATCH (n:Concept {name: $name}) RETURN n.id AS id, n.aliases AS aliases",
            name="试用期",
        )
        record = result.single()
        assert record["id"] == node_id
        assert "试用期间" in record["aliases"]


def test_upsert_concept_idempotent_same_node(graph_store):
    id1 = graph_store.upsert_concept(
        name="试用期", aliases=["试用期间"], embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    id2 = graph_store.upsert_concept(
        name="试用期", aliases=["试用期间"], embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    assert id1 == id2
    with graph_store.session() as s:
        result = s.run("MATCH (n:Concept {name: $name}) RETURN count(*) AS c", name="试用期")
        assert result.single()["c"] == 1


def test_upsert_concept_alias_dedupe_on_reingest(graph_store):
    """Task 7 handoff 回归：重复写入同一别名不得产生重复项。"""
    graph_store.upsert_concept(
        name="试用期", aliases=["试用期间", "试用"],
        embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    # 再摄入：resolver Level-2 会把已存在的别名原样传回
    graph_store.upsert_concept(
        name="试用期", aliases=["试用期间", "试用"],
        embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    # 第三次：部分旧别名 + 一个新别名
    graph_store.upsert_concept(
        name="试用期", aliases=["试用期间", "试用期长度"],
        embedding=[0.1] * 1024, source_chunk_ids=["c1", "c1", "c2"],
    )
    with graph_store.session() as s:
        result = s.run(
            "MATCH (n:Concept {name: $name}) RETURN n.aliases AS aliases, n.source_chunk_ids AS chunk_ids",
            name="试用期",
        )
        record = result.single()
        assert set(record["aliases"]) == {"试用", "试用期间", "试用期长度"}
        assert len(record["aliases"]) == 3  # 无重复
        assert set(record["chunk_ids"]) == {"c1", "c2"}
        assert len(record["chunk_ids"]) == 2  # 无重复


def test_upsert_concept_dedupes_within_single_write(graph_store):
    """单次写入内部含重复别名也应去重。"""
    graph_store.upsert_concept(
        name="经济补偿", aliases=["经济补偿金", "经济补偿金"],
        embedding=[0.1] * 1024, source_chunk_ids=[],
    )
    with graph_store.session() as s:
        result = s.run(
            "MATCH (n:Concept {name: $name}) RETURN n.aliases AS aliases", name="经济补偿",
        )
        assert result.single()["aliases"] == ["经济补偿金"]


# ---------------------------------------------------------------------------
# upsert_party（别名去重同样适用）
# ---------------------------------------------------------------------------

def test_upsert_party_idempotent_and_alias_dedupe(graph_store):
    id1 = graph_store.upsert_party(name="用人单位", aliases=["雇主"], source_chunk_ids=["c1"])
    id2 = graph_store.upsert_party(name="用人单位", aliases=["雇主", "用工单位"], source_chunk_ids=["c1"])
    id3 = graph_store.upsert_party(name="用人单位", aliases=["雇主", "用工单位"], source_chunk_ids=["c1"])
    assert id1 == id2 == id3
    with graph_store.session() as s:
        result = s.run(
            "MATCH (n:Party {name: $name}) RETURN count(*) AS c, n.aliases AS aliases",
            name="用人单位",
        )
        record = result.single()
        assert record["c"] == 1
        assert set(record["aliases"]) == {"雇主", "用工单位"}
        assert len(record["aliases"]) == 2  # 无重复


# ---------------------------------------------------------------------------
# upsert_region / upsert_document
# ---------------------------------------------------------------------------

def test_upsert_region_idempotent(graph_store):
    id1 = graph_store.upsert_region(name="北京市", level="市")
    id2 = graph_store.upsert_region(name="北京市", level="市")
    assert id1 == id2
    with graph_store.session() as s:
        result = s.run("MATCH (n:Region {name: $name}) RETURN count(*) AS c", name="北京市")
        assert result.single()["c"] == 1


def test_upsert_document_idempotent(graph_store):
    doc = DocumentNode(id="doc-1", source_file="a.pdf", uploaded_at="2026-08-12T00:00:00", doc_type="law")
    assert graph_store.upsert_document(doc) == "doc-1"
    assert graph_store.upsert_document(doc) == "doc-1"
    with graph_store.session() as s:
        result = s.run("MATCH (n:Document {id: $id}) RETURN count(*) AS c, n.doc_type AS t", id="doc-1")
        record = result.single()
        assert record["c"] == 1
        assert record["t"] == "law"


# ---------------------------------------------------------------------------
# merge_relation
# ---------------------------------------------------------------------------

def test_merge_relation_creates_edge(graph_store):
    law_id = graph_store.upsert_law(LawNode(id="law-1", name="劳动合同法", level="法律"))
    art_id = graph_store.upsert_article(ArticleNode(
        id="art-1", law_id=law_id, article_no=19, content_hash="abc",
        chunk_ids=[], status="active", char_start=0, char_end=100,
    ))
    graph_store.merge_relation(law_id, art_id, "CONTAINS", {})
    with graph_store.session() as s:
        result = s.run(
            "MATCH (l:Law {id: $law_id})-[:CONTAINS]->(a:Article {id: $art_id}) RETURN count(*) AS c",
            law_id=law_id, art_id=art_id,
        )
        assert result.single()["c"] == 1


def test_merge_relation_idempotent_and_sets_props(graph_store):
    art_id = graph_store.upsert_article(ArticleNode(
        id="art-1", law_id="law-1", article_no=19, content_hash="abc",
        chunk_ids=[], status="active", char_start=0, char_end=100,
    ))
    concept_id = graph_store.upsert_concept(
        name="试用期", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=[],
    )
    graph_store.merge_relation(art_id, concept_id, "EXPLAINS", {"confidence": 0.9})
    graph_store.merge_relation(art_id, concept_id, "EXPLAINS", {"confidence": 0.9})
    with graph_store.session() as s:
        result = s.run(
            "MATCH (:Article {id: $a})-[r:EXPLAINS]->(:Concept {id: $c}) "
            "RETURN count(*) AS c, r.confidence AS conf",
            a=art_id, c=concept_id,
        )
        record = result.single()
        assert record["c"] == 1
        assert record["conf"] == pytest.approx(0.9)


def test_merge_relation_rejects_invalid_edge_type(graph_store):
    with pytest.raises(ValueError):
        graph_store.merge_relation("a", "b", "HACKED; MATCH (n) DETACH DELETE n", {})


def test_merge_relation_rejects_invalid_prop_key(graph_store):
    with pytest.raises(ValueError):
        graph_store.merge_relation("a", "b", "CITES", {"bad key": 1})


# ---------------------------------------------------------------------------
# find_articles_by_concept
# ---------------------------------------------------------------------------

def _make_article(graph_store, art_id, chunk_ids, article_no=19):
    # (law_id, article_no) 有唯一约束，同一测试内创建多个 Article 时 article_no 必须互异
    return graph_store.upsert_article(ArticleNode(
        id=art_id, law_id="law-1", article_no=article_no, content_hash=art_id,
        chunk_ids=chunk_ids, status="active", char_start=0, char_end=100,
    ))


def test_find_articles_by_concept(graph_store):
    concept_id = graph_store.upsert_concept(
        name="试用期", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=[],
    )
    art_id = _make_article(graph_store, "art-1", ["c1"])
    graph_store.merge_relation(art_id, concept_id, "EXPLAINS", {"confidence": 0.9})

    results = graph_store.find_articles_by_concept([concept_id])
    assert len(results) == 1
    assert results[0]["article_id"] == art_id
    assert results[0]["concept_hit_count"] == 1


def test_find_articles_by_concept_ranks_by_hit_count(graph_store):
    c1 = graph_store.upsert_concept(name="试用期", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=[])
    c2 = graph_store.upsert_concept(name="劳动合同", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=[])
    art_both = _make_article(graph_store, "art-both", ["c1"], article_no=1)
    art_one = _make_article(graph_store, "art-one", ["c2"], article_no=2)
    graph_store.merge_relation(art_both, c1, "EXPLAINS", {})
    graph_store.merge_relation(art_both, c2, "EXPLAINS", {})
    graph_store.merge_relation(art_one, c2, "EXPLAINS", {})

    results = graph_store.find_articles_by_concept([c1, c2])
    assert len(results) == 2
    assert results[0]["article_id"] == art_both
    assert results[0]["concept_hit_count"] == 2
    assert set(results[0]["matched_concepts"]) == {"试用期", "劳动合同"}
    assert results[1]["article_id"] == art_one
    assert results[1]["concept_hit_count"] == 1


def test_find_articles_by_concept_empty_input(graph_store):
    assert graph_store.find_articles_by_concept([]) == []


# ---------------------------------------------------------------------------
# 重试逻辑（mock，不起容器）
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, record):
        self._record = record

    def single(self):
        return self._record


class _FakeSession:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **kwargs):
        self._store.calls += 1
        if self._store.calls <= self._store.fail_n:
            raise self._store.exc("transient failure")
        return _FakeResult({"id": "fake-id"})


class _FakeStore:
    """只提供 session()，继承写 mixin 用于重试测试。"""

    def __init__(self, fail_n=0, exc=ServiceUnavailable):
        self.fail_n = fail_n
        self.exc = exc
        self.calls = 0

    def session(self):
        return _FakeSession(self)


def _make_fake_store(fail_n=0, exc=ServiceUnavailable):
    from app.knowledge_graph.graph_store import _Neo4jStoreWriteMixin

    class _Store(_FakeStore, _Neo4jStoreWriteMixin):
        pass

    return _Store(fail_n=fail_n, exc=exc)


def test_write_retries_transient_error_and_succeeds():
    store = _make_fake_store(fail_n=2)
    start = time.monotonic()
    node_id = store.upsert_law(LawNode(id="law-1", name="劳动合同法", level="法律"))
    elapsed = time.monotonic() - start
    assert node_id == "law-1"
    assert store.calls == 3  # 1 initial + 2 retries
    assert elapsed >= 0.5  # 等待 100ms + 500ms


def test_write_gives_up_after_3_retries():
    store = _make_fake_store(fail_n=99)
    with pytest.raises(ServiceUnavailable):
        store.upsert_law(LawNode(id="law-1", name="x", level="法律"))
    assert store.calls == 4  # 1 initial + 3 retries（100ms/500ms/2s）


def test_write_does_not_retry_non_retriable_error():
    store = _make_fake_store(fail_n=99, exc=ValueError)
    with pytest.raises(ValueError):
        store.upsert_law(LawNode(id="law-1", name="x", level="法律"))
    assert store.calls == 1  # 非瞬时错误不重试


def test_retry_applies_to_concept_upsert():
    store = _make_fake_store(fail_n=1)
    node_id = store.upsert_concept(
        name="试用期", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=[],
    )
    assert node_id == "fake-id"
    assert store.calls == 2
