"""KGExtractor 管道编排测试：DB/LLM/Neo4j 全部 mock。"""
import logging
from unittest.mock import MagicMock, patch

from app.knowledge_graph.entity_resolver import ResolvedEntity
from app.knowledge_graph.extractor import ExtractionReport, KGExtractor
from app.knowledge_graph.llm_extractor import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
)
from app.knowledge_graph.rule_parser import ParsedDocument
from app.knowledge_graph.schema import ArticleNode, DocumentNode, LawNode


def _make_store() -> MagicMock:
    store = MagicMock()
    store.upsert_law.return_value = "law-1"
    store.upsert_article.return_value = "art-1"
    store.upsert_concept.return_value = "concept-1"
    store.upsert_party.return_value = "party-1"
    store.upsert_document.return_value = "doc-1"
    store.merge_relation.return_value = None
    return store


def _make_parsed(chunk_ids: list[str] | None = None) -> ParsedDocument:
    return ParsedDocument(
        law=LawNode(id="law-1", name="劳动合同法", level="法律"),
        document=DocumentNode(
            id="doc-1", source_file="x.txt", uploaded_at="2026-08-11", doc_type="law"
        ),
        articles=[
            ArticleNode(
                id="art-1",
                law_id="law-1",
                article_no=19,
                content_hash="abc",
                chunk_ids=chunk_ids if chunk_ids is not None else [],
                status="active",
                char_start=0,
                char_end=100,
            )
        ],
    )


def _make_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        entities=[ExtractedEntity(type="Concept", name="试用期")],
        relations=[
            ExtractedRelation(
                type="EXPLAINS",
                from_ref="article:劳动合同法:19",
                to_ref="concept:试用期",
                confidence=0.9,
            )
        ],
    )


def _make_resolver():
    resolver = MagicMock()
    resolver.resolve.side_effect = lambda entities, source_chunk_id: [
        ResolvedEntity(
            node_type=e.type,
            name=e.name,
            existing_node_id=None,
            source_chunk_id=source_chunk_id,
        )
        for e in entities
    ]
    return resolver


def _run_extractor(store: MagicMock, extraction_result) -> ExtractionReport:
    """用给定抽取结果跑一遍管道（单 chunk），返回 ExtractionReport。"""
    llm = MagicMock()
    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed(chunk_ids=["c1"])
        mock_extract.return_value = extraction_result
        mock_resolver_cls.return_value = _make_resolver()
        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(return_value=[0.1] * 1024),
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第十九条...", "char_start": 0, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=None,
        )
        return extractor.run(document_id="doc-1")


def test_extractor_pipeline_calls_all_steps():
    store = _make_store()
    embedding_fn = MagicMock(return_value=[0.1] * 1024)
    llm = MagicMock()

    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed()
        mock_extract.return_value = _make_extraction_result()
        mock_resolver = _make_resolver()
        mock_resolver_cls.return_value = mock_resolver

        extractor = KGExtractor(
            store=store,
            embedding_fn=embedding_fn,
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第十九条...", "char_start": 0, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=None,  # skip conflict detection in this test
        )
        report = extractor.run(document_id="doc-1")

    assert isinstance(report, ExtractionReport)
    assert report.entities_count >= 1
    assert report.relations_count >= 1


def test_extractor_writes_rule_nodes_and_resolved_relations():
    """Law/Document/Article 走规则写入，Article->Concept 关系端点解析为节点 ID。"""
    store = _make_store()
    llm = MagicMock()

    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed(chunk_ids=["c1"])
        mock_extract.return_value = _make_extraction_result()
        mock_resolver_cls.return_value = _make_resolver()

        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(return_value=[0.1] * 1024),
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第十九条...", "char_start": 0, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=None,
        )
        report = extractor.run(document_id="doc-1")

    # 规则节点写入
    store.upsert_law.assert_called_once()
    store.upsert_document.assert_called_once()
    store.upsert_article.assert_called_once()
    # 新建 Concept 节点带 embedding 和来源 chunk
    store.upsert_concept.assert_called_once_with(
        name="试用期", aliases=[], embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    # Law/Document CONTAINS Article + Article EXPLAINS Concept（共 3 次写入）
    relation_calls = [c.args for c in store.merge_relation.call_args_list]
    assert ("law-1", "art-1", "CONTAINS", {}) in relation_calls
    assert ("doc-1", "art-1", "CONTAINS", {}) in relation_calls
    assert ("art-1", "concept-1", "EXPLAINS", {"confidence": 0.9}) in relation_calls
    assert store.merge_relation.call_count == 3

    # relations_count 只统计 LLM 抽取的关系（规则写入的 CONTAINS 不计入）
    assert report.entities_count == 1
    assert report.relations_count == 1
    assert report.conflicts_count == 0
    assert report.document_id == "doc-1"
    assert report.duration_ms >= 0


def test_extractor_continues_on_chunk_failure():
    """单个 chunk 抽取失败不应中断整个文档：记录后继续处理后续 chunk。"""
    store = _make_store()
    llm = MagicMock()

    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed(chunk_ids=["c1", "c2"])
        mock_extract.side_effect = [
            RuntimeError("LLM provider down"),  # c1 失败
            _make_extraction_result(),          # c2 正常
        ]
        mock_resolver_cls.return_value = _make_resolver()

        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(return_value=[0.1] * 1024),
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第一条...", "char_start": 0, "char_end": 50, "document_id": "doc-1"},
                {"id": "c2", "content": "第十九条...", "char_start": 50, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=None,
        )
        report = extractor.run(document_id="doc-1")

    # c2 的结果仍然写入
    store.upsert_concept.assert_called_once()
    assert report.entities_count == 1
    assert report.relations_count >= 1


def test_extractor_run_never_raises_on_pipeline_failure(caplog):
    """run() 是 BackgroundTasks 直接调度的入口：管道抛异常（如非法规文档触发
    parse_document 的 ValueError）也不得逃逸，而是记录错误日志并返回带
    error 字段的失败报告，duration_ms 仍有值。"""
    store = _make_store()
    extractor = KGExtractor(
        store=store,
        embedding_fn=MagicMock(),
        chunks_loader=MagicMock(),
        document_loader=MagicMock(),
        conflict_detector=MagicMock(),
    )

    with patch.object(KGExtractor, "_run_impl", side_effect=ValueError("非法规文档")), \
         caplog.at_level(logging.ERROR, logger="app.knowledge_graph.extractor"):
        report = extractor.run(document_id="doc-1")

    assert isinstance(report, ExtractionReport)
    assert report.document_id == "doc-1"
    assert report.error == "ValueError: 非法规文档"
    assert report.entities_count == 0
    assert report.relations_count == 0
    assert report.duration_ms >= 0
    # 错误日志包含 document_id，便于排查上传触发失败
    assert any("document_id=doc-1" in r.getMessage() for r in caplog.records)


def test_extractor_skips_relation_when_ref_unresolvable():
    """from/to 引用无法解析到节点 ID 时跳过该关系，不抛异常。"""
    store = _make_store()
    llm = MagicMock()
    bad_result = ExtractionResult(
        entities=[ExtractedEntity(type="Concept", name="试用期")],
        relations=[
            ExtractedRelation(
                type="EXPLAINS",
                from_ref="article:劳动合同法:999",  # 不存在的条号
                to_ref="concept:试用期",
                confidence=0.9,
            ),
            ExtractedRelation(
                type="IS_A",
                from_ref="concept:试用期",
                to_ref="concept:不存在概念",  # 未抽取到的概念
                confidence=0.8,
            ),
        ],
    )

    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed(chunk_ids=["c1"])
        mock_extract.return_value = bad_result
        mock_resolver_cls.return_value = _make_resolver()

        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(return_value=[0.1] * 1024),
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第十九条...", "char_start": 0, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=None,
        )
        report = extractor.run(document_id="doc-1")

    assert report.relations_count == 0
    # 只剩规则写入的 Law/Document -> Article 关系
    assert store.merge_relation.call_count == 2


def test_extractor_drops_cross_law_article_refs():
    """法名不匹配的 article 引用必须丢弃：不得把跨法 CITES 误链到当前法同号条文。"""
    store = _make_store()
    cross_result = ExtractionResult(
        entities=[ExtractedEntity(type="Concept", name="试用期")],
        relations=[
            ExtractedRelation(
                type="CITES",
                from_ref="article:劳动合同法:19",
                to_ref="article:劳动法:19",  # 其它法；当前文档同样有第 19 条，不能误链
                confidence=0.9,
            ),
        ],
    )

    report = _run_extractor(store, cross_result)

    assert report.relations_count == 0
    relation_calls = [c.args for c in store.merge_relation.call_args_list]
    assert all(args[2] != "CITES" for args in relation_calls)
    # 只剩规则写入的 Law/Document -> Article 关系
    assert store.merge_relation.call_count == 2


def test_extractor_resolves_article_ref_with_matching_law_name():
    """法名匹配的 article 引用仍正常解析；'中华人民共和国' 前缀变体归一化后同样匹配。"""
    store = _make_store()
    result = ExtractionResult(
        entities=[ExtractedEntity(type="Concept", name="试用期")],
        relations=[
            ExtractedRelation(
                type="EXPLAINS",
                from_ref="article:中华人民共和国劳动合同法:19",  # 带国名前缀的变体
                to_ref="concept:试用期",
                confidence=0.9,
            ),
        ],
    )

    report = _run_extractor(store, result)

    assert report.relations_count == 1
    relation_calls = [c.args for c in store.merge_relation.call_args_list]
    assert ("art-1", "concept-1", "EXPLAINS", {"confidence": 0.9}) in relation_calls


# ---------- Task 10: conflict_detector 装配 ----------

def test_extractor_auto_instantiates_conflict_detector_when_none():
    """conflict_detector=None 时自动构造默认 ConflictDetector（LLM 走 extractor 的 provider）。"""
    store = _make_store()
    llm = MagicMock()
    with patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(),
            chunks_loader=MagicMock(),
            document_loader=MagicMock(),
            conflict_detector=None,
        )
    assert extractor.conflict_detector is not None
    assert extractor.conflict_detector.store is store
    assert extractor.conflict_detector.llm is llm


def test_extractor_uses_injected_conflict_detector_per_article():
    """注入的 conflict_detector 被 run() 逐 Article 调用，返回值累加进 report.conflicts_count。"""
    store = _make_store()
    llm = MagicMock()
    detector = MagicMock()
    detector.detect_for_article.return_value = 2

    with patch("app.knowledge_graph.extractor.parse_document") as mock_parse, \
         patch("app.knowledge_graph.extractor.extract_from_chunk") as mock_extract, \
         patch("app.knowledge_graph.extractor.EntityResolver") as mock_resolver_cls, \
         patch("app.knowledge_graph.extractor.get_extraction_llm", return_value=llm):
        mock_parse.return_value = _make_parsed(chunk_ids=["c1"])
        mock_extract.return_value = _make_extraction_result()
        mock_resolver_cls.return_value = _make_resolver()

        extractor = KGExtractor(
            store=store,
            embedding_fn=MagicMock(return_value=[0.1] * 1024),
            chunks_loader=MagicMock(return_value=[
                {"id": "c1", "content": "第十九条...", "char_start": 0, "char_end": 100, "document_id": "doc-1"},
            ]),
            document_loader=MagicMock(return_value="全文..."),
            conflict_detector=detector,
        )
        report = extractor.run(document_id="doc-1")

    detector.detect_for_article.assert_called_once_with("art-1")
    assert report.conflicts_count == 2
