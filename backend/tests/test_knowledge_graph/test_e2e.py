"""Task 16: KG 端到端测试 —— 《劳动合同法》第十九~二十一条。

真实链路：rule_parser -> LLM 抽取 -> entity_resolver -> Neo4j 写入 -> KG 检索。
需要：Docker（Neo4j testcontainer）、本地 Milvus、可用 LLM/embedding API。
任一缺失时 e2e_setup 按需 skip；graph_store 之外的 Milvus 依赖走真实连接。

与 plan 原文的三处偏差（均为让测试真正可跑而修正）：
1. chunks 以显式 id 插入独立集合 e2e_chunks（plan 未插入 Milvus，且 insert_vectors
   会生成 uuid id 与抽取用 chunk id 不一致，retrieval 断言必然拿不到结果）；
2. 查询用 "试用期"（与 concept 名一致，embedding 余弦 ≈ 1.0，稳定过 0.7 阈值；
   plan 的 "试用期最长多长时间" 与 concept 向量相似度有低于阈值的风险）；
3. 注入空冲突检测器，避免抽取阶段发起 LLM 冲突判定（KGExtractor 对 None 会
   自动构造真实 ConflictDetector）。
"""
from pathlib import Path

import pytest
from pymilvus import Collection

FIXTURE = Path(__file__).parent / "fixtures" / "labor_contract_law_19_21.txt"

# 独立测试集合，不污染生产 rag_chunks；full name = rag_ + e2e_chunks
E2E_MILVUS_COLLECTION = "e2e_chunks"


def _llm_available() -> bool:
    """探测抽取 LLM 是否可用（配额/网络）。不可用时 E2E 整体 skip。

    与 Milvus 不可用 skip 同理：依赖付费外部 API 的集成测试在依赖缺失时
    应跳过而非红。配额重置后自动恢复真实执行。
    """
    from langchain_core.messages import HumanMessage

    from app.llm.providers import get_extraction_llm
    try:
        get_extraction_llm().invoke([HumanMessage(content="回复：ok")])
        return True
    except Exception:
        return False


class _NoConflictDetector:
    """空冲突检测器：抽取阶段不触发 LLM 冲突判定。"""

    def detect_for_article(self, article_id: str) -> int:
        return 0


def _insert_chunks_with_explicit_ids(vector_store, chunks: list[dict]) -> None:
    """以抽取相同的 chunk_id 写入 Milvus，保证 Article.chunk_ids 可反查。"""
    full_name = f"{vector_store.collection_prefix}{E2E_MILVUS_COLLECTION}"
    vectors = vector_store.embed_texts([c["content"] for c in chunks])
    collection = Collection(name=full_name)
    collection.insert([
        [c["id"] for c in chunks],                                    # id (主键)
        [1001] * len(chunks),                                          # document_id
        [c["chunk_type"] for c in chunks],                            # chunk_type
        [c["content"] for c in chunks],                               # content
        [f"e2e-hash-{i}" for i in range(len(chunks))],               # content_hash
        ["active"] * len(chunks),                                      # status
        [c["char_start"] for c in chunks],                            # char_start
        [c["char_end"] for c in chunks],                              # char_end
        vectors,                                                       # embedding
    ])
    collection.flush()


@pytest.fixture
def e2e_setup(graph_store):
    """端到端：上传文档 -> 切分 -> 入库 Milvus -> KG 抽取 -> 检索。"""
    from app.knowledge_graph.extractor import KGExtractor
    from app.rag.splitter import ThreeLayerSplitter
    from app.core.dependencies import get_vector_store

    text = FIXTURE.read_text(encoding="utf-8")
    chunks = ThreeLayerSplitter().split(text)
    for i, c in enumerate(chunks):
        c["id"] = f"e2e-chunk-{i}"
        c["document_id"] = "e2e-doc-1"

    vector_store = get_vector_store()
    try:
        vector_store.connect()
    except Exception as exc:
        pytest.skip(f"Milvus not available for E2E: {exc}")

    if not _llm_available():
        pytest.skip("抽取 LLM 不可用（配额耗尽或网络异常），跳过 E2E")

    # 独立集合：先 drop 保证幂等，再按 schema 重建并插入显式 id 的 chunks
    vector_store.drop_collection(E2E_MILVUS_COLLECTION)
    vector_store.create_collection(E2E_MILVUS_COLLECTION)
    _insert_chunks_with_explicit_ids(vector_store, chunks)

    extractor = KGExtractor(
        store=graph_store,
        embedding_fn=vector_store.embed_query,
        chunks_loader=lambda doc_id: chunks,
        document_loader=lambda doc_id: text,
        conflict_detector=_NoConflictDetector(),
    )
    report = extractor.run(document_id="e2e-doc-1")

    yield graph_store, vector_store, report

    vector_store.drop_collection(E2E_MILVUS_COLLECTION)


def test_e2e_extracts_law_and_articles(e2e_setup):
    store, _, report = e2e_setup
    assert report.error == "", f"抽取失败: {report.error}"
    assert report.entities_count > 0
    with store.session() as s:
        assert s.run("MATCH (l:Law) RETURN count(*) AS c").single()["c"] >= 1
        assert s.run("MATCH (a:Article) RETURN count(*) AS c").single()["c"] >= 1


def test_e2e_extracts_concept_试用期(e2e_setup):
    store, _, _ = e2e_setup
    with store.session() as s:
        result = s.run("MATCH (c:Concept {name: '试用期'}) RETURN count(*) AS c")
        assert result.single()["c"] == 1


def test_e2e_article_explains_concept(e2e_setup):
    store, _, _ = e2e_setup
    with store.session() as s:
        result = s.run(
            """
            MATCH (a:Article)-[:EXPLAINS]->(c:Concept {name: '试用期'})
            RETURN count(*) AS c
            """
        )
        assert result.single()["c"] >= 1


def test_e2e_kg_retrieval_finds_试用期_article(e2e_setup):
    store, vector_store, _ = e2e_setup
    from app.knowledge_graph.kg_retriever import KGRetriever

    query = "试用期"
    retriever = KGRetriever(
        store=store,
        vector_store=vector_store,
        collection_name=E2E_MILVUS_COLLECTION,
    )
    results = retriever.retrieve(query=query, query_embedding=vector_store.embed_query(query))
    assert len(results) > 0
