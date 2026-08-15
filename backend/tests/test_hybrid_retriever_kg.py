"""HybridRetriever 三路 RRF 融合单测（Task 13）：全部 mock，不依赖真实数据库。

覆盖：
- KG 结果作为第三路参与 RRF 融合（与 dense/sparse 同一去重/晋升管道）；
- KG 失败回退两路（KGRetriever 契约上不抛异常，这里 mock 抛异常验证兜底）；
- KG 返回 [] 时行为与两路一致；
- 未注入 kg_retriever（KG_ENABLED=False）时完全走旧两路逻辑；
- 查询向量只计算一次，dense 与 KG 共享；
- 稀疏结果没有 id 字段（metadata 只有 document_id/chunk_type），
  RRF 合并必须按 content 键合并而不是按 id（回归防护）。
"""
from unittest.mock import MagicMock, patch

from app.rag.retriever import HybridRetriever


def _make_retriever(vector_store, sparse_retriever, kg_retriever):
    """构造不加载重排序模型的 HybridRetriever。"""
    with patch("app.rag.retriever.Reranker"):
        return HybridRetriever(
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            use_reranker=False,
            kg_retriever=kg_retriever,
        )


def _make_vector_store(dense_results):
    vector_store = MagicMock()
    vector_store.connect.return_value = None
    vector_store.embed_query.return_value = [0.1] * 1024
    vector_store.search_vectors.return_value = dense_results
    return vector_store


# ---------- 三路融合：KG 结果进入最终结果 ----------

def test_kg_results_merged_into_rrf():
    vector_store = _make_vector_store([
        {"id": "c1", "content": "dense", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        {"id": "c2", "content": "sparse", "chunk_type": "small", "sparse_score": 0.8},
    ]
    kg_retriever = MagicMock()
    kg_retriever.retrieve.return_value = [
        {"id": "c3", "content": "kg", "chunk_type": "small", "kg_score": 0.7},
    ]

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    chunks, debug = retriever.retrieve("query", top_k=5)

    # Should include chunks from all 3 paths
    chunk_ids = {c.get("id") for c in chunks}
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" in chunk_ids
    # debug_info 记录 KG 路径
    assert debug["kg_results"] == 1
    assert any(s["step"] == "kg_search" for s in debug["steps"])


# ---------- KG 失败 -> 回退两路 ----------

def test_kg_failure_falls_back_to_two_paths():
    vector_store = _make_vector_store([
        {"id": "c1", "content": "dense", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        {"id": "c2", "content": "sparse", "chunk_type": "small", "sparse_score": 0.8},
    ]
    kg_retriever = MagicMock()
    from app.knowledge_graph.exceptions import KGQueryError
    kg_retriever.retrieve.side_effect = KGQueryError("Neo4j down")

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    chunks, debug = retriever.retrieve("query", top_k=5)

    chunk_ids = {c.get("id") for c in chunks}
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    # c3 (KG) not included due to failure
    assert "c3" not in chunk_ids
    assert debug["kg_results"] == 0


# ---------- 三路同 chunk：RRF 分数累加 ----------

def test_rrf_score_sums_across_three_paths():
    shared = [
        {"id": "c1", "content": "shared content", "chunk_type": "small", "score": 0.9},
    ]
    vector_store = _make_vector_store(shared)
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        {"id": "c1", "content": "shared content", "chunk_type": "small", "sparse_score": 0.8},
    ]
    kg_retriever = MagicMock()
    kg_retriever.retrieve.return_value = [
        {"id": "c1", "chunk_id": "c1", "content": "shared content",
         "chunk_type": "small", "kg_score": 0.7, "document_id": 1},
    ]

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    chunks, _ = retriever.retrieve("query", top_k=5)

    # 同一 chunk（同内容）只出现一次，且 RRF 分数为三路之和
    assert len(chunks) == 1
    expected = 3 * (1 / (60 + 1))  # 三路各 rank=1，rrf_k=60
    assert abs(chunks[0]["rrf_score"] - expected) < 1e-9


# ---------- KG 返回 [] -> 行为与两路一致 ----------

def test_kg_empty_results_behaves_as_two_way():
    vector_store = _make_vector_store([
        {"id": "c1", "content": "dense", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        {"id": "c2", "content": "sparse", "chunk_type": "small", "sparse_score": 0.8},
    ]
    kg_retriever = MagicMock()
    kg_retriever.retrieve.return_value = []  # 无匹配

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    chunks, debug = retriever.retrieve("query", top_k=5)

    chunk_ids = {c.get("id") for c in chunks}
    assert chunk_ids == {"c1", "c2"}
    assert debug["kg_results"] == 0
    kg_steps = [s for s in debug["steps"] if s["step"] == "kg_search"]
    assert len(kg_steps) == 1 and kg_steps[0]["count"] == 0


# ---------- 未注入 kg_retriever（KG 关闭）-> 完全两路 ----------

def test_kg_retriever_none_disables_kg_path():
    vector_store = _make_vector_store([
        {"id": "c1", "content": "dense", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        {"id": "c2", "content": "sparse", "chunk_type": "small", "sparse_score": 0.8},
    ]

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever=None)
    chunks, debug = retriever.retrieve("query", top_k=5)

    chunk_ids = {c.get("id") for c in chunks}
    assert chunk_ids == {"c1", "c2"}
    # debug_info 与旧两路一致：不出现 KG 痕迹
    assert "kg_results" not in debug
    assert all(s["step"] != "kg_search" for s in debug["steps"])


# ---------- 查询向量只算一次，dense 与 KG 共享 ----------

def test_query_embedding_computed_once_and_shared_with_kg():
    vector_store = _make_vector_store([
        {"id": "c1", "content": "dense", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = []
    kg_retriever = MagicMock()
    kg_retriever.retrieve.return_value = []

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    retriever.retrieve("query", top_k=5)

    vector_store.embed_query.assert_called_once()
    kg_retriever.retrieve.assert_called_once_with(
        query="query", query_embedding=vector_store.embed_query.return_value
    )


# ---------- 稀疏结果无 id：合并键必须是 content（回归防护） ----------

def test_sparse_results_without_id_still_merged_in_three_way():
    """真实 SparseRetriever 结果没有 id 字段（metadata 只有 document_id/chunk_type），
    RRF 按 content 合并时不能丢掉稀疏路结果。"""
    vector_store = _make_vector_store([
        {"id": "c1", "content": "甲条内容", "chunk_type": "small", "score": 0.9},
    ])
    sparse_retriever = MagicMock()
    sparse_retriever.retrieve.return_value = [
        # 模拟真实 BM25 结果形状：无 id，chunk_type 在 metadata 里
        {"content": "乙条内容", "metadata": {"document_id": 1, "chunk_type": "small"}, "score": 0.5},
    ]
    kg_retriever = MagicMock()
    kg_retriever.retrieve.return_value = [
        {"id": "c3", "chunk_id": "c3", "content": "丙条内容",
         "chunk_type": "small", "kg_score": 0.7, "document_id": 1},
    ]

    retriever = _make_retriever(vector_store, sparse_retriever, kg_retriever)
    chunks, _ = retriever.retrieve("query", top_k=5)

    contents = {c.get("content") for c in chunks}
    assert contents == {"甲条内容", "乙条内容", "丙条内容"}


# ---------- dependencies.get_retriever 的 KG 装配 ----------

def _reset_and_patch_deps():
    """重置 get_retriever 单例并 mock 掉所有外部依赖，返回各 mock。"""
    import app.core.dependencies as deps

    deps._retriever_instance = None
    patched = {
        "deps": deps,
        "get_vector_store": patch("app.core.dependencies.get_vector_store").start(),
        "get_sparse_retriever": patch("app.core.dependencies.get_sparse_retriever").start(),
        "hybrid": patch("app.core.dependencies.HybridRetriever").start(),
        "kg_cls": patch("app.knowledge_graph.kg_retriever.KGRetriever").start(),
        "graph_store": patch("app.knowledge_graph.graph_store.get_graph_store").start(),
    }
    return patched


def _stop_patches(patched):
    patch.stopall()
    patched["deps"]._retriever_instance = None


def test_get_retriever_injects_kg_when_enabled():
    patched = _reset_and_patch_deps()
    try:
        patched["deps"].get_retriever()
        # KG_ENABLED 默认 True：KGRetriever 被构造并注入 HybridRetriever
        patched["kg_cls"].assert_called_once()
        kg_kwargs = patched["kg_cls"].call_args.kwargs
        assert kg_kwargs["store"] is patched["graph_store"].return_value
        kwargs = patched["hybrid"].call_args.kwargs
        assert kwargs["kg_retriever"] is patched["kg_cls"].return_value
    finally:
        _stop_patches(patched)


def test_get_retriever_skips_kg_when_disabled(monkeypatch):
    patched = _reset_and_patch_deps()
    try:
        monkeypatch.setattr(patched["deps"]._settings, "KG_ENABLED", False)
        patched["deps"].get_retriever()
        patched["kg_cls"].assert_not_called()
        assert patched["hybrid"].call_args.kwargs["kg_retriever"] is None
    finally:
        _stop_patches(patched)


def test_get_retriever_neo4j_down_falls_back_to_two_way(monkeypatch):
    from app.knowledge_graph.exceptions import KGConnectionError

    patched = _reset_and_patch_deps()
    try:
        monkeypatch.setattr(patched["deps"]._settings, "KG_ENABLED", True)
        patched["graph_store"].side_effect = KGConnectionError("Neo4j down")
        patched["deps"].get_retriever()
        # Neo4j 连不上：不抛异常，降级为 kg_retriever=None 的两路检索
        assert patched["hybrid"].call_args.kwargs["kg_retriever"] is None
    finally:
        _stop_patches(patched)
