"""EntityResolver 单测：全部 mock Neo4j，不依赖真实数据库。

mock 约定：
- session().run(...).single() 依次消费 ``_make_store(single_results)`` 传入的结果序列。
  Concept 查询顺序：exact name -> alias -> embedding；Party：exact name -> alias。
- 查询结果用 ``{"n": {...}}`` 形状，与真实 Neo4j Record 的 ``record["n"]`` 行为一致；
  embedding 查询用 ``{"node": {...}}``（Cypher 中 ``YIELD node, score``）。
"""
from unittest.mock import MagicMock, patch

from app.knowledge_graph.entity_resolver import (
    EMBEDDING_SIMILARITY_THRESHOLD,
    EntityResolver,
    ResolvedEntity,
)
from app.knowledge_graph.llm_extractor import ExtractedEntity


class _Ctx:
    """模拟 driver.session() 返回的上下文管理器。"""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *args):
        return False


def _make_store(single_results):
    """构造 mock Neo4jStore：每次 run().single() 依次消费 single_results。"""
    store = MagicMock()
    session = MagicMock()
    session.run.return_value.single.side_effect = list(single_results)
    store.session.return_value = _Ctx(session)
    return store, session


def _llm_response(text: str):
    resp = MagicMock()
    resp.content = text
    return resp


# ---------- Level 1: Concept 精确名称匹配 ----------

def test_concept_exact_match_merges():
    store, _ = _make_store([
        {"n": {"id": "concept-1", "name": "试用期", "aliases": ["试用期间"]}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="试用期", aliases=[])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert len(result) == 1
    assert result[0].existing_node_id == "concept-1"
    assert result[0].aliases_to_add == []  # no new aliases


def test_concept_exact_match_filters_known_aliases():
    """实体自带别名中，已在目标节点 aliases 里的不重复添加。"""
    store, _ = _make_store([
        {"n": {"id": "concept-1", "name": "试用期", "aliases": ["试用期间"]}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="试用期", aliases=["试用期间", "试用条款"])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id == "concept-1"
    assert result[0].aliases_to_add == ["试用条款"]


def test_concept_exact_match_handles_null_aliases():
    """已有节点 aliases 为 null 时不抛 TypeError。"""
    store, _ = _make_store([
        {"n": {"id": "concept-1", "name": "试用期", "aliases": None}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="试用期", aliases=["试用期间"])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id == "concept-1"
    assert result[0].aliases_to_add == ["试用期间"]


# ---------- Level 2: Concept 别名匹配 ----------

def test_concept_alias_match_merges():
    """exact miss -> alias hit：合并到已有节点，实体名作为新别名回写。"""
    store, _ = _make_store([
        None,  # exact name miss
        {"n": {"id": "concept-1", "name": "试用期", "aliases": ["试用期间"]}},  # alias hit
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="试用期间", aliases=[])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id == "concept-1"
    assert "试用期间" in result[0].aliases_to_add or result[0].aliases_to_add == []


# ---------- Level 3: Concept embedding + LLM 二次确认 ----------

def test_concept_embedding_match_llm_confirmed_merges():
    """exact/alias miss -> embedding 候选 + LLM 确认同义 -> 合并。"""
    store, session = _make_store([
        None,  # exact name miss
        None,  # alias miss
        {"node": {"id": "concept-9", "name": "医疗期", "aliases": []}},  # vector hit
    ])
    embedding_fn = MagicMock(return_value=[0.1] * 1024)
    resolver = EntityResolver(store, embedding_fn=embedding_fn, llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="病假", aliases=[])]

    with patch(
        "app.knowledge_graph.entity_resolver.invoke_llm_threadsafe",
        return_value=_llm_response("true"),
    ) as mock_invoke:
        result = resolver.resolve(entities, source_chunk_id="chunk-1")

    assert result[0].existing_node_id == "concept-9"
    assert result[0].aliases_to_add == ["病假"]
    embedding_fn.assert_called_once_with("病假")
    # 第 3 次查询走向量索引，且阈值/向量参数正确
    vector_call = session.run.call_args_list[2]
    assert "db.index.vector.queryNodes" in vector_call.args[0]
    assert vector_call.kwargs["threshold"] == EMBEDDING_SIMILARITY_THRESHOLD
    assert vector_call.kwargs["embedding"] == [0.1] * 1024
    # LLM 收到两个概念名的确认 prompt
    prompt = mock_invoke.call_args.args[1][0].content
    assert "病假" in prompt and "医疗期" in prompt


def test_concept_embedding_match_llm_rejected_creates_new():
    """embedding 候选命中但 LLM 判定不同概念 -> 新建节点。"""
    store, _ = _make_store([
        None,  # exact name miss
        None,  # alias miss
        {"node": {"id": "concept-9", "name": "医疗期", "aliases": []}},  # vector hit
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="加班费", aliases=["加班工资"])]

    with patch(
        "app.knowledge_graph.entity_resolver.invoke_llm_threadsafe",
        return_value=_llm_response("false"),
    ):
        result = resolver.resolve(entities, source_chunk_id="chunk-1")

    assert result[0].existing_node_id is None
    assert result[0].aliases_to_add == ["加班工资"]


def test_concept_all_levels_miss_creates_new_without_llm():
    """三级全部未命中 -> 新建节点，embedding 未命中时不调 LLM。"""
    store, _ = _make_store([
        None,  # exact name miss
        None,  # alias miss
        None,  # vector miss
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="经济补偿", aliases=["经济补偿金"])]

    with patch("app.knowledge_graph.entity_resolver.invoke_llm_threadsafe") as mock_invoke:
        result = resolver.resolve(entities, source_chunk_id="chunk-1")

    assert result[0].existing_node_id is None
    assert result[0].aliases_to_add == ["经济补偿金"]
    mock_invoke.assert_not_called()


def test_concept_embedding_failure_falls_back_to_new_node():
    """embedding_fn 抛异常时降级为新建节点，而不是崩溃。"""
    store, _ = _make_store([
        None,  # exact name miss
        None,  # alias miss
    ])
    embedding_fn = MagicMock(side_effect=RuntimeError("model not loaded"))
    resolver = EntityResolver(store, embedding_fn=embedding_fn, llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="年休假", aliases=[])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id is None
    assert result[0].name == "年休假"


def test_concept_llm_failure_rejected_creates_new():
    """LLM 确认调用异常时保守处理：视为不同概念，新建节点。"""
    store, _ = _make_store([
        None,  # exact name miss
        None,  # alias miss
        {"node": {"id": "concept-9", "name": "医疗期", "aliases": []}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Concept", name="病假", aliases=[])]

    with patch(
        "app.knowledge_graph.entity_resolver.invoke_llm_threadsafe",
        side_effect=RuntimeError("llm down"),
    ):
        result = resolver.resolve(entities, source_chunk_id="chunk-1")

    assert result[0].existing_node_id is None


# ---------- Party：两级合并，无 embedding ----------

def test_party_exact_match_merges():
    store, _ = _make_store([
        {"n": {"id": "party-1", "name": "用人单位", "aliases": []}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Party", name="用人单位", aliases=["企业"])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id == "party-1"
    assert result[0].aliases_to_add == ["企业"]


def test_party_alias_match_merges():
    store, _ = _make_store([
        None,  # exact name miss
        {"n": {"id": "party-1", "name": "用人单位", "aliases": ["雇主"]}},
    ])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Party", name="雇主", aliases=[])]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id == "party-1"
    assert "雇主" in result[0].aliases_to_add or result[0].aliases_to_add == []


def test_party_only_two_levels_no_embedding():
    """Party 不走 embedding 模糊匹配。"""
    store, _ = _make_store([
        None,  # exact name miss
        None,  # alias miss
    ])
    embedding_fn = MagicMock()
    resolver = EntityResolver(store, embedding_fn=embedding_fn, llm=MagicMock())
    entities = [ExtractedEntity(type="Party", name="新主体")]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert result[0].existing_node_id is None  # new node
    embedding_fn.assert_not_called()


# ---------- 其他类型 / 边界 ----------

def test_non_concept_party_types_pass_through_without_query():
    """Law/Region/Document 等由调用方直接 MERGE：原样返回且不查 Neo4j。"""
    store, _ = _make_store([])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    entities = [ExtractedEntity(type="Region", name="北京市"), ExtractedEntity(type="Law", name="劳动合同法")]
    result = resolver.resolve(entities, source_chunk_id="chunk-1")
    assert len(result) == 2
    assert all(r.existing_node_id is None for r in result)
    assert [r.node_type for r in result] == ["Region", "Law"]
    assert [r.name for r in result] == ["北京市", "劳动合同法"]
    store.session.assert_not_called()


def test_empty_entities_returns_empty():
    store, _ = _make_store([])
    resolver = EntityResolver(store, embedding_fn=MagicMock(), llm=MagicMock())
    assert resolver.resolve([], source_chunk_id="chunk-1") == []


def test_resolved_entity_defaults():
    """ResolvedEntity 默认值：existing_node_id=None 表示新建。"""
    r = ResolvedEntity(node_type="Concept", name="试用期")
    assert r.existing_node_id is None
    assert r.aliases_to_add == []
    assert r.source_chunk_id == ""
