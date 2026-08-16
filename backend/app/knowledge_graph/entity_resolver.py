"""实体合并：Concept 三级 / Party 两级 / Law/Region 直接 MERGE。

- Concept 三级：精确名称 -> 别名 -> embedding 相似 + LLM 二次确认。
- Party 两级：精确名称 -> 别名（不做 embedding 模糊匹配，主体名称误合并代价高）。
- Law/Region/Document 等其他类型原样透传（existing_node_id=None），由调用方直接 MERGE。

LLM 确认通过 ``invoke_llm_threadsafe`` 走线程安全包装；LLM / embedding 异常时
保守处理（视为不同概念），降级为新建节点。
"""
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from app.core.observability import get_tracer
from app.knowledge_graph.llm_extractor import ExtractedEntity
from app.knowledge_graph.schema import NodeType
from app.llm.providers import invoke_llm_threadsafe

tracer = get_tracer("kg.extract")

EMBEDDING_SIMILARITY_THRESHOLD = 0.92


@dataclass
class ResolvedEntity:
    node_type: str  # "Concept" / "Party" / ...
    name: str
    existing_node_id: str | None = None  # None 表示新建
    aliases_to_add: list[str] = field(default_factory=list)
    source_chunk_id: str = ""


class EntityResolver:
    def __init__(self, store, embedding_fn, llm):
        """
        store: Neo4jStore
        embedding_fn: callable(str) -> list[float]，用于 Concept 模糊匹配
        llm: 用于"X vs Y 是否同一概念"的二次确认
        """
        self.store = store
        self.embedding_fn = embedding_fn
        self.llm = llm

    def resolve(self, entities: list[ExtractedEntity], source_chunk_id: str) -> list[ResolvedEntity]:
        with tracer.start_as_current_span("kg.extract.resolve") as span:
            span.set_attribute("chunk.id", source_chunk_id)
            span.set_attribute("entity.count", len(entities))

            results: list[ResolvedEntity] = []
            for ent in entities:
                if ent.type == NodeType.CONCEPT.value:
                    results.append(self._resolve_concept(ent, source_chunk_id))
                elif ent.type == NodeType.PARTY.value:
                    results.append(self._resolve_party(ent, source_chunk_id))
                else:
                    # Law/Region/Document: 调用方直接 MERGE
                    results.append(ResolvedEntity(
                        node_type=ent.type, name=ent.name,
                        existing_node_id=None, source_chunk_id=source_chunk_id,
                    ))

            merged = sum(1 for r in results if r.existing_node_id is not None)
            span.set_attribute("resolve.merged_count", merged)
            span.set_attribute("resolve.new_count", len(results) - merged)
            return results

    def _find_by_name(self, label: str, name: str) -> dict | None:
        with self.store.session() as s:
            result = s.run(
                f"MATCH (n:{label}) WHERE n.name = $name RETURN n LIMIT 1",
                name=name,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def _find_by_alias(self, label: str, name: str) -> dict | None:
        with self.store.session() as s:
            result = s.run(
                f"MATCH (n:{label}) WHERE $name IN n.aliases RETURN n LIMIT 1",
                name=name,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def _resolve_concept(self, ent: ExtractedEntity, source_chunk_id: str) -> ResolvedEntity:
        # Level 1: exact name
        existing = self._find_by_name(NodeType.CONCEPT.value, ent.name)
        if existing:
            return ResolvedEntity(
                node_type=NodeType.CONCEPT.value, name=ent.name,
                existing_node_id=existing["id"],
                aliases_to_add=[a for a in ent.aliases if a not in (existing.get("aliases") or [])],
                source_chunk_id=source_chunk_id,
            )

        # Level 2: alias match
        existing = self._find_by_alias(NodeType.CONCEPT.value, ent.name)
        if existing:
            return ResolvedEntity(
                node_type=NodeType.CONCEPT.value, name=ent.name,
                existing_node_id=existing["id"],
                aliases_to_add=[ent.name] if ent.name != existing.get("name") else [],
                source_chunk_id=source_chunk_id,
            )

        # Level 3: embedding similarity + LLM verify
        candidate = self._find_by_embedding_similarity(ent.name)
        if candidate and self._llm_confirm_same_concept(ent.name, candidate["name"]):
            return ResolvedEntity(
                node_type=NodeType.CONCEPT.value, name=ent.name,
                existing_node_id=candidate["id"],
                aliases_to_add=[ent.name],
                source_chunk_id=source_chunk_id,
            )

        # No match: new node
        return ResolvedEntity(
            node_type=NodeType.CONCEPT.value, name=ent.name,
            existing_node_id=None,
            aliases_to_add=ent.aliases,
            source_chunk_id=source_chunk_id,
        )

    def _resolve_party(self, ent: ExtractedEntity, source_chunk_id: str) -> ResolvedEntity:
        # Level 1: exact name
        existing = self._find_by_name(NodeType.PARTY.value, ent.name)
        if existing:
            return ResolvedEntity(
                node_type=NodeType.PARTY.value, name=ent.name,
                existing_node_id=existing["id"],
                aliases_to_add=[a for a in ent.aliases if a not in (existing.get("aliases") or [])],
                source_chunk_id=source_chunk_id,
            )

        # Level 2: alias match
        existing = self._find_by_alias(NodeType.PARTY.value, ent.name)
        if existing:
            return ResolvedEntity(
                node_type=NodeType.PARTY.value, name=ent.name,
                existing_node_id=existing["id"],
                aliases_to_add=[ent.name] if ent.name != existing.get("name") else [],
                source_chunk_id=source_chunk_id,
            )

        # No level 3 for Party
        return ResolvedEntity(
            node_type=NodeType.PARTY.value, name=ent.name,
            existing_node_id=None,
            aliases_to_add=ent.aliases,
            source_chunk_id=source_chunk_id,
        )

    def _find_by_embedding_similarity(self, name: str) -> dict | None:
        try:
            embedding = self.embedding_fn(name)
        except Exception:  # noqa: BLE001 - embedding 模型异常不应中断合并
            return None
        with self.store.session() as s:
            result = s.run(
                """
                CALL db.index.vector.queryNodes('concept_embedding', 5, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node, score
                ORDER BY score DESC
                LIMIT 1
                """,
                embedding=embedding,
                threshold=EMBEDDING_SIMILARITY_THRESHOLD,
            )
            record = result.single()
            return dict(record["node"]) if record else None

    def _llm_confirm_same_concept(self, name_a: str, name_b: str) -> bool:
        prompt = (
            "判断以下两个法律概念是否指同一概念。只回答 true 或 false。"
            f"\n概念A：{name_a}\n概念B：{name_b}\n回答："
        )
        try:
            response = invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])
            return "true" in response.content.strip().lower()
        except Exception:  # noqa: BLE001 - LLM 异常时保守视为不同概念
            return False
