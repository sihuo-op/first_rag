"""KG 检索路径：query 实体抽取 + 多跳 Cypher + 反查 chunks。

作为 HybridRetriever 的第三路检索源（Task 13 接入 RRF 融合）。

查询路径零 LLM 调用（全局约束）：
- Concept：Neo4j 向量索引 concept_embedding + query embedding 相似度匹配；
- Party / Region：词表精确匹配（O(1)），仅作准入门槛信号。

失败回退（全局约束）：任何 KG 异常都在 retrieve() 内部吞掉并返回 []，
HybridRetriever 的 RRF 融合自动降级为 dense+sparse 两路，KG 绝不拖垮主链路。
"""
import logging

from app.core.observability import get_tracer
from app.knowledge_graph.exceptions import KGError

logger = logging.getLogger(__name__)

tracer = get_tracer("kg.retrieve")

# 常用词表（O(1) 匹配，扩容时改这里）
PARTY_WORDS = ["用人单位", "劳动者", "工会", "劳动行政部门", "用人单位一方", "劳动者一方"]
REGION_WORDS = ["全国", "北京", "上海", "天津", "重庆", "广东", "江苏", "浙江", "山东", "四川"]

# Concept 向量检索取 top-k（阈值过滤前的候选数）
_CONCEPT_TOP_K = 10


class KGRetriever:
    """KG 检索器：Concept 向量匹配 -> 多跳 Cypher 找 Article -> 反查 chunks。

    Args:
        store: Neo4jStore（复用 Task 8 的 find_articles_by_concept）
        vector_store: MilvusStore（按 chunk_id 反查内容）
        similarity_threshold: Concept 向量相似度阈值（查询路径专用，
            区别于 entity_resolver 的 0.92 合并阈值），对应配置
            KG_CONCEPT_SIMILARITY_THRESHOLD
        max_depth: Concept -> Article 多跳深度，对应配置 KG_MULTI_HOP_DEPTH
        collection_name: chunk 反查的 Milvus 集合名
    """

    def __init__(
        self,
        store,
        vector_store,
        similarity_threshold: float = 0.7,
        max_depth: int = 2,
        collection_name: str = "chunks",
    ):
        self.store = store
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.collection_name = collection_name

    def retrieve(self, query: str, query_embedding: list[float]) -> list[dict]:
        """返回 chunks 列表，每个含 chunk_id/content/chunk_type/kg_score。

        失败回退：任何异常（含 KGError）内部吞掉，返回 []。
        """
        try:
            with tracer.start_as_current_span("kg.retrieve") as span:
                span.set_attribute("query", query[:200])
                return self._retrieve_impl(query, query_embedding, span)
        except KGError as e:
            logger.warning("KG 检索失败（KGError），降级为空结果: %s", e)
            self._record_error_span(query, e)
            return []
        except Exception as e:
            logger.warning("KG 检索失败（意外异常），降级为空结果: %s", e)
            self._record_error_span(query, e)
            return []

    def _retrieve_impl(self, query: str, query_embedding: list[float], span) -> list[dict]:
        # Step 1: query 实体抽取（零 LLM）
        with tracer.start_as_current_span("kg.retrieve.entity_extract"):
            concept_ids = self._extract_concepts(query_embedding)
            parties, regions = self._extract_party_region(query)
            span.set_attribute("concept.count", len(concept_ids))
            span.set_attribute("party.count", len(parties))
            span.set_attribute("region.count", len(regions))

        # 三个信号全空：query 与 KG 无关，直接短路
        if not concept_ids and not parties and not regions:
            return []

        # Step 2: 多跳 Cypher 查 Article（复用 Task 8）
        with tracer.start_as_current_span("kg.retrieve.cypher_query"):
            articles = self.store.find_articles_by_concept(concept_ids, max_depth=self.max_depth)
            span.set_attribute("article.count", len(articles))

        if not articles:
            return []

        # Step 3: 反查 chunks 并归一化 kg_score
        with tracer.start_as_current_span("kg.retrieve.chunk_fetch"):
            chunks = self._fetch_chunks(articles)
            span.set_attribute("chunk.count", len(chunks))

        return chunks

    def _fetch_chunks(self, articles: list[dict]) -> list[dict]:
        """Article -> chunk 映射，kg_score = concept_hit_count / max_hit（归一化）。

        同一 chunk 被多个 Article 引用时去重，保留最高分。
        """
        max_hit = max((a.get("concept_hit_count", 0) for a in articles), default=0) or 1
        by_chunk: dict[str, dict] = {}
        for art in articles:
            kg_score = float(art.get("concept_hit_count", 0)) / max_hit
            chunk_ids = art.get("chunk_ids") or []
            if not chunk_ids:
                continue
            fetched = self.vector_store.get_chunks_by_ids(self.collection_name, chunk_ids)
            for ch in fetched:
                chunk_id = ch.get("id")
                if chunk_id is None:
                    continue
                if chunk_id in by_chunk:
                    if kg_score > by_chunk[chunk_id]["kg_score"]:
                        by_chunk[chunk_id]["kg_score"] = kg_score
                    continue
                by_chunk[chunk_id] = {
                    "chunk_id": chunk_id,
                    "id": chunk_id,  # 兼容 RRF 融合按 id 合并的既有约定
                    "content": ch.get("content", ""),
                    "chunk_type": ch.get("chunk_type", "small"),
                    "document_id": ch.get("document_id"),
                    "kg_score": kg_score,
                }
        return list(by_chunk.values())

    def _extract_concepts(self, query_embedding: list[float]) -> list[str]:
        """Concept 向量检索：复用 concept_embedding 索引，零 LLM。"""
        with self.store.session() as s:
            result = s.run(
                f"""
                CALL db.index.vector.queryNodes('concept_embedding', {_CONCEPT_TOP_K}, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.id AS id
                """,
                embedding=query_embedding,
                threshold=self.similarity_threshold,
            )
            return [r["id"] for r in result if r["id"] is not None]

    def _extract_party_region(self, query: str) -> tuple[list[str], list[str]]:
        """Party/Region 词表精确匹配（零 LLM、零向量）。"""
        parties = [w for w in PARTY_WORDS if w in query]
        regions = [w for w in REGION_WORDS if w in query]
        return parties, regions

    @staticmethod
    def _record_error_span(query: str, exc: Exception) -> None:
        with tracer.start_as_current_span("kg.retrieve.error") as span:
            span.set_attribute("query", query[:200])
            span.set_attribute("error.type", type(exc).__name__)
            span.record_exception(exc)
