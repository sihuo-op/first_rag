"""Neo4j 连接与 Cypher 读写封装。"""
import re

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from app.core.observability import get_tracer
from app.knowledge_graph.exceptions import KGConnectionError, KGQueryError
from app.knowledge_graph.schema import ArticleNode, DocumentNode, EdgeType, LawNode

tracer = get_tracer("kg.store")

CONSTRAINTS_CYPHER = [
    "CREATE CONSTRAINT law_unique IF NOT EXISTS FOR (n:Law) REQUIRE (n.name, n.region_id) IS UNIQUE",
    "CREATE CONSTRAINT article_unique IF NOT EXISTS FOR (n:Article) REQUIRE (n.law_id, n.article_no) IS UNIQUE",
    "CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT party_unique IF NOT EXISTS FOR (n:Party) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT region_unique IF NOT EXISTS FOR (n:Region) REQUIRE n.name IS UNIQUE",
]

VECTOR_INDEX_CYPHER = """
CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
  FOR (n:Concept) ON (n.embedding)
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }}
"""

# ---------------------------------------------------------------------------
# 写入重试：最多重试 3 次（共 4 次尝试），指数退避 100ms / 500ms / 2s。
# 仅重试瞬时错误；约束冲突等永久错误直接抛出。
# ---------------------------------------------------------------------------
_WRITE_RETRY_WAITS = (0.1, 0.5, 2.0)
_RETRIABLE_EXCEPTIONS = (ServiceUnavailable, SessionExpired, TransientError)

# merge_relation 的 edge_type / prop key 会拼进 Cypher，必须校验防注入。
_VALID_EDGE_TYPES = frozenset(e.value for e in EdgeType)
_PROP_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _write_retry_wait(retry_state) -> float:
    """按全局约定返回固定退避序列 100ms/500ms/2s。"""
    idx = min(retry_state.attempt_number - 1, len(_WRITE_RETRY_WAITS) - 1)
    return _WRITE_RETRY_WAITS[idx]


def _retry_write():
    return retry(
        stop=stop_after_attempt(len(_WRITE_RETRY_WAITS) + 1),  # 1 次初始 + 3 次重试
        wait=_write_retry_wait,
        retry=retry_if_exception_type(_RETRIABLE_EXCEPTIONS),
        reraise=True,
    )


# Cypher 纯净写法（不依赖 APOC，testcontainers 默认无 apoc）：
# 列表合并 + 去重。O(n^2)，但 aliases/chunk_ids 都很短。
_DEDUPE_ALIASES = (
    "reduce(acc = [], a IN coalesce(n.aliases, []) + $aliases | "
    "CASE WHEN a IS NULL OR a IN acc THEN acc ELSE acc + a END)"
)
_DEDUPE_CHUNK_IDS = (
    "reduce(acc = [], c IN coalesce(n.source_chunk_ids, []) + $source_chunk_ids | "
    "CASE WHEN c IS NULL OR c IN acc THEN acc ELSE acc + c END)"
)


class _Neo4jStoreWriteMixin:
    """分离写方法到 mixin，避免 graph_store.py 过大。

    所有 upsert 基于 MERGE，天然幂等。aliases / source_chunk_ids 采用
    去重合并（Task 7 entity_resolver 的 Level-2 别名合并会重复传回已存在
    的别名，这里绝不能朴素追加）。
    """

    @_retry_write()
    def upsert_law(self, law: LawNode) -> str:
        with tracer.start_as_current_span("kg.store.upsert_law") as span:
            span.set_attribute("node.id", law.id)
            span.set_attribute("node.type", "Law")
            with self.session() as s:
                s.run(
                    """
                    MERGE (n:Law {id: $id})
                    SET n.name = $name, n.level = $level,
                        n.effective_date = $effective_date, n.issuer = $issuer,
                        n.region_id = $region_id
                    RETURN n.id AS id
                    """,
                    id=law.id, name=law.name, level=law.level,
                    effective_date=law.effective_date, issuer=law.issuer,
                    region_id=law.region_id,
                ).single()
            return law.id

    @_retry_write()
    def upsert_article(self, article: ArticleNode) -> str:
        with tracer.start_as_current_span("kg.store.upsert_article") as span:
            span.set_attribute("node.id", article.id)
            span.set_attribute("node.type", "Article")
            with self.session() as s:
                s.run(
                    """
                    MERGE (n:Article {id: $id})
                    SET n.law_id = $law_id, n.article_no = $article_no,
                        n.content_hash = $content_hash, n.chunk_ids = $chunk_ids,
                        n.status = $status, n.char_start = $char_start, n.char_end = $char_end
                    RETURN n.id AS id
                    """,
                    id=article.id, law_id=article.law_id, article_no=article.article_no,
                    content_hash=article.content_hash, chunk_ids=article.chunk_ids,
                    status=article.status, char_start=article.char_start, char_end=article.char_end,
                ).single()
            return article.id

    @_retry_write()
    def upsert_concept(
        self, name: str, aliases: list[str],
        embedding: list[float], source_chunk_ids: list[str],
    ) -> str:
        with tracer.start_as_current_span("kg.store.upsert_concept") as span:
            span.set_attribute("concept.name", name)
            with self.session() as s:
                result = s.run(
                    f"""
                    MERGE (n:Concept {{name: $name}})
                    ON CREATE SET n.id = randomUUID(),
                                  n.aliases = {_DEDUPE_ALIASES},
                                  n.embedding = $embedding,
                                  n.source_chunk_ids = {_DEDUPE_CHUNK_IDS}
                    ON MATCH SET n.aliases = {_DEDUPE_ALIASES},
                                n.source_chunk_ids = {_DEDUPE_CHUNK_IDS}
                    RETURN n.id AS id
                    """,
                    name=name, aliases=aliases, embedding=embedding,
                    source_chunk_ids=source_chunk_ids,
                ).single()
            span.set_attribute("node.id", result["id"])
            return result["id"]

    @_retry_write()
    def upsert_party(self, name: str, aliases: list[str], source_chunk_ids: list[str]) -> str:
        with tracer.start_as_current_span("kg.store.upsert_party") as span:
            span.set_attribute("party.name", name)
            with self.session() as s:
                result = s.run(
                    f"""
                    MERGE (n:Party {{name: $name}})
                    ON CREATE SET n.id = randomUUID(),
                                  n.aliases = {_DEDUPE_ALIASES},
                                  n.source_chunk_ids = {_DEDUPE_CHUNK_IDS}
                    ON MATCH SET n.aliases = {_DEDUPE_ALIASES},
                                n.source_chunk_ids = {_DEDUPE_CHUNK_IDS}
                    RETURN n.id AS id
                    """,
                    name=name, aliases=aliases, source_chunk_ids=source_chunk_ids,
                ).single()
            span.set_attribute("node.id", result["id"])
            return result["id"]

    @_retry_write()
    def upsert_region(self, name: str, level: str) -> str:
        with tracer.start_as_current_span("kg.store.upsert_region") as span:
            span.set_attribute("region.name", name)
            with self.session() as s:
                result = s.run(
                    """
                    MERGE (n:Region {name: $name})
                    ON CREATE SET n.id = randomUUID(), n.level = $level
                    RETURN n.id AS id
                    """,
                    name=name, level=level,
                ).single()
            span.set_attribute("node.id", result["id"])
            return result["id"]

    @_retry_write()
    def upsert_document(self, doc: DocumentNode) -> str:
        with tracer.start_as_current_span("kg.store.upsert_document") as span:
            span.set_attribute("node.id", doc.id)
            span.set_attribute("node.type", "Document")
            with self.session() as s:
                s.run(
                    """
                    MERGE (n:Document {id: $id})
                    SET n.source_file = $source_file, n.uploaded_at = $uploaded_at,
                        n.doc_type = $doc_type
                    RETURN n.id AS id
                    """,
                    id=doc.id, source_file=doc.source_file,
                    uploaded_at=doc.uploaded_at, doc_type=doc.doc_type,
                ).single()
            return doc.id

    @_retry_write()
    def merge_relation(self, from_id: str, to_id: str, edge_type: str, props: dict) -> None:
        if edge_type not in _VALID_EDGE_TYPES:
            raise ValueError(
                f"非法边类型 {edge_type!r}，允许值: {sorted(_VALID_EDGE_TYPES)}"
            )
        for k in props:
            if not _PROP_KEY_RE.match(k):
                raise ValueError(f"非法关系属性名 {k!r}")
        props_cypher = ", ".join(f"r.{k} = ${k}" for k in props) if props else ""
        set_clause = f"SET {props_cypher}" if props_cypher else ""
        with tracer.start_as_current_span("kg.store.merge_relation") as span:
            span.set_attribute("edge.type", edge_type)
            span.set_attribute("edge.from", from_id)
            span.set_attribute("edge.to", to_id)
            with self.session() as s:
                s.run(
                    f"""
                    MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[r:{edge_type}]->(b)
                    {set_clause}
                    """,
                    from_id=from_id, to_id=to_id, **props,
                )

    def find_articles_by_concept(self, concept_ids: list[str], max_depth: int = 2) -> list[dict]:
        """查询与给定 Concept 关联（EXPLAINS/IS_A，深度 <= max_depth）的 active Article。

        返回按 concept_hit_count 降序：article_id / chunk_ids / matched_concepts /
        concept_hit_count。Neo4j 不支持变长路径参数化深度，max_depth 强转 int 后
        拼入 Cypher。
        """
        if not concept_ids:
            return []
        depth = int(max_depth)
        with tracer.start_as_current_span("kg.store.find_articles_by_concept") as span:
            span.set_attribute("concept.count", len(concept_ids))
            try:
                with self.session() as s:
                    result = s.run(
                        f"""
                        MATCH (c:Concept)-[:EXPLAINS|IS_A*1..{depth}]-(a:Article)
                        WHERE c.id IN $concept_ids AND a.status = 'active'
                        RETURN DISTINCT a.id AS article_id, a.chunk_ids AS chunk_ids,
                               collect(DISTINCT c.name) AS matched_concepts,
                               count(DISTINCT c) AS concept_hit_count
                        ORDER BY concept_hit_count DESC
                        LIMIT 20
                        """,
                        concept_ids=concept_ids,
                    )
                    return [dict(r) for r in result]
            except Exception as e:
                raise KGQueryError(f"Cypher 查询失败: {e}") from e


class Neo4jStore(_Neo4jStoreWriteMixin):
    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
        except Exception as e:
            raise KGConnectionError(f"Neo4j 连接失败: {e}") from e
        self._init_constraints()

    def _init_constraints(self) -> None:
        with self.session() as session:
            for cypher in CONSTRAINTS_CYPHER:
                session.run(cypher)
            session.run(VECTOR_INDEX_CYPHER)

    def session(self) -> Session:
        return self.driver.session()

    def close(self) -> None:
        self.driver.close()


_store: Neo4jStore | None = None


def get_graph_store() -> Neo4jStore:
    global _store
    if _store is None:
        from app.core.config import get_settings
        s = get_settings()
        _store = Neo4jStore(uri=s.NEO4J_URI, user=s.NEO4J_USER, password=s.NEO4J_PASSWORD)
    return _store


def reset_graph_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None
