"""Neo4j 连接与 Cypher 读写封装。"""
from neo4j import GraphDatabase, Driver, Session
from app.core.observability import get_tracer
from app.knowledge_graph.exceptions import KGConnectionError

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


class Neo4jStore:
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
