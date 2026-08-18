import pytest
from testcontainers.neo4j import Neo4jContainer


@pytest.fixture(scope="session")
def neo4j_container():
    with Neo4jContainer("neo4j:5.20") as container:
        yield container


@pytest.fixture
def graph_store(neo4j_container):
    from app.knowledge_graph.graph_store import Neo4jStore
    # 兼容不同 testcontainers 版本：统一从容器环境变量读取密码
    neo4j_auth = neo4j_container.env.get("NEO4J_AUTH", "neo4j/password")
    store = Neo4jStore(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=neo4j_auth.split("/", 1)[1],
    )
    yield store
    with store.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    store.close()
