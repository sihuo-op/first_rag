import pytest
from testcontainers.neo4j import Neo4jContainer


@pytest.fixture(scope="session")
def neo4j_container():
    with Neo4jContainer("neo4j:5.20") as container:
        yield container


@pytest.fixture
def graph_store(neo4j_container):
    from app.knowledge_graph.graph_store import Neo4jStore
    store = Neo4jStore(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=neo4j_container.password,
    )
    yield store
    with store.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    store.close()
