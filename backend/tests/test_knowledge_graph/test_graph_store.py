def test_init_creates_unique_constraints(graph_store):
    with graph_store.session() as s:
        result = s.run("SHOW CONSTRAINTS YIELD name RETURN count(*) AS c")
        assert result.single()["c"] >= 5


def test_init_creates_concept_vector_index(graph_store):
    with graph_store.session() as s:
        result = s.run("SHOW INDEXES YIELD name WHERE name = 'concept_embedding' RETURN count(*) AS c")
        assert result.single()["c"] == 1


def test_init_is_idempotent(graph_store):
    graph_store._init_constraints()
    # no exception
