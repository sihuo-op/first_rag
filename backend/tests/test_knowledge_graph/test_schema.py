from app.knowledge_graph.schema import (
    NodeType, EdgeType, ConflictStatus, ArticleStatus,
    ArticleNode, ConceptNode, CYPHER_LABELS,
)


def test_node_type_has_six_types():
    assert {n.value for n in NodeType} == {"Law", "Article", "Concept", "Party", "Region", "Document"}


def test_edge_type_has_six_types():
    assert {e.value for e in EdgeType} == {
        "CITES", "IS_A", "CONFLICTS_WITH", "APPLIES_TO", "CONTAINS", "EXPLAINS"
    }


def test_conflict_status_states():
    assert {c.value for c in ConflictStatus} == {"pending_review", "confirmed", "dismissed"}


def test_article_status_states():
    assert {c.value for c in ArticleStatus} == {"active", "superseded", "archived"}


def test_article_node_validates_required_fields():
    a = ArticleNode(
        id="art-1", law_id="law-1", article_no=19, content_hash="abc",
        chunk_ids=["c1", "c2"], status="active", char_start=0, char_end=100,
    )
    assert a.id == "art-1"
    assert a.chunk_ids == ["c1", "c2"]


def test_concept_node_requires_embedding():
    c = ConceptNode(
        id="c-1", name="试用期", aliases=["试用期间"],
        embedding=[0.1] * 1024, source_chunk_ids=["c1"],
    )
    assert len(c.embedding) == 1024


def test_cypher_labels_match_node_types():
    for nt in NodeType:
        assert nt.value in CYPHER_LABELS.values()
