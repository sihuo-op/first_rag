"""kg_admin API 单元测试：mock Neo4j（FastAPI TestClient + dependency_overrides）。

注意：confirm/dismiss 的 edge_id 走 Neo4j `id(r)`（整数），因此测试 URL
使用数字 edge id（brief 中 "edge-1" 与 `edge_id: int` 路径参数不一致，此处修正）。
"""
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def _make_client(store: MagicMock) -> TestClient:
    from fastapi import FastAPI
    from app.knowledge_graph.kg_admin import router, _get_store

    app = FastAPI()
    app.dependency_overrides[_get_store] = lambda: store
    app.include_router(router)
    return TestClient(app)


def test_list_pending_conflicts_returns_empty():
    store = MagicMock()
    store.session.return_value.__enter__.return_value.run.return_value = []

    client = _make_client(store)

    resp = client.get("/api/v1/admin/kg/conflicts?status=pending_review")
    assert resp.status_code == 200
    assert resp.json() == {"conflicts": []}


def test_confirm_conflict_updates_edge_status():
    store = MagicMock()

    client = _make_client(store)

    resp = client.post(
        "/api/v1/admin/kg/conflicts/123/confirm",
        json={"review_note": "确认冲突"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "confirmed"}
    run_mock = store.session.return_value.__enter__.return_value.run
    run_mock.assert_called()
    # 只更新边状态，绝不触碰 Article 状态
    cypher = run_mock.call_args[0][0]
    assert "CONFLICTS_WITH" in cypher
    assert "Article" not in cypher
    assert run_mock.call_args[1]["status"] == "confirmed"


def test_dismiss_conflict_updates_edge_status():
    store = MagicMock()

    client = _make_client(store)

    resp = client.post(
        "/api/v1/admin/kg/conflicts/456/dismiss",
        json={"review_note": "误报"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "dismissed"}
    run_mock = store.session.return_value.__enter__.return_value.run
    cypher = run_mock.call_args[0][0]
    assert "CONFLICTS_WITH" in cypher
    assert "Article" not in cypher
    assert run_mock.call_args[1]["status"] == "dismissed"


def test_confirm_missing_edge_returns_404():
    store = MagicMock()
    run_mock = store.session.return_value.__enter__.return_value.run
    run_mock.return_value.consume.return_value.counters.properties_set = 0

    client = _make_client(store)

    resp = client.post(
        "/api/v1/admin/kg/conflicts/999/confirm",
        json={"review_note": "确认冲突"},
    )
    assert resp.status_code == 404


def test_supersede_article_updates_status():
    store = MagicMock()

    client = _make_client(store)

    resp = client.post(
        "/api/v1/admin/kg/articles/art-1/supersede",
        json={"reason": "新法生效"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "superseded", "article_id": "art-1"}
    run_mock = store.session.return_value.__enter__.return_value.run
    cypher = run_mock.call_args[0][0]
    # supersede 独立于 conflict：只改 Article 节点，Cypher 不出现 CONFLICTS_WITH
    assert "MATCH (a:Article" in cypher
    assert "CONFLICTS_WITH" not in cypher
    assert run_mock.call_args[1]["status"] == "superseded"


def test_supersede_missing_article_returns_404():
    store = MagicMock()
    run_mock = store.session.return_value.__enter__.return_value.run
    run_mock.return_value.consume.return_value.counters.properties_set = 0

    client = _make_client(store)

    resp = client.post(
        "/api/v1/admin/kg/articles/nope/supersede",
        json={"reason": "新法生效"},
    )
    assert resp.status_code == 404


def test_invalid_status_filter_returns_400():
    store = MagicMock()

    client = _make_client(store)

    resp = client.get("/api/v1/admin/kg/conflicts?status=bogus")
    assert resp.status_code == 400


def test_stats_returns_counts():
    store = MagicMock()
    # Mock stats query results（单条 Cypher 返回三个计数）
    session_mock = MagicMock()
    session_mock.run.return_value.single.return_value = {
        "node_count": 100, "edge_count": 200, "pending_count": 5,
    }
    store.session.return_value.__enter__.return_value = session_mock

    client = _make_client(store)

    resp = client.get("/api/v1/admin/kg/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "node_count" in data
    assert "edge_count" in data
    assert "pending_count" in data
    assert data["node_count"] == 100
    assert data["edge_count"] == 200
    assert data["pending_count"] == 5
