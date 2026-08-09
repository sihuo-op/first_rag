"""测试 PATCH /admin/chunks/{id} 动作接口。

覆盖：
- confirm: pending_review -> superseded，写 superseded_at/reviewed_by/reviewed_at，sync Milvus
- dismiss: pending_review -> active，清空冲突字段，sync Milvus
- archive: active -> archived（manual），sync Milvus
- restore: archived -> active，清 archived 字段，sync Milvus
- hard_delete: 物理删除 PG + Milvus
- 状态前置校验（400）
- 未知 action（400）
- chunk 不存在（404）
- milvus_id 缺失时跳过 vector_store 调用
"""
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.api.admin import router
from app.core.security import get_current_admin_user
from app.core.dependencies import get_vector_store
from app.db.session import get_db
from app.entities.database import (
    Base,
    User,
    UserRole,
    Document,
    DocumentStatus,
    DocumentChunk,
    ChunkType,
)


@pytest.fixture()
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def admin_user(test_db):
    u = User(
        username="admin1",
        email="admin1@example.com",
        hashed_password="x",
        role=UserRole.ADMIN,
        is_active=True,
    )
    test_db.add(u)
    test_db.commit()
    test_db.refresh(u)
    return u


@pytest.fixture()
def sample_doc(test_db, admin_user):
    d = Document(
        title="Doc",
        file_name="d.md",
        file_path="/tmp/d.md",
        file_type="md",
        user_id=admin_user.id,
        status=DocumentStatus.COMPLETED,
    )
    test_db.add(d)
    test_db.commit()
    test_db.refresh(d)
    return d


@pytest.fixture()
def mock_vector_store():
    """Mock MilvusStore，记录所有调用。"""
    store = MagicMock()
    store.upsert_status = MagicMock()
    store.delete_vectors = MagicMock()
    return store


def _make_chunk(
    document_id,
    *,
    milvus_id,
    content,
    chunk_type=ChunkType.MEDIUM,
    status="active",
    archived_reason=None,
    archived_at=None,
    conflict_with_chunk_id=None,
    conflict_detected_at=None,
    confidence=None,
    review_reason=None,
    superseded_at=None,
):
    return DocumentChunk(
        document_id=document_id,
        chunk_type=chunk_type,
        content=content,
        milvus_id=milvus_id,
        status=status,
        archived_reason=archived_reason,
        archived_at=archived_at,
        conflict_with_chunk_id=conflict_with_chunk_id,
        conflict_detected_at=conflict_detected_at,
        confidence=confidence,
        review_reason=review_reason,
        superseded_at=superseded_at,
    )


def _build_app(test_db, current_user, vector_store):
    app = FastAPI()
    app.include_router(router)

    def _override_db():
        try:
            yield test_db
        finally:
            pass

    async def _override_user():
        return current_user

    def _override_vector_store():
        return vector_store

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_admin_user] = _override_user
    app.dependency_overrides[get_vector_store] = _override_vector_store
    return app


# ============ confirm ============

def test_confirm_pending_review_to_superseded(test_db, admin_user, sample_doc, mock_vector_store):
    """confirm: pending_review -> superseded，写 superseded_at/reviewed_by/reviewed_at，sync Milvus。"""
    chunk = _make_chunk(
        sample_doc.id,
        milvus_id="mv-old-1",
        content="旧内容",
        status="pending_review",
        conflict_with_chunk_id="mv-new-1",
        conflict_detected_at=datetime(2026, 1, 1),
        confidence=0.7,
        review_reason="结论矛盾",
    )
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "confirm"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": f"Chunk {chunk.id} confirm done"}

    test_db.refresh(chunk)
    assert chunk.status == "superseded"
    assert chunk.superseded_at is not None
    assert chunk.reviewed_by == admin_user.id
    assert chunk.reviewed_at is not None

    mock_vector_store.upsert_status.assert_called_once_with("chunks", "mv-old-1", "superseded")


def test_confirm_rejects_non_pending_review(test_db, admin_user, sample_doc, mock_vector_store):
    """confirm 只能用于 pending_review 状态。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "confirm"})
    assert resp.status_code == 400
    assert "pending_review" in resp.json()["detail"]
    mock_vector_store.upsert_status.assert_not_called()


# ============ dismiss ============

def test_dismiss_pending_review_to_active_clears_conflict_fields(
    test_db, admin_user, sample_doc, mock_vector_store
):
    """dismiss: pending_review -> active，清空冲突字段，sync Milvus。"""
    chunk = _make_chunk(
        sample_doc.id,
        milvus_id="mv-old-1",
        content="旧内容",
        status="pending_review",
        conflict_with_chunk_id="mv-new-1",
        conflict_detected_at=datetime(2026, 1, 1),
        confidence=0.6,
        review_reason="结论矛盾",
    )
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "dismiss"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": f"Chunk {chunk.id} dismiss done"}

    test_db.refresh(chunk)
    assert chunk.status == "active"
    assert chunk.conflict_with_chunk_id is None
    assert chunk.conflict_detected_at is None
    assert chunk.confidence is None
    assert chunk.review_reason is None
    assert chunk.reviewed_by == admin_user.id
    assert chunk.reviewed_at is not None

    mock_vector_store.upsert_status.assert_called_once_with("chunks", "mv-old-1", "active")


def test_dismiss_rejects_non_pending_review(test_db, admin_user, sample_doc, mock_vector_store):
    """dismiss 只能用于 pending_review 状态。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="superseded")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "dismiss"})
    assert resp.status_code == 400
    mock_vector_store.upsert_status.assert_not_called()


# ============ archive ============

def test_archive_active_to_archived_manual(test_db, admin_user, sample_doc, mock_vector_store):
    """archive: active -> archived（manual），sync Milvus。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "archive"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": f"Chunk {chunk.id} archive done"}

    test_db.refresh(chunk)
    assert chunk.status == "archived"
    assert chunk.archived_reason == "manual"
    assert chunk.archived_at is not None
    assert chunk.reviewed_by == admin_user.id
    assert chunk.reviewed_at is not None

    mock_vector_store.upsert_status.assert_called_once_with("chunks", "mv-1", "archived")


def test_archive_rejects_non_active(test_db, admin_user, sample_doc, mock_vector_store):
    """archive 只能用于 active 状态。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="pending_review")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "archive"})
    assert resp.status_code == 400
    mock_vector_store.upsert_status.assert_not_called()


# ============ restore ============

def test_restore_archived_to_active_clears_archived_fields(
    test_db, admin_user, sample_doc, mock_vector_store
):
    """restore: archived -> active，清 archived_reason/archived_at，sync Milvus。"""
    chunk = _make_chunk(
        sample_doc.id,
        milvus_id="mv-1",
        content="c",
        status="archived",
        archived_reason="low_hit",
        archived_at=datetime(2026, 1, 1),
    )
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "restore"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": f"Chunk {chunk.id} restore done"}

    test_db.refresh(chunk)
    assert chunk.status == "active"
    assert chunk.archived_reason is None
    assert chunk.archived_at is None
    assert chunk.reviewed_by == admin_user.id
    assert chunk.reviewed_at is not None

    mock_vector_store.upsert_status.assert_called_once_with("chunks", "mv-1", "active")


def test_restore_rejects_non_archived(test_db, admin_user, sample_doc, mock_vector_store):
    """restore 只能用于 archived 状态。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "restore"})
    assert resp.status_code == 400
    mock_vector_store.upsert_status.assert_not_called()


# ============ hard_delete ============

def test_hard_delete_removes_pg_and_milvus(test_db, admin_user, sample_doc, mock_vector_store):
    """hard_delete: 物理删除 PG + Milvus。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)
    chunk_id = chunk.id

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk_id}", json={"action": "hard_delete"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": f"Chunk {chunk_id} hard_delete done"}

    # PG 已删除
    assert test_db.query(DocumentChunk).filter_by(id=chunk_id).first() is None
    # Milvus 已删除
    mock_vector_store.delete_vectors.assert_called_once_with("chunks", ids=["mv-1"])


def test_hard_delete_skips_milvus_when_no_milvus_id(test_db, admin_user, sample_doc, mock_vector_store):
    """milvus_id 缺失时跳过 vector_store.delete_vectors。"""
    chunk = _make_chunk(sample_doc.id, milvus_id=None, content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)
    chunk_id = chunk.id

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk_id}", json={"action": "hard_delete"})
    assert resp.status_code == 200, resp.text
    assert test_db.query(DocumentChunk).filter_by(id=chunk_id).first() is None
    mock_vector_store.delete_vectors.assert_not_called()


# ============ 通用 ============

def test_patch_returns_404_when_chunk_not_found(test_db, admin_user, mock_vector_store):
    """chunk 不存在时返回 404。"""
    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch("/api/v1/admin/chunks/99999", json={"action": "archive"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chunk not found"


def test_patch_rejects_unknown_action(test_db, admin_user, sample_doc, mock_vector_store):
    """未知 action 返回 400。"""
    chunk = _make_chunk(sample_doc.id, milvus_id="mv-1", content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "bogus"})
    assert resp.status_code == 400
    assert "Unknown action" in resp.json()["detail"]
    mock_vector_store.upsert_status.assert_not_called()
    mock_vector_store.delete_vectors.assert_not_called()


def test_patch_skips_milvus_when_no_milvus_id_on_status_change(
    test_db, admin_user, sample_doc, mock_vector_store
):
    """milvus_id 缺失时跳过 vector_store.upsert_status（archive 场景）。"""
    chunk = _make_chunk(sample_doc.id, milvus_id=None, content="c", status="active")
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(chunk)

    app = _build_app(test_db, admin_user, mock_vector_store)
    client = TestClient(app)
    resp = client.patch(f"/api/v1/admin/chunks/{chunk.id}", json={"action": "archive"})
    assert resp.status_code == 200, resp.text
    test_db.refresh(chunk)
    assert chunk.status == "archived"
    mock_vector_store.upsert_status.assert_not_called()


def test_patch_forbidden_for_non_admin(test_db, mock_vector_store):
    """非 admin 用户访问应被依赖拦截（403）。"""
    app = FastAPI()
    app.include_router(router)

    def _override_db():
        try:
            yield test_db
        finally:
            pass

    async def _override_user():
        from fastapi import HTTPException, status as http_status
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    def _override_vector_store():
        return mock_vector_store

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_admin_user] = _override_user
    app.dependency_overrides[get_vector_store] = _override_vector_store

    client = TestClient(app)
    resp = client.patch("/api/v1/admin/chunks/1", json={"action": "archive"})
    assert resp.status_code == 403
