"""测试 GET /admin/chunks 列表接口。

用 FastAPI TestClient + 依赖注入 override，配合 SQLite in-memory DB。
覆盖：
- 默认列表（按 id desc 排序，分页）
- status 过滤
- archived_reason 过滤
- conflict_with_chunk_id join 出 conflict_with_content
- 非 admin 用户被 403 拦截
"""
import sys
from pathlib import Path
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker, Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.api.admin import router
from app.core.security import get_current_admin_user
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
    """内存 SQLite，建表后 yield session 工厂。

    用 StaticPool 共享单连接，避免 SQLite in-memory 每连接一个独立 DB
    （否则 FastAPI 请求线程看到的 DB 里没有表）。
    """
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
def normal_user(test_db):
    u = User(
        username="user1",
        email="user1@example.com",
        hashed_password="x",
        role=UserRole.USER,
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


def _make_chunk(
    document_id,
    *,
    milvus_id,
    content,
    chunk_type=ChunkType.MEDIUM,
    status="active",
    archived_reason=None,
    conflict_with_chunk_id=None,
    content_hash=None,
    confidence=None,
    review_reason=None,
):
    return DocumentChunk(
        document_id=document_id,
        chunk_type=chunk_type,
        content=content,
        milvus_id=milvus_id,
        status=status,
        archived_reason=archived_reason,
        conflict_with_chunk_id=conflict_with_chunk_id,
        content_hash=content_hash,
        confidence=confidence,
        review_reason=review_reason,
    )


def _build_app(test_db, current_user):
    """构造最小 FastAPI app，仅挂 admin router，override 依赖。"""
    app = FastAPI()
    app.include_router(router)

    def _override_db():
        try:
            yield test_db
        finally:
            pass

    async def _override_user():
        return current_user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_admin_user] = _override_user
    return app


def test_list_chunks_returns_all_sorted_by_id_desc(test_db, admin_user, sample_doc):
    """无过滤时返回所有 chunk，按 id desc 排序。"""
    test_db.add_all([
        _make_chunk(sample_doc.id, milvus_id="m1", content="c1"),
        _make_chunk(sample_doc.id, milvus_id="m2", content="c2"),
        _make_chunk(sample_doc.id, milvus_id="m3", content="c3"),
    ])
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 3
    # id desc 排序
    assert [c["milvus_id"] for c in data] == ["m3", "m2", "m1"]


def test_list_chunks_filter_by_status(test_db, admin_user, sample_doc):
    """status 过滤只返回匹配的 chunk。"""
    test_db.add_all([
        _make_chunk(sample_doc.id, milvus_id="a1", content="active1", status="active"),
        _make_chunk(sample_doc.id, milvus_id="pr1", content="pending1", status="pending_review"),
        _make_chunk(sample_doc.id, milvus_id="sp1", content="superseded1", status="superseded"),
        _make_chunk(sample_doc.id, milvus_id="pr2", content="pending2", status="pending_review"),
    ])
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks", params={"status": "pending_review"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert {c["milvus_id"] for c in data} == {"pr1", "pr2"}
    assert all(c["status"] == "pending_review" for c in data)


def test_list_chunks_filter_by_archived_reason(test_db, admin_user, sample_doc):
    """archived_reason 过滤只返回匹配的 chunk。"""
    test_db.add_all([
        _make_chunk(sample_doc.id, milvus_id="a1", content="active", status="active"),
        _make_chunk(
            sample_doc.id,
            milvus_id="ar1",
            content="archived-cold",
            status="archived",
            archived_reason="low_hit",
        ),
        _make_chunk(
            sample_doc.id,
            milvus_id="ar2",
            content="archived-stale",
            status="archived",
            archived_reason="stale",
        ),
    ])
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks", params={"archived_reason": "low_hit"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [c["milvus_id"] for c in data] == ["ar1"]


def test_list_chunks_joins_conflict_with_content(test_db, admin_user, sample_doc):
    """有 conflict_with_chunk_id 时，应 join 出新 chunk 的 content 填到 conflict_with_content。"""
    # 新 chunk（增量更新产生）
    new_chunk = _make_chunk(
        sample_doc.id,
        milvus_id="new-mv-1",
        content="所有商品可 7 天无理由退款",
        status="active",
    )
    test_db.add(new_chunk)
    test_db.flush()

    # 旧 chunk（被检测出冲突，转 pending_review）
    old_chunk = _make_chunk(
        sample_doc.id,
        milvus_id="old-mv-1",
        content="电池类商品不可退货退款",
        status="pending_review",
        conflict_with_chunk_id="new-mv-1",
        confidence=0.6,
        review_reason="结论矛盾",
    )
    test_db.add(old_chunk)
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks", params={"status": "pending_review"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    item = data[0]
    assert item["milvus_id"] == "old-mv-1"
    assert item["conflict_with_chunk_id"] == "new-mv-1"
    assert item["conflict_with_content"] == "所有商品可 7 天无理由退款"
    assert item["confidence"] == 0.6
    assert item["review_reason"] == "结论矛盾"


def test_list_chunks_no_conflict_when_conflict_with_chunk_id_missing(test_db, admin_user, sample_doc):
    """conflict_with_chunk_id 为空时 conflict_with_content 也为 None。"""
    test_db.add(_make_chunk(
        sample_doc.id,
        milvus_id="active1",
        content="无冲突",
        status="active",
    ))
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    assert data[0]["conflict_with_chunk_id"] is None
    assert data[0]["conflict_with_content"] is None


def test_list_chunks_pagination(test_db, admin_user, sample_doc):
    """skip / limit 分页。"""
    test_db.add_all([
        _make_chunk(sample_doc.id, milvus_id=f"m{i}", content=f"c{i}") for i in range(5)
    ])
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    # 第 1 页（id desc），skip=0 limit=2 → m4, m3
    resp = client.get("/api/v1/admin/chunks", params={"skip": 0, "limit": 2})
    assert resp.status_code == 200
    assert [c["milvus_id"] for c in resp.json()] == ["m4", "m3"]
    # 第 2 页
    resp = client.get("/api/v1/admin/chunks", params={"skip": 2, "limit": 2})
    assert resp.status_code == 200
    assert [c["milvus_id"] for c in resp.json()] == ["m2", "m1"]


def test_list_chunks_returns_new_schema_fields(test_db, admin_user, sample_doc):
    """返回的 chunk 应包含新 schema 字段（content_hash/status/统计字段/归档字段）。"""
    test_db.add(_make_chunk(
        sample_doc.id,
        milvus_id="m1",
        content="c1",
        status="active",
        content_hash="abc123",
    ))
    test_db.commit()

    app = _build_app(test_db, admin_user)
    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks")
    assert resp.status_code == 200
    item = resp.json()[0]
    # 新字段全部存在
    for field in (
        "content_hash",
        "status",
        "conflict_with_chunk_id",
        "conflict_with_content",
        "conflict_detected_at",
        "confidence",
        "review_reason",
        "superseded_at",
        "access_count",
        "hit_count",
        "avg_score",
        "archived_reason",
        "archived_at",
    ):
        assert field in item, f"missing field: {field}"
    assert item["content_hash"] == "abc123"
    assert item["status"] == "active"
    assert item["access_count"] == 0
    assert item["hit_count"] == 0


def test_list_chunks_forbidden_for_non_admin(test_db, normal_user, sample_doc):
    """非 admin 用户访问应被依赖拦截（403）。

    这里通过 override get_current_admin_user 模拟普通用户来验证依赖路径；
    实际生产环境由 get_current_admin_user 内部抛 403。
    """
    test_db.add(_make_chunk(sample_doc.id, milvus_id="m1", content="c1"))
    test_db.commit()

    app = FastAPI()
    app.include_router(router)

    def _override_db():
        try:
            yield test_db
        finally:
            pass

    async def _override_user():
        # 模拟 get_current_admin_user 内部的 403 检查
        from fastapi import HTTPException, status as http_status
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_admin_user] = _override_user

    client = TestClient(app)
    resp = client.get("/api/v1/admin/chunks")
    assert resp.status_code == 403
