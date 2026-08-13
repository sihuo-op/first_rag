"""测试 Milvus rag_chunks collection 和 PG document_chunks 表含 char_start/char_end 字段。

Task 4: Milvus + PG Schema Migration for char_start/char_end

- Milvus: rag_chunks collection schema 含 char_start / char_end INT64 字段
- PG: document_chunks 表含 char_start / char_end INT 列
- DocumentChunk SQLAlchemy model 含 char_start / char_end 属性
- insert_vectors 时 char_start/char_end 从 metadata 读取（缺省 0）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_milvus_rag_chunks_has_char_offset_fields():
    """集成测试：连接 Milvus，验证 rag_chunks collection 含 char_start/char_end。

    Milvus 不可用时 skip。
    调用 create_collection 触发 drop+recreate 迁移路径（如缺字段）。
    """
    from app.core.dependencies import get_vector_store

    try:
        vs = get_vector_store()
        # create_collection 是幂等的：若现有 collection 缺 char_start/char_end，
        # 会 drop + recreate（生产迁移路径）；若已有字段则直接返回。
        vs.create_collection("chunks")
        if not vs.has_collection("chunks"):
            pytest.skip("Milvus chunks collection not available")
    except Exception as e:
        pytest.skip(f"Milvus not available: {e}")

    fields = vs.get_collection_fields("chunks")
    assert "char_start" in fields, f"char_start missing from rag_chunks fields: {fields}"
    assert "char_end" in fields, f"char_end missing from rag_chunks fields: {fields}"


def test_pg_document_chunks_has_char_offset_columns():
    """验证 PG/SQLite document_chunks 表含 char_start / char_end 列。

    用独立 in-memory SQLite + create_all + _migrate_add_columns 模拟生产迁移路径。
    """
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.pool import StaticPool

    from app.entities.database import Base
    from app.db.init_db import _migrate_add_columns

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns(engine=engine)

    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("document_chunks")]
    assert "char_start" in columns, f"char_start missing from document_chunks columns: {columns}"
    assert "char_end" in columns, f"char_end missing from document_chunks columns: {columns}"


def test_document_chunk_model_has_char_offset_attributes():
    """DocumentChunk SQLAlchemy model 应含 char_start / char_end 列定义。"""
    from app.entities.database import DocumentChunk

    assert hasattr(DocumentChunk, "char_start"), "DocumentChunk.char_start missing"
    assert hasattr(DocumentChunk, "char_end"), "DocumentChunk.char_end missing"

    # 列类型应为 Integer
    char_start_col = DocumentChunk.__table__.c.char_start
    char_end_col = DocumentChunk.__table__.c.char_end
    # SQLAlchemy Integer 类型
    from sqlalchemy import Integer
    assert isinstance(char_start_col.type, Integer), f"char_start type: {type(char_start_col.type)}"
    assert isinstance(char_end_col.type, Integer), f"char_end type: {type(char_end_col.type)}"


def test_milvus_collection_schema_definition_includes_char_offset():
    """单元测试：MilvusStore.create_collection 创建的 schema 含 char_start/char_end INT64 字段。

    不需要连接 Milvus：直接检查 create_collection 方法源代码中字段定义。
    用 inspect.getsource 验证字段在 schema 中声明。
    """
    import inspect as pyinspect
    from app.rag.vector_store import MilvusStore

    source = pyinspect.getsource(MilvusStore.create_collection)
    assert 'name="char_start"' in source, "char_start FieldSchema missing in create_collection"
    assert 'name="char_end"' in source, "char_end FieldSchema missing in create_collection"
    assert "DataType.INT64" in source, "INT64 dtype missing in create_collection"


def test_insert_vectors_reads_char_offset_from_metadata():
    """单元测试：insert_vectors 应从 metadata 读取 char_start/char_end（缺省 0）。

    用 mock collection 验证 insert 调用参数包含 char_start/char_end 列。
    """
    from unittest.mock import MagicMock, patch
    from app.rag.vector_store import MilvusStore

    vs = MilvusStore(host="localhost", port=19530)
    # 跳过实际 Milvus 连接
    vs.connect = MagicMock()
    mock_collection = MagicMock()
    mock_collection.insert = MagicMock(return_value=MagicMock())
    vs._get_collection = MagicMock(return_value=mock_collection)
    vs._get_full_name = MagicMock(return_value="rag_chunks")

    # metadata 含 char_start/char_end
    metadata_list = [
        {"document_id": 1, "chunk_type": "small", "char_start": 10, "char_end": 50},
        {"document_id": 1, "chunk_type": "small"},  # 缺省应填 0
    ]
    documents = ["hello world", "foo bar"]
    vectors = [[0.1] * 4, [0.2] * 4]

    vs.insert_vectors("chunks", vectors, documents, metadata_list)

    # 验证 insert 被调用，且第 7、8 个参数是 char_starts / char_ends
    assert mock_collection.insert.called, "collection.insert not called"
    call_args = mock_collection.insert.call_args[0][0]
    # schema 顺序: ids, document_ids, chunk_types, documents, content_hashes, statuses, char_starts, char_ends, vectors
    # 验证至少 9 个列（原 7 + char_start + char_end）
    assert len(call_args) >= 9, f"expected >=9 columns in insert, got {len(call_args)}"
    char_starts = call_args[6]
    char_ends = call_args[7]
    assert char_starts == [10, 0], f"char_starts mismatch: {char_starts}"
    assert char_ends == [50, 0], f"char_ends mismatch: {char_ends}"
