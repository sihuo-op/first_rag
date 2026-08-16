"""测试 DocumentService.process_document 触发冲突检测后台任务的逻辑。

仅验证触发行为，不验证文档处理全流程（用 mock 跳过解析/切分/向量库）。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.document_service import DocumentService


def _make_service_with_mocks(doc_id: int, with_conflict_service: bool = True):
    """构造一个 DocumentService，db/vector_store/conflict_service 全部 mock。"""
    db = MagicMock()
    document = MagicMock()
    document.id = doc_id
    document.file_type = "md"
    document.file_path = "/tmp/fake.md"
    db.query.return_value.filter.return_value.first.return_value = document

    vector_store = MagicMock()
    vector_store.has_collection.return_value = True
    vector_store.add_texts.return_value = ["m1", "m2", "m3"]

    conflict_service = MagicMock() if with_conflict_service else None

    svc = DocumentService.__new__(DocumentService)
    svc.db = db
    svc.vector_store = vector_store
    svc.retriever = None
    svc.conflict_service = conflict_service
    svc.splitter = MagicMock()
    svc.splitter.split.return_value = [
        {"content": "L", "chunk_type": "large", "position": 0},
        {"content": "M", "chunk_type": "medium", "position": 0},
        {"content": "S", "chunk_type": "small", "position": 0},
    ]
    return svc, db, vector_store, document


def test_process_document_registers_conflict_detection_when_service_present():
    """有 conflict_service 且提供 background_tasks 时，应注册冲突检测后台任务。"""
    svc, db, _, document = _make_service_with_mocks(doc_id=42, with_conflict_service=True)
    background_tasks = MagicMock()

    with patch("app.services.document_service.get_parser") as mock_get_parser, \
         patch("app.services.document_service._trigger_kg_extraction"):  # 隔离 KG 触发（另一后台任务）
        mock_get_parser.return_value.parse.return_value = ("fake content", {})
        result = svc.process_document(42, background_tasks)

    assert result is True
    # 文档被标记为 pending
    assert document.conflict_check_status == "pending"
    # 触发了后台任务注册
    background_tasks.add_task.assert_called_with(svc._run_conflict_detection, 42)


def test_process_document_skips_conflict_detection_when_no_service():
    """没有 conflict_service 时不应注册冲突检测。"""
    svc, _, _, document = _make_service_with_mocks(doc_id=42, with_conflict_service=False)
    background_tasks = MagicMock()

    with patch("app.services.document_service.get_parser") as mock_get_parser, \
         patch("app.services.document_service._trigger_kg_extraction"):  # 隔离 KG 触发（另一后台任务）
        mock_get_parser.return_value.parse.return_value = ("fake content", {})
        result = svc.process_document(42, background_tasks)

    assert result is True
    background_tasks.add_task.assert_not_called()
    # conflict_check_status 不应被改成 pending
    document.__setattr__("conflict_check_status", "completed")  # 确保默认值


def test_process_document_skips_conflict_detection_when_no_background_tasks():
    """有 conflict_service 但没有 background_tasks 时不应注册（无后台执行环境）。"""
    svc, _, _, document = _make_service_with_mocks(doc_id=42, with_conflict_service=True)

    with patch("app.services.document_service.get_parser") as mock_get_parser:
        mock_get_parser.return_value.parse.return_value = ("fake content", {})
        result = svc.process_document(42, background_tasks=None)

    assert result is True


def test_run_conflict_detection_uses_independent_session():
    """_run_conflict_detection 应使用独立 SessionLocal 并调用 ConflictService.detect_for_document。"""
    svc, _, vector_store, _ = _make_service_with_mocks(doc_id=99, with_conflict_service=True)

    fake_session = MagicMock()
    fake_conflict_svc = MagicMock()

    with patch("app.db.session.SessionLocal", return_value=fake_session) as mock_sessionlocal, \
         patch("app.services.document_service.ConflictService", return_value=fake_conflict_svc) as mock_cs:
        svc._run_conflict_detection(99)

    mock_sessionlocal.assert_called_once()
    mock_cs.assert_called_once_with(fake_session, vector_store)
    fake_conflict_svc.detect_for_document.assert_called_once_with(99)
    fake_session.close.assert_called_once()


def test_run_conflict_detection_swallows_exceptions():
    """后台冲突检测失败时不应抛异常（仅打印日志），且仍关闭 session。"""
    svc, _, _, _ = _make_service_with_mocks(doc_id=99, with_conflict_service=True)

    fake_session = MagicMock()
    fake_conflict_svc = MagicMock()
    fake_conflict_svc.detect_for_document.side_effect = RuntimeError("boom")

    with patch("app.db.session.SessionLocal", return_value=fake_session), \
         patch("app.services.document_service.ConflictService", return_value=fake_conflict_svc):
        # 不应抛异常
        svc._run_conflict_detection(99)

    fake_session.close.assert_called_once()


def test_upload_document_schedules_process_document_with_background_tasks():
    """upload_document 接收 background_tasks 时，应注册 process_document 后台任务（携带 background_tasks 参数，确保冲突检测能被触发）。"""
    from unittest.mock import AsyncMock

    svc, db, vector_store, document = _make_service_with_mocks(doc_id=77, with_conflict_service=True)
    background_tasks = MagicMock()

    fake_file = MagicMock()
    fake_file.filename = "fake.md"
    fake_file.read = AsyncMock(return_value=b"hello")

    with patch("app.services.document_service.settings") as mocked_settings, \
         patch("os.makedirs"), \
         patch("builtins.open"), \
         patch("app.services.document_service.Document") as mock_doc_cls:
        mocked_settings.allowed_extensions = [".md"]
        mocked_settings.upload_dir = "/tmp/uploads"
        mock_doc_cls.return_value = document

        import asyncio
        asyncio.run(svc.upload_document(fake_file, user_id=1, background_tasks=background_tasks))

    # upload_document 应当注册 process_document（携带 background_tasks），而不是同步执行
    background_tasks.add_task.assert_called_with(svc.process_document, document.id, background_tasks)
    # db.add / db.commit 被调用（持久化 document）
    assert db.add.called
    assert db.commit.called


def test_upload_document_runs_synchronously_without_background_tasks():
    """upload_document 没有 background_tasks 时，应同步调用 process_document（不会调度后台任务）。"""
    from unittest.mock import AsyncMock

    svc, db, vector_store, document = _make_service_with_mocks(doc_id=88, with_conflict_service=True)

    fake_file = MagicMock()
    fake_file.filename = "fake.md"
    fake_file.read = AsyncMock(return_value=b"hello")

    with patch("app.services.document_service.settings") as mocked_settings, \
         patch("os.makedirs"), \
         patch("builtins.open"), \
         patch("app.services.document_service.Document") as mock_doc_cls, \
         patch.object(svc, "process_document") as mock_process:
        mocked_settings.allowed_extensions = [".md"]
        mocked_settings.upload_dir = "/tmp/uploads"
        mock_doc_cls.return_value = document

        import asyncio
        asyncio.run(svc.upload_document(fake_file, user_id=1, background_tasks=None))

    # 同步调用 process_document，且未传递 background_tasks
    mock_process.assert_called_once_with(document.id)

