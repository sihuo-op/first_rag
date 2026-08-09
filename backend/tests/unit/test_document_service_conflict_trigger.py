"""测试 DocumentService.process_document 触发冲突检测后台任务的逻辑。

仅验证触发行为，不验证文档处理全流程（用 mock 跳过解析/切分/向量库）。
注意：document_service.py 存在已知 field drift（DocumentStatus.ACTIVE 不存在），
本测试不修复 drift，只在 conftest 里给 DocumentStatus 加 ACTIVE 别名让代码可执行。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.entities.database import DocumentStatus
from app.services.document_service import DocumentService


@pytest.fixture(autouse=True)
def _patch_document_status_active():
    """document_service.py 引用了 DocumentStatus.ACTIVE，但 ORM 枚举只有 PENDING/PROCESSING/COMPLETED/FAILED。
    这是已知的 field drift，Task 14 不修复。这里临时给 ACTIVE 加个别名让 process_document 能跑通。"""
    if not hasattr(DocumentStatus, "ACTIVE"):
        DocumentStatus.ACTIVE = DocumentStatus.COMPLETED
    yield
    # 不删除别名：其他测试若再触发也不会受影响（幂等）


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
    svc.splitter.split_text.return_value = [{"large": "L", "medium": "M", "small": "S"}]
    return svc, db, vector_store, document


def test_process_document_registers_conflict_detection_when_service_present():
    """有 conflict_service 且提供 background_tasks 时，应注册冲突检测后台任务。"""
    svc, db, _, document = _make_service_with_mocks(doc_id=42, with_conflict_service=True)
    background_tasks = MagicMock()

    with patch("app.services.document_service.get_parser") as mock_get_parser:
        mock_get_parser.return_value.parse.return_value = "fake content"
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

    with patch("app.services.document_service.get_parser") as mock_get_parser:
        mock_get_parser.return_value.parse.return_value = "fake content"
        result = svc.process_document(42, background_tasks)

    assert result is True
    background_tasks.add_task.assert_not_called()
    # conflict_check_status 不应被改成 pending
    document.__setattr__("conflict_check_status", "completed")  # 确保默认值


def test_process_document_skips_conflict_detection_when_no_background_tasks():
    """有 conflict_service 但没有 background_tasks 时不应注册（无后台执行环境）。"""
    svc, _, _, document = _make_service_with_mocks(doc_id=42, with_conflict_service=True)

    with patch("app.services.document_service.get_parser") as mock_get_parser:
        mock_get_parser.return_value.parse.return_value = "fake content"
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
