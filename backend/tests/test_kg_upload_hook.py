"""Task 14: 文档上传处理成功后触发 KG 抽取的接线测试。

覆盖三点：
1. KG 启用时通过 BackgroundTasks 调度 KGExtractor.run；
2. KG 关闭时干净 no-op；
3. KG 基础设施不可用时触发本身绝不抛异常（不影响文档处理）。
另补充 process_document 成功路径实际调用触发器、以及 document loader
重新解析文件（Document 无 full_text 字段）的行为验证。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_trigger_kg_extraction_schedules_run():
    """KG 启用时，_trigger_kg_extraction 通过 BackgroundTasks 调度 KGExtractor.run。"""
    from fastapi import BackgroundTasks
    bg_tasks = MagicMock(spec=BackgroundTasks)

    with patch("app.services.document_service.KGExtractor") as extractor_cls, \
         patch("app.services.document_service.get_graph_store"), \
         patch("app.services.document_service.get_vector_store"), \
         patch("app.services.document_service.get_settings") as mock_settings:
        mock_settings.return_value.KG_ENABLED = True
        mock_extractor = MagicMock()
        extractor_cls.return_value = mock_extractor

        from app.services.document_service import _trigger_kg_extraction
        _trigger_kg_extraction(document_id="doc-1", background_tasks=bg_tasks)

        bg_tasks.add_task.assert_called_once()
        called_args = bg_tasks.add_task.call_args
        assert called_args[0][0] == mock_extractor.run
        assert called_args[1]["document_id"] == "doc-1"


def test_trigger_kg_extraction_disabled_noop():
    """KG 关闭时不调度任何任务。"""
    from fastapi import BackgroundTasks
    bg_tasks = MagicMock(spec=BackgroundTasks)

    with patch("app.services.document_service.get_settings") as mock_settings:
        mock_settings.return_value.KG_ENABLED = False
        from app.services.document_service import _trigger_kg_extraction
        _trigger_kg_extraction(document_id="doc-1", background_tasks=bg_tasks)

        bg_tasks.add_task.assert_not_called()


def test_trigger_kg_extraction_never_raises():
    """KG 基础设施不可用时（如 Neo4j down），触发本身不抛异常、不影响文档处理。"""
    from fastapi import BackgroundTasks
    bg_tasks = MagicMock(spec=BackgroundTasks)

    with patch("app.services.document_service.get_settings") as mock_settings, \
         patch("app.services.document_service.get_graph_store",
               side_effect=Exception("neo4j down")):
        mock_settings.return_value.KG_ENABLED = True
        from app.services.document_service import _trigger_kg_extraction
        _trigger_kg_extraction(document_id="doc-1", background_tasks=bg_tasks)

        bg_tasks.add_task.assert_not_called()


# ---------------------------------------------------------------------------
# 补充：process_document 成功路径的接线行为
# ---------------------------------------------------------------------------

def _make_service_with_mocks(doc_id: int):
    """构造 DocumentService，db/vector_store/splitter 全 mock（跳过解析/切分/向量库）。"""
    from app.services.document_service import DocumentService

    db = MagicMock()
    document = MagicMock()
    document.id = doc_id
    document.file_type = "md"
    document.file_path = "/tmp/fake.md"
    db.query.return_value.filter.return_value.first.return_value = document

    vector_store = MagicMock()
    vector_store.has_collection.return_value = True
    vector_store.add_texts.return_value = ["m1", "m2", "m3"]

    svc = DocumentService.__new__(DocumentService)
    svc.db = db
    svc.vector_store = vector_store
    svc.retriever = None
    svc.conflict_service = None
    svc.splitter = MagicMock()
    svc.splitter.split.return_value = [
        {"content": "L", "chunk_type": "large", "position": 0},
        {"content": "M", "chunk_type": "medium", "position": 0},
        {"content": "S", "chunk_type": "small", "position": 0},
    ]
    return svc


def test_process_document_triggers_kg_extraction():
    """process_document 成功后应调用 _trigger_kg_extraction（str(doc_id) + background_tasks）。"""
    svc = _make_service_with_mocks(doc_id=42)
    background_tasks = MagicMock()

    with patch("app.services.document_service.get_parser") as mock_get_parser, \
         patch("app.services.document_service._trigger_kg_extraction") as mock_trigger:
        mock_get_parser.return_value.parse.return_value = ("fake content", {})
        result = svc.process_document(42, background_tasks)

    assert result is True
    mock_trigger.assert_called_once_with("42", background_tasks)


def test_process_document_skips_kg_extraction_without_background_tasks():
    """无后台执行环境（background_tasks=None）时不触发 KG 抽取。"""
    svc = _make_service_with_mocks(doc_id=42)

    with patch("app.services.document_service.get_parser") as mock_get_parser, \
         patch("app.services.document_service._trigger_kg_extraction") as mock_trigger:
        mock_get_parser.return_value.parse.return_value = ("fake content", {})
        result = svc.process_document(42, None)

    assert result is True
    mock_trigger.assert_not_called()


def test_load_document_text_for_kg_reparses_file():
    """Document 无 full_text 字段：document loader 应通过 get_parser 重新解析文件。"""
    fake_doc = MagicMock()
    fake_doc.file_type = "md"
    fake_doc.file_path = "/tmp/fake.md"

    fake_session = MagicMock()
    fake_session.__enter__.return_value = fake_session
    fake_session.query.return_value.filter.return_value.first.return_value = fake_doc

    with patch("app.db.session.SessionLocal", return_value=fake_session), \
         patch("app.services.document_service.get_parser") as mock_get_parser:
        mock_get_parser.return_value.parse.return_value = ("全文内容", {"meta": 1})

        from app.services.document_service import _load_document_text_for_kg
        text = _load_document_text_for_kg("7")

    assert text == "全文内容"
    mock_get_parser.assert_called_once_with("md")
    mock_get_parser.return_value.parse.assert_called_once_with("/tmp/fake.md")


def test_load_document_text_for_kg_missing_doc_returns_empty():
    """文档不存在时返回空字符串（抽取管道自行兜底），不抛异常。"""
    fake_session = MagicMock()
    fake_session.__enter__.return_value = fake_session
    fake_session.query.return_value.filter.return_value.first.return_value = None

    with patch("app.db.session.SessionLocal", return_value=fake_session):
        from app.services.document_service import _load_document_text_for_kg
        assert _load_document_text_for_kg("999") == ""
