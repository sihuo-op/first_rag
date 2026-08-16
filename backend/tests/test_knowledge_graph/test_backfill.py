"""backfill CLI 冒烟测试：单测级别，不依赖 Neo4j/Milvus。

main() 内的 deferred import（from X import Y）在调用时做属性查找，
因此测试通过 patch 目标模块属性来拦截：app.core.dependencies.get_vector_store、
app.db.session.SessionLocal、app.knowledge_graph.extractor.KGExtractor、
app.knowledge_graph.graph_store.get_graph_store。
"""
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.knowledge_graph import backfill
from app.knowledge_graph.extractor import ExtractionReport


class _FakeExtractor:
    """记录构造参数与 run 调用，按 document_id 返回预设报告。"""

    def __init__(self, reports=None, **kwargs):
        self.kwargs = kwargs
        self._reports = reports or {}
        self.calls = []

    def run(self, document_id):
        self.calls.append(document_id)
        return self._reports.get(document_id, ExtractionReport(document_id=document_id))


class _FakeVectorStore:
    def embed_query(self, text):
        return [0.0]


class _FakeQuery:
    def all(self):
        return [SimpleNamespace(id=7), SimpleNamespace(id=8)]


class _FakeDb:
    def query(self, _entity):
        return _FakeQuery()


class _FakeSessionLocal:
    def __enter__(self):
        return _FakeDb()

    def __exit__(self, *args):
        return False


def _patch_cli_deps(fake, argv):
    """打包 main() 全部外部依赖的 patch，返回 ExitStack（调用方 with 使用）。

    fake 为 _FakeExtractor 实例；KGExtractor 的构造参数会被记录到 fake.kwargs。
    """
    def _record_factory(**kwargs):
        fake.kwargs = kwargs
        return fake

    stack = ExitStack()
    stack.enter_context(patch.object(sys, "argv", argv))
    ms = stack.enter_context(patch.object(backfill, "get_settings"))
    ms.return_value.KG_ENABLED = True
    stack.enter_context(patch("app.knowledge_graph.extractor.KGExtractor", _record_factory))
    stack.enter_context(
        patch("app.knowledge_graph.graph_store.get_graph_store", return_value=object())
    )
    stack.enter_context(
        patch("app.core.dependencies.get_vector_store", return_value=_FakeVectorStore())
    )
    stack.enter_context(patch("app.db.session.SessionLocal", _FakeSessionLocal))
    return stack


def test_backfill_cli_requires_arg():
    """无参数时 argparse 报错退出（exit code 2）。"""
    with patch.object(sys, "argv", ["backfill"]):
        with pytest.raises(SystemExit) as e:
            backfill.main()
    assert e.value.code == 2


def test_backfill_cli_disabled_exits_clean(capsys):
    """KG_ENABLED=false 时打印提示并以 exit 0 退出。"""
    with patch.object(sys, "argv", ["backfill", "--all-documents"]), \
         patch.object(backfill, "get_settings") as ms:
        ms.return_value.KG_ENABLED = False
        with pytest.raises(SystemExit) as e:
            backfill.main()
    assert e.value.code == 0
    assert "KG_ENABLED=false" in capsys.readouterr().out


def test_backfill_cli_all_documents_success(capsys):
    """--all-documents：复用 Task 14 loader，doc_id 转 str，成功路径正常返回。"""
    fake = _FakeExtractor()
    with _patch_cli_deps(fake,["backfill", "--all-documents"]):
        backfill.main()  # 成功路径不 sys.exit，正常返回即隐式 exit 0

    # 复用 document_service 的模块级 loader，而非重复实现
    from app.services.document_service import _load_chunks_for_kg, _load_document_text_for_kg
    assert fake.kwargs["chunks_loader"] is _load_chunks_for_kg
    assert fake.kwargs["document_loader"] is _load_document_text_for_kg
    assert fake.kwargs["store"] is not None
    assert callable(fake.kwargs["embedding_fn"])

    assert fake.calls == ["7", "8"]  # doc_id 以 str 传入 extractor.run
    out = capsys.readouterr().out
    assert "Backfilling document 7..." in out
    assert "entities=0, relations=0, conflicts=0" in out
    assert "Done: 2/2 succeeded" in out


def test_backfill_cli_document_id_failure_exits_1(capsys):
    """--document-id 抽取失败（report.error）时打印 FAILED 并 exit 1。"""
    fake = _FakeExtractor(reports={"42": ExtractionReport(
        document_id="42", error="ValueError: 非法规文档", duration_ms=3,
    )})
    with _patch_cli_deps(fake,["backfill", "--document-id", "42"]):
        with pytest.raises(SystemExit) as e:
            backfill.main()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "FAILED: ValueError: 非法规文档" in out
    assert "Done: 0/1 succeeded" in out


def test_backfill_cli_skip_conflicts_uses_null_detector():
    """--skip-conflicts 时传入 _NullConflictDetector（跳过 LLM 冲突判定）。"""
    fake = _FakeExtractor()
    argv = ["backfill", "--all-documents", "--skip-conflicts"]
    with _patch_cli_deps(fake,argv):
        backfill.main()
    assert isinstance(fake.kwargs["conflict_detector"], backfill._NullConflictDetector)


def test_backfill_cli_default_keeps_conflict_detection():
    """不带 --skip-conflicts 时 conflict_detector 为 None（KGExtractor 默认构造）。"""
    fake = _FakeExtractor()
    with _patch_cli_deps(fake,["backfill", "--all-documents"]):
        backfill.main()  # 成功路径正常返回
    assert fake.kwargs["conflict_detector"] is None


def test_run_backfill_empty_doc_list(capsys):
    """空文档列表：无输出、失败数为 0。"""
    fake = _FakeExtractor()
    assert backfill._run_backfill(fake, []) == 0
    assert fake.calls == []
    assert capsys.readouterr().out == ""
