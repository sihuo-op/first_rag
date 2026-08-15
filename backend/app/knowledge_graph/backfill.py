"""现有文档回填 KG 抽取 CLI。

用法：
    cd backend
    python -m app.knowledge_graph.backfill --all-documents
    python -m app.knowledge_graph.backfill --document-id 1
    python -m app.knowledge_graph.backfill --all-documents --skip-conflicts
"""
import argparse
import sys

from app.core.config import get_settings


class _NullConflictDetector:
    """--skip-conflicts 时的空冲突检测器：恒返 0，不触发 LLM 冲突判定。

    KGExtractor 对 conflict_detector=None 会自动构造默认实例（内部含 LLM），
    因此跳过冲突检测必须传入本空实现而非 None。
    """

    def detect_for_article(self, article_id: str) -> int:
        return 0


def _run_backfill(extractor, doc_ids: list[int]) -> int:
    """逐文档执行抽取并打印报告，返回失败文档数。

    extractor.run 永不抛出（失败以 report.error 返回），此处只做结果分发。
    """
    failed = 0
    for doc_id in doc_ids:
        print(f"Backfilling document {doc_id}...")
        report = extractor.run(document_id=str(doc_id))
        if report.error:
            failed += 1
            print(f"  FAILED: {report.error} ({report.duration_ms}ms)")
        else:
            print(
                f"  entities={report.entities_count}, relations={report.relations_count}, "
                f"conflicts={report.conflicts_count}, duration={report.duration_ms}ms"
            )
    return failed


def main():
    parser = argparse.ArgumentParser(description="回填 KG 抽取")
    parser.add_argument("--all-documents", action="store_true", help="回填所有文档")
    parser.add_argument("--document-id", type=int, help="回填指定文档")
    parser.add_argument(
        "--skip-conflicts", action="store_true",
        help="跳过 LLM 冲突检测（重跑回填时省 LLM 调用）",
    )
    args = parser.parse_args()

    if not args.all_documents and not args.document_id:
        parser.error("必须指定 --all-documents 或 --document-id")

    settings = get_settings()
    if not settings.KG_ENABLED:
        print("KG_ENABLED=false, 退出")
        sys.exit(0)

    # Deferred imports so --help works without Neo4j/Milvus deps loaded
    from app.core.dependencies import get_vector_store
    from app.knowledge_graph.extractor import KGExtractor
    from app.knowledge_graph.graph_store import get_graph_store
    from app.services.document_service import _load_chunks_for_kg, _load_document_text_for_kg
    from app.db.session import SessionLocal
    from app.entities.database import Document

    conflict_detector = _NullConflictDetector() if args.skip_conflicts else None
    extractor = KGExtractor(
        store=get_graph_store(),
        embedding_fn=get_vector_store().embed_query,
        chunks_loader=_load_chunks_for_kg,
        document_loader=_load_document_text_for_kg,
        conflict_detector=conflict_detector,
    )

    def get_all_document_ids() -> list[int]:
        with SessionLocal() as db:
            return [d.id for d in db.query(Document).all()]

    doc_ids = [args.document_id] if args.document_id else get_all_document_ids()
    failed = _run_backfill(extractor, doc_ids)

    total = len(doc_ids)
    print(f"\nDone: {total - failed}/{total} succeeded")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
