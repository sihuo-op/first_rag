"""
Milvus 迁移脚本：drop 旧 chunks collection，用新 schema 重建，重处理所有 COMPLETED 文档。

执行方式：
    cd backend
    python scripts/migrate_milvus.py

注意：会重新 embedding，耗时取决于文档数量。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.db.session import SessionLocal
from app.entities.database import Document, DocumentChunk, DocumentStatus, ChunkType
from app.entities.schemas import DocumentCreate
from app.rag.parsers import get_parser
from app.rag.splitter import ThreeLayerSplitter
from app.services.document_service import DocumentService
import hashlib


def main():
    settings = get_settings()
    vector_store = get_vector_store()
    db = SessionLocal()
    splitter = ThreeLayerSplitter()

    try:
        print("=" * 50)
        print("Milvus 迁移：drop + 重建 + 重处理")
        print("=" * 50)

        # 1. drop 旧 collection
        print("[1/3] Dropping chunks collection...")
        if vector_store.has_collection("chunks"):
            vector_store.drop_collection("chunks")
        print("Dropped")

        # 2. 用新 schema 重建
        print("[2/3] Creating chunks collection with new schema...")
        vector_store.create_collection("chunks")
        print("Created")

        # 3. 重处理所有 COMPLETED 文档
        print("[3/3] Reprocessing documents...")
        docs = db.query(Document).filter(Document.status == DocumentStatus.COMPLETED).all()
        print(f"Found {len(docs)} documents to reprocess")

        for i, doc in enumerate(docs, 1):
            print(f"  [{i}/{len(docs)}] Doc {doc.id}: {doc.title}")
            try:
                # 解析 + 切分
                parser = get_parser(doc.file_type)
                content, _ = parser.parse(doc.file_path)
                chunks = splitter.split(content)

                # 删除旧 chunk 记录
                db.query(DocumentChunk).filter_by(document_id=doc.id).delete()
                db.commit()

                # 写入 Milvus（带 content_hash）
                texts = []
                metadata_list = []
                for chunk in chunks:
                    text = chunk["content"]
                    texts.append(text)
                    metadata_list.append({
                        "document_id": doc.id,
                        "chunk_type": chunk["chunk_type"],
                        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
                    })
                milvus_ids = vector_store.add_texts("chunks", texts, metadata_list)

                # 写入 PG
                for idx, chunk in enumerate(chunks):
                    db_chunk = DocumentChunk(
                        document_id=doc.id,
                        content=chunk["content"],
                        chunk_type=getattr(ChunkType, chunk["chunk_type"].upper()),
                        position=chunk.get("position", idx),
                        milvus_id=milvus_ids[idx],
                        content_hash=metadata_list[idx]["content_hash"],
                        status="active"
                    )
                    db.add(db_chunk)
                db.commit()
                print(f"    -> {len(texts)} chunks reinserted")
            except Exception as e:
                print(f"    ERROR: {e}")
                db.rollback()

        print("=" * 50)
        print("Migration complete!")
        print("=" * 50)
    finally:
        db.close()


if __name__ == "__main__":
    main()
