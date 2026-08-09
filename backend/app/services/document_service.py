from typing import Optional, List, Dict, Any
import os
import uuid
import hashlib
from sqlalchemy.orm import Session
from fastapi import UploadFile, BackgroundTasks
from app.entities.database import Document, DocumentChunk, DocumentStatus, ChunkType
from app.entities.schemas import DocumentCreate, DocumentUpdate
from app.rag.parsers import get_parser
from app.rag.splitter import ThreeLayerSplitter
from app.rag.vector_store import MilvusStore
from app.rag.retriever import HybridRetriever
from app.services.conflict_service import ConflictService
from app.core.config import get_settings

settings = get_settings()


class DocumentService:
    """
    文档管理服务

    负责文档的上传、解析、切分、向量化和删除等全流程管理。

    处理流程：
    1. upload_document: 保存文件到磁盘，创建数据库记录
    2. process_document: 解析文档内容 -> 三层切分 -> 向量化 -> 存入 Milvus + BM25 索引
    3. delete_document: 清理向量库、BM25 索引、删除文件、删除数据库记录

    三层切分策略：
    - LARGE (~2000字): 完整语义单元，提供上下文
    - MEDIUM (~500字): 句子组，平衡粒度
    - SMALL (~150字): 精确匹配，适合短查询

    混合检索：
    - 密集向量：语义理解，由 Milvus 支持
    - 稀疏索引：关键词匹配，由 BM25 支持

    依赖：
    - ThreeLayerSplitter: 文档切分器
    - MilvusStore: 向量存储（集成 embedding）
    - HybridRetriever: 混合检索器（包含 BM25 索引）
    """

    def __init__(
        self,
        db: Session,
        vector_store: MilvusStore,
        retriever: HybridRetriever = None,
        conflict_service: ConflictService = None
    ):
        """
        初始化文档服务

        Args:
            db: 数据库会话
            vector_store: 向量存储实例（集成 embedding）
            retriever: 混合检索器实例（可选，用于检索）
            conflict_service: 冲突检测服务实例（可选，用于上传后触发冲突检测）
        """
        self.db = db
        self.vector_store = vector_store
        self.retriever = retriever
        self.conflict_service = conflict_service
        self.splitter = ThreeLayerSplitter()

    def get_documents(self, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[Document]:
        """
        获取文档列表

        Args:
            user_id: 用户 ID，为空则获取所有文档
            skip: 跳过的记录数（分页）
            limit: 返回的记录数（分页）

        Returns:
            文档列表
        """
        query = self.db.query(Document).filter(Document.status == DocumentStatus.COMPLETED)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()

    async def upload_document(
        self,
        file: UploadFile,
        user_id: int,
        background_tasks: BackgroundTasks = None
    ) -> Document:
        """
        上传并处理文档

        Args:
            file: 上传的文件
            user_id: 用户 ID
            background_tasks: 后台任务处理器

        Returns:
            创建的文档记录
        """
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {file_ext}")

        file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}{file_ext}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        db_document = Document(
            user_id=user_id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_ext.lstrip('.'),
            status=DocumentStatus.PENDING
        )
        self.db.add(db_document)
        self.db.commit()
        self.db.refresh(db_document)

        if background_tasks:
            background_tasks.add_task(self.process_document, db_document.id, background_tasks)
        else:
            self.process_document(db_document.id)

        return db_document

    def process_document(self, doc_id: int, background_tasks=None) -> bool:
        """
        处理文档：解析 -> 切分 -> 向量化 -> 存储

        Args:
            doc_id: 文档 ID
            background_tasks: 后台任务处理器（用于触发冲突检测）

        Returns:
            处理是否成功
        """
        document = self.db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            return False

        try:
            parser = get_parser(document.file_type)
            content = parser.parse(document.file_path)

            chunks = self.splitter.split_text(content)

            if not self.vector_store.has_collection("chunks"):
                self.vector_store.create_collection("chunks")

            texts = []
            metadata_list = []
            for i, chunk in enumerate(chunks):
                for chunk_type in ['large', 'medium', 'small']:
                    text = chunk[chunk_type]
                    texts.append(text)
                    metadata_list.append({
                        "document_id": doc_id,
                        "chunk_type": chunk_type,
                        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
                    })

            milvus_ids = self.vector_store.add_texts("chunks", texts, metadata_list)

            idx = 0
            for i, chunk in enumerate(chunks):
                for chunk_type in ['large', 'medium', 'small']:
                    db_chunk = DocumentChunk(
                        document_id=doc_id,
                        content=chunk[chunk_type],
                        chunk_type=getattr(ChunkType, chunk_type.upper()),
                        position=i,
                        milvus_id=milvus_ids[idx],
                        content_hash=metadata_list[idx]["content_hash"],
                        status="active"
                    )
                    self.db.add(db_chunk)
                    idx += 1

            document.status = DocumentStatus.COMPLETED
            self.db.commit()

            # 触发冲突检测（后台任务）
            if self.conflict_service and background_tasks:
                # 标记文档待检测
                document.conflict_check_status = "pending"
                self.db.commit()
                background_tasks.add_task(self._run_conflict_detection, doc_id)

            return True

        except Exception as e:
            document.status = DocumentStatus.FAILED
            self.db.commit()
            raise e

    def _run_conflict_detection(self, doc_id: int):
        """后台执行冲突检测（独立 db session）"""
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            svc = ConflictService(db, self.vector_store)
            svc.detect_for_document(doc_id)
        except Exception as e:
            print(f"[DocumentService] conflict detection failed for doc {doc_id}: {e}")
        finally:
            db.close()

    def delete_document(self, doc_id: int) -> bool:
        """
        删除文档及其相关数据

        Args:
            doc_id: 文档 ID

        Returns:
            删除是否成功
        """
        document = self.db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            return False

        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)

            self.vector_store.delete_vectors("chunks", filter_expr=f"document_id == {doc_id}")
            if self.retriever:
                self.retriever.remove_from_sparse_index(doc_id)

            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()

            self.db.delete(document)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False

    def get_document_chunks(self, doc_id: int) -> List[DocumentChunk]:
        """获取文档的所有块"""
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id
        ).order_by(DocumentChunk.chunk_index).all()
