from typing import Optional, List, Dict, Any
import os
import uuid
import hashlib
import logging
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
from app.core.dependencies import get_vector_store
from app.knowledge_graph.extractor import KGExtractor
from app.knowledge_graph.graph_store import get_graph_store

logger = logging.getLogger(__name__)

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

    def get_document_by_id(self, doc_id: int, user_id: Optional[int] = None) -> Optional[Document]:
        """获取单个文档（不做状态过滤，status 端点需能看到 pending/failed）。"""
        query = self.db.query(Document).filter(Document.id == doc_id)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.first()

    async def upload_document(
        self,
        file: UploadFile,
        user_id: int,
        title: str = None,
        background_tasks: BackgroundTasks = None
    ) -> Document:
        """
        上传并处理文档

        Args:
            file: 上传的文件
            user_id: 用户 ID
            title: 文档标题（可选，默认用文件名）
            background_tasks: 后台任务处理器

        Returns:
            创建的文档记录
        """
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext.lstrip(".") not in settings.allowed_extensions_list:
            raise ValueError(f"不支持的文件类型: {file_ext}")

        file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        db_document = Document(
            user_id=user_id,
            title=title or file.filename,
            file_name=file.filename,
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
            content, _ = parser.parse(document.file_path)

            chunks = self.splitter.split(content)

            if not self.vector_store.has_collection("chunks"):
                self.vector_store.create_collection("chunks")

            texts = []
            metadata_list = []
            for chunk in chunks:
                text = chunk["content"]
                texts.append(text)
                metadata_list.append({
                    "document_id": doc_id,
                    "chunk_type": chunk["chunk_type"],
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", 0),
                })

            milvus_ids = self.vector_store.add_texts("chunks", texts, metadata_list)

            for idx, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    chunk_type=getattr(ChunkType, chunk["chunk_type"].upper()),
                    position=chunk.get("position", idx),
                    milvus_id=milvus_ids[idx],
                    content_hash=metadata_list[idx]["content_hash"],
                    char_start=chunk.get("char_start"),
                    char_end=chunk.get("char_end"),
                    status="active"
                )
                self.db.add(db_chunk)

            document.status = DocumentStatus.COMPLETED
            self.db.commit()

            # 触发冲突检测（后台任务）
            if self.conflict_service and background_tasks:
                # 标记文档待检测
                document.conflict_check_status = "pending"
                self.db.commit()
                background_tasks.add_task(self._run_conflict_detection, doc_id)

            # 触发 KG 抽取（后台任务，失败不影响文档状态）
            if background_tasks:
                _trigger_kg_extraction(str(doc_id), background_tasks)

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


# ---------------------------------------------------------------------------
# KG 抽取触发（文档处理成功后由 process_document 调用）
# ---------------------------------------------------------------------------

def _load_chunks_for_kg(doc_id: str) -> list[dict]:
    """加载文档 chunks 供 KG 抽取（独立 db session，后台线程安全）。

    id 使用 Milvus 主键（DocumentChunk.milvus_id）而非 PG 自增主键：
    Article.chunk_ids 最终经 kg_retriever -> get_chunks_by_ids 反查 Milvus，
    而 Milvus id 是 uuid4 VARCHAR（vector_store.insert_vectors 生成），
    用 PG int 会查不到任何 chunk，KG 检索路径静默失效。
    历史文档 milvus_id 为 NULL 时回退 PG id（保持抽取内部匹配一致，
    反查为空即退化，与既有 legacy 限制一致）。
    """
    from app.db.session import SessionLocal
    with SessionLocal() as db:
        chunks = db.query(DocumentChunk).filter_by(document_id=int(doc_id)).all()
        return [
            {
                "id": c.milvus_id or c.id,
                "content": c.content,
                "char_start": c.char_start or 0,
                "char_end": c.char_end or 0,
                "document_id": c.document_id,
            }
            for c in chunks
        ]


def _load_document_text_for_kg(doc_id: str) -> str:
    """重新解析文件获取全文（Document 无 full_text 字段）。"""
    from app.db.session import SessionLocal
    with SessionLocal() as db:
        doc = db.query(Document).filter(Document.id == int(doc_id)).first()
        if not doc:
            return ""
        parser = get_parser(doc.file_type)
        content, _ = parser.parse(doc.file_path)
        return content


def _trigger_kg_extraction(document_id: str, background_tasks) -> None:
    """文档处理成功后异步触发 KG 抽取。KG 侧任何失败只打日志，不影响文档流程。"""
    try:
        if not get_settings().KG_ENABLED:
            return
        extractor = KGExtractor(
            store=get_graph_store(),
            embedding_fn=get_vector_store().embed_query,
            chunks_loader=_load_chunks_for_kg,
            document_loader=_load_document_text_for_kg,
        )
        background_tasks.add_task(extractor.run, document_id=document_id)
    except Exception as e:
        logger.warning(f"KG extraction trigger failed for document {document_id}: {e}")
