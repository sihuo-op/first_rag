import threading

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.rag.retriever import HybridRetriever, SparseRetriever
from app.rag.vector_store import MilvusStore
from app.services.cold_knowledge_service import ColdKnowledgeService
from app.services.conflict_service import ConflictService

_settings = get_settings()
_singleton_lock = threading.RLock()
_vector_store_instance: MilvusStore = None
_sparse_retriever_instance: SparseRetriever = None
_retriever_instance: HybridRetriever = None


def get_vector_store() -> MilvusStore:
    """获取 MilvusStore 实例（单例）"""
    global _vector_store_instance
    if _vector_store_instance is None:
        with _singleton_lock:
            if _vector_store_instance is None:
                _vector_store_instance = MilvusStore(
                    host=_settings.MILVUS_HOST,
                    port=_settings.MILVUS_PORT,
                    collection_prefix=_settings.MILVUS_COLLECTION_PREFIX,
                    embedding_model=_settings.SENTENCE_TRANSFORMER_MODEL,
                    dimension=_settings.EMBEDDING_DIMENSION
                )
    return _vector_store_instance


def get_sparse_retriever() -> SparseRetriever:
    """获取稀疏检索器实例（单例）"""
    global _sparse_retriever_instance
    if _sparse_retriever_instance is None:
        with _singleton_lock:
            if _sparse_retriever_instance is None:
                _sparse_retriever_instance = SparseRetriever()
    return _sparse_retriever_instance


def _build_kg_retriever(vector_store: MilvusStore):
    """按配置构造 KGRetriever（KG 第三路检索）。

    - KG_ENABLED=False：返回 None（HybridRetriever 保持旧两路行为）；
    - Neo4j 连接失败等任何初始化异常：返回 None 回退两路，
      KG 绝不拖垮主检索链路（与 KGRetriever 内部失败回退同一约束）。
    """
    if not _settings.KG_ENABLED:
        return None
    try:
        from app.knowledge_graph.graph_store import get_graph_store
        from app.knowledge_graph.kg_retriever import KGRetriever
        return KGRetriever(
            store=get_graph_store(),
            vector_store=vector_store,
            similarity_threshold=_settings.KG_CONCEPT_SIMILARITY_THRESHOLD,
            max_depth=_settings.KG_MULTI_HOP_DEPTH,
        )
    except Exception as e:
        print(f"[get_retriever] KG 初始化失败，回退为 dense+sparse 两路: {e}")
        return None


def get_retriever() -> HybridRetriever:
    """获取混合检索器实例（单例）"""
    global _retriever_instance
    if _retriever_instance is None:
        with _singleton_lock:
            if _retriever_instance is None:
                vector_store = get_vector_store()
                sparse_retriever = get_sparse_retriever()
                _retriever_instance = HybridRetriever(
                    vector_store=vector_store,
                    sparse_retriever=sparse_retriever,
                    use_reranker=_settings.RERANKER_ENABLED,
                    reranker_model=_settings.RERANKER_MODEL,
                    top_n=_settings.RERANKER_TOP_N,
                    rerank_max_candidates=_settings.RERANK_MAX_CANDIDATES,
                    rerank_max_length=_settings.RERANKER_MAX_LENGTH,
                    kg_retriever=_build_kg_retriever(vector_store),
                )
    return _retriever_instance


def get_conflict_service(
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store)
) -> ConflictService:
    """获取 ConflictService 实例（每请求新建，绑定到当前 db session）"""
    return ConflictService(db, vector_store)


def get_cold_knowledge_service(
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store)
) -> ColdKnowledgeService:
    """获取 ColdKnowledgeService 实例（每请求新建，绑定到当前 db session）"""
    return ColdKnowledgeService(db, vector_store)
