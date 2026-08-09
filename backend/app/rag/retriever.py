"""
检索器

包含稀疏检索（BM25）、重排序、混合检索
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import time
import jieba
from rank_bm25 import BM25Okapi

from app.core.observability import get_tracer

tracer = get_tracer("rag.retrieve")


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass


class SparseRetriever(BaseRetriever):
    """BM25 稀疏检索器"""

    def __init__(self):
        self.bm25 = None
        self.documents: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self._tokenized_docs: List[List[str]] = []
        self._lock = threading.Lock()

    def add_documents(self, documents: List[str], metadata_list: List[Dict[str, Any]] = None) -> None:
        if not documents:
            return
        with self._lock:
            self.documents.extend(documents)
            self.metadata_list.extend(metadata_list if metadata_list else [{} for _ in documents])
            self._tokenized_docs.extend([self._tokenize(doc) for doc in documents])
            self.bm25 = BM25Okapi(self._tokenized_docs)

    def remove_documents(self, document_id: int) -> None:
        with self._lock:
            indices_to_remove = [i for i, meta in enumerate(self.metadata_list) if meta.get("document_id") == document_id]
            for i in reversed(indices_to_remove):
                del self.documents[i]
                del self.metadata_list[i]
                del self._tokenized_docs[i]
            self.bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

    def clear(self) -> None:
        with self._lock:
            self.bm25 = None
            self.documents = []
            self.metadata_list = []
            self._tokenized_docs = []

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.bm25 or not self.documents:
                return []
            scores = self.bm25.get_scores(self._tokenize(query))
            results = []
            for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]:
                if score > 0:
                    results.append({"content": self.documents[idx], "metadata": self.metadata_list[idx], "score": float(score), "sparse_score": float(score)})
            return results

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut_for_search(text))


# 全局模型缓存
_loaded_models = {}
_loaded_models_lock = threading.Lock()


class Reranker:
    """CrossEncoder 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return
            with _loaded_models_lock:
                if self.model_name in _loaded_models:
                    self._model = _loaded_models[self.model_name]
                    return
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name)
                    _loaded_models[self.model_name] = self._model
                    print(f"Reranker model loaded: {self.model_name}")
                except Exception as e:
                    print(f"Reranker load failed: {e}")
                    self._model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
        self._load_model()
        if self._model is None:
            return self._simple_rerank(query, documents, top_n)
        return self._model_rerank(query, documents, top_n)

    def _simple_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        for doc in documents:
            content = doc.get("content", "").lower()
            overlap = len(query_words & set(content.split()))
            doc["rerank_score"] = doc.get("score", 0) * 0.7 + overlap * 0.3
        return sorted(documents, key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True)[:top_n]

    def _model_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        pairs = [(query, doc.get("content", "")) for doc in documents]
        with _loaded_models_lock:
            scores = self._model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        return sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_n]


class HybridRetriever(BaseRetriever):
    """
    混合检索器 - RAG 核心检索组件

    结合密集向量检索和稀疏检索（BM25），使用 RRF 融合和 CrossEncoder 重排序
    """

    def __init__(
        self,
        vector_store,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,  # RRF 平滑常数
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        top_n: int = 5
    ):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k  # RRF 参数
        self.use_reranker = use_reranker
        self.top_n = top_n

        if use_reranker:
            self.reranker = Reranker(model_name=reranker_model)
        else:
            self.reranker = None

        self.vector_store.connect()

    def retrieve(self, query: str, top_k: int = 10, background_tasks=None) -> tuple:
        debug_info = {
            "query": query,
            "steps": [],
            "dense_results": 0,
            "sparse_results": 0,
            "chunks_by_type": {"small": 0, "medium": 0, "large": 0},
            "rerank_used": False,
            "detail": {"dense_by_type": {}, "sparse_results": [], "merged_results": [], "deduped_results": [], "reranked_results": []}
        }

        # 1. 密集向量检索（只查小块）
        with tracer.start_as_current_span("rag.retrieve.embedding") as span:
            embedding_start = time.time()
            query_vector = self.vector_store.embed_query(query)
            embedding_time = time.time() - embedding_start
            span.set_attribute("retrieve.vector_dim", len(query_vector))
            span.set_attribute("retrieve.embedding_time_s", round(embedding_time, 3))
            debug_info["steps"].append({"step": "embedding", "desc": "查询向量化", "vector_dim": len(query_vector), "time_s": round(embedding_time, 3)})

        filter_expr = 'chunk_type == "small" && status == "active"'
        with tracer.start_as_current_span("rag.retrieve.dense") as span:
            dense_start = time.time()
            results = self.vector_store.search_vectors(collection_name="chunks", query_vector=query_vector, top_k=top_k, filter_expr=filter_expr)
            dense_time = time.time() - dense_start
            span.set_attribute("retrieve.dense_count", len(results))
            span.set_attribute("retrieve.dense_time_s", round(dense_time, 3))

        dense_results = []
        dense_by_type = {"small": [], "medium": [], "large": []}
        for r in results:
            r["chunk_type"] = "small"
            r["dense_score"] = r.get("score", 0)
            dense_by_type["small"].append({"content": r.get("content", ""), "dense_score": round(r.get("dense_score", 0), 4), "chunk_type": "small"})
        dense_results.extend(results)

        debug_info["dense_results"] = len(dense_results)
        debug_info["chunks_by_type"]["small"] = len(results)
        debug_info["detail"]["dense_by_type"] = dense_by_type
        debug_info["steps"].append({"step": "dense_search", "desc": "密集向量检索（仅小块）", "count": len(dense_results), "time_s": round(dense_time, 3)})

        # 2. 稀疏检索（BM25）
        with tracer.start_as_current_span("rag.retrieve.sparse") as span:
            sparse_start = time.time()
            sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k)
            sparse_time = time.time() - sparse_start
            span.set_attribute("retrieve.sparse_count", len(sparse_results))
            span.set_attribute("retrieve.sparse_time_s", round(sparse_time, 3))
        for r in sparse_results:
            r["sparse_score"] = r.pop("score", 0)
        debug_info["sparse_results"] = len(sparse_results)
        debug_info["detail"]["sparse_results"] = [{"content": r.get("content", ""), "sparse_score": round(r.get("sparse_score", 0), 4)} for r in sparse_results]
        debug_info["steps"].append({"step": "sparse_search", "desc": "稀疏检索（BM25）", "count": len(sparse_results), "time_s": round(sparse_time, 3)})

        # 3. RRF 排名融合
        with tracer.start_as_current_span("rag.retrieve.merge") as span:
            merge_start = time.time()
            all_results = self._merge_with_rrf(dense_results, sparse_results, top_k)
            merge_time = time.time() - merge_start
            span.set_attribute("retrieve.merged_count", len(all_results))
            span.set_attribute("retrieve.rrf_k", self.rrf_k)
        debug_info["detail"]["merged_results"] = [
            {"content": r.get("content", ""), "rrf_score": round(r.get("rrf_score", 0), 4)}
            for r in sorted(all_results, key=lambda x: x.get("rrf_score", 0), reverse=True)[:10]
        ]
        debug_info["steps"].append({"step": "merge", "desc": f"RRF 排名融合（k={self.rrf_k}）", "count": len(all_results), "time_s": round(merge_time, 3)})

        # 4. 去重
        with tracer.start_as_current_span("rag.retrieve.dedup") as span:
            dedup_start = time.time()
            unique_results = []
            seen_contents = set()
            for result in sorted(all_results, key=lambda x: x.get("rrf_score", 0), reverse=True):
                content = result.get("content", "")
                if content not in seen_contents:
                    seen_contents.add(content)
                    unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
            dedup_time = time.time() - dedup_start
            span.set_attribute("retrieve.dedup_count", len(unique_results))
        debug_info["detail"]["deduped_results"] = [{"content": r.get("content", ""), "rrf_score": round(r.get("rrf_score", 0), 4), "chunk_type": r.get("chunk_type", "unknown")} for r in unique_results]
        debug_info["steps"].append({"step": "dedup", "desc": "去重", "count": len(unique_results), "time_s": round(dedup_time, 3)})

        # 5. 重排序
        rerank_time = 0
        if self.use_reranker and self.reranker and len(unique_results) > 0:
            with tracer.start_as_current_span("rag.retrieve.rerank") as span:
                rerank_start = time.time()
                reranked_results = self.reranker.rerank(query, unique_results.copy(), self.top_n)
                rerank_time = time.time() - rerank_start
                span.set_attribute("retrieve.rerank_count", len(reranked_results))
                span.set_attribute("retrieve.rerank_time_s", round(rerank_time, 3))
                span.set_attribute("retrieve.reranker_model", self.reranker.model_name)
            debug_info["detail"]["reranked_results"] = [
                {"content": r.get("content", ""), "rrf_score": round(r.get("rrf_score", 0), 4),
                 "rerank_score": round(r.get("rerank_score", 0), 4) if r.get("rerank_score") else None, "chunk_type": r.get("chunk_type", "unknown")}
                for r in reranked_results
            ]
            unique_results = reranked_results
            debug_info["rerank_used"] = True
        debug_info["steps"].append({"step": "rerank", "desc": "CrossEncoder 重排序", "count": len(unique_results), "time_s": round(rerank_time, 3)})

        # 注册统计更新（fire-and-forget）
        # db session 在 wrapper 内部创建，避免注册时创建到执行时过期
        if background_tasks is not None and unique_results:
            background_tasks.add_task(self._update_stats_wrapper, unique_results)

        return unique_results, debug_info

    @staticmethod
    def _update_stats_wrapper(hits):
        """包装器：创建独立 db session，执行统计更新，关闭"""
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            # 归一化分数字段：优先 rerank_score，其次 rrf_score，最后 0
            normalized = []
            for h in hits:
                normalized.append({
                    "id": h.get("id"),
                    "score": h.get("rerank_score") or h.get("rrf_score") or h.get("dense_score") or 0.0
                })
            update_chunk_stats(db, normalized)
        finally:
            db.close()

    def _merge_with_rrf(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int) -> List[Dict]:
        """
        使用 RRF（Reciprocal Rank Fusion）融合两个检索器的结果
        
        RRF 公式：score(d) = Σ(1 / (k + rank_i(d)))
        
        Args:
            dense_results: 向量检索结果（已按分数降序）
            sparse_results: BM25 检索结果（已按分数降序）
            top_k: 取前 top_k 个结果用于融合
        
        Returns:
            融合后的结果列表（带 rrf_score）
        """
        # 创建内容到文档的映射
        content_map: Dict[str, Dict] = {}
        
        # 处理密集检索结果（按排名计算 RRF 分数）
        for rank, result in enumerate(dense_results[:top_k], start=1):
            content = result.get("content", "")
            if content not in content_map:
                content_map[content] = result.copy()
                content_map[content]["rrf_score"] = 0.0
            # RRF 公式：1 / (k + rank)
            content_map[content]["rrf_score"] += 1 / (self.rrf_k + rank)
        
        # 处理稀疏检索结果（按排名计算 RRF 分数）
        for rank, result in enumerate(sparse_results[:top_k], start=1):
            content = result.get("content", "")
            if content not in content_map:
                content_map[content] = result.copy()
                content_map[content]["rrf_score"] = 0.0
            # RRF 公式：1 / (k + rank)
            content_map[content]["rrf_score"] += 1 / (self.rrf_k + rank)
        
        # 转换为列表并按 RRF 分数降序排序
        results = list(content_map.values())
        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        return results

    def add_to_sparse_index(self, documents: List[str], metadata_list: List[Dict]) -> None:
        self.sparse_retriever.add_documents(documents, metadata_list)

    def remove_from_sparse_index(self, document_id: int) -> None:
        self.sparse_retriever.remove_documents(document_id)


def update_chunk_stats(db, hits):
    """
    检索命中后更新 chunk 统计（access_count / last_accessed_at / total_score / avg_score）。
    由 BackgroundTasks 调用，fire-and-forget。
    """
    from app.entities.database import DocumentChunk
    for hit in hits:
        chunk = db.query(DocumentChunk).filter_by(milvus_id=hit.get("id")).first()
        if not chunk:
            continue
        chunk.access_count += 1
        chunk.last_accessed_at = datetime.utcnow()
        chunk.hit_count += 1
        chunk.total_score += float(hit.get("score", 0.0))
        chunk.avg_score = chunk.total_score / chunk.hit_count if chunk.hit_count > 0 else 0.0
    try:
        db.commit()
    except Exception as e:
        print(f"[update_chunk_stats] failed: {e}")
        db.rollback()
