"""
检索器

包含稀疏检索（BM25）、重排序、混合检索
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import os
import re
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
    """CrossEncoder 重排序器（ONNX Runtime int8 量化）

    用 optimum 导出为 ONNX + AVX2 动态 int8 量化，替代 PyTorch FP32 推理。
    首次加载会做导出+量化（5-10s），结果缓存到 ~/.cache/huggingface/onnx/<model>-int8/。
    输出仍是 raw logits（与原 CrossEncoder.predict() 默认行为一致），不破坏 steps.py 的阈值标定。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return
            with _loaded_models_lock:
                if self.model_name in _loaded_models:
                    self._model, self._tokenizer = _loaded_models[self.model_name]
                    return
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    from transformers import AutoTokenizer

                    cache_dir = os.path.expanduser(
                        f"~/.cache/huggingface/onnx/{self.model_name.replace('/', '--')}-int8"
                    )

                    if os.path.exists(cache_dir):
                        print(f"Loading cached ONNX int8 reranker from {cache_dir}")
                        self._model = ORTModelForSequenceClassification.from_pretrained(
                            cache_dir,
                            provider="CPUExecutionProvider",
                        )
                    else:
                        print(f"Exporting {self.model_name} to ONNX...")
                        base_model = ORTModelForSequenceClassification.from_pretrained(
                            self.model_name,
                            export=True,
                            provider="CPUExecutionProvider",
                        )
                        print(f"Quantizing to int8 (AVX2 dynamic)...")
                        quantizer = ORTQuantizer.from_pretrained(base_model)
                        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
                        os.makedirs(cache_dir, exist_ok=True)
                        quantizer.quantize(qconfig, save_dir=cache_dir)
                        self._model = ORTModelForSequenceClassification.from_pretrained(
                            cache_dir,
                            provider="CPUExecutionProvider",
                        )

                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    _loaded_models[self.model_name] = (self._model, self._tokenizer)
                    print(f"Reranker model loaded (ONNX int8): {self.model_name}")
                except Exception as e:
                    print(f"Reranker ONNX load failed: {e}")
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
        import torch
        pairs = [(query, doc.get("content", "")) for doc in documents]
        with _loaded_models_lock:
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()
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

    def retrieve(self, query: str, top_k: int = 10) -> tuple:
        debug_info = {
            "query": query,
            "steps": [],
            "dense_results": 0,
            "sparse_results": 0,
            "chunks_by_type": {"small": 0, "medium": 0, "large": 0},
            "rerank_used": False,
            "detail": {"dense_by_type": {}, "sparse_results": [], "merged_results": [], "deduped_results": [], "reranked_results": []}
        }

        # 1. 密集向量检索（所有块类型，利用父子块提供更多上下文）
        with tracer.start_as_current_span("rag.retrieve.embedding") as span:
            embedding_start = time.time()
            query_vector = self.vector_store.embed_query(query)
            embedding_time = time.time() - embedding_start
            span.set_attribute("retrieve.vector_dim", len(query_vector))
            span.set_attribute("retrieve.embedding_time_s", round(embedding_time, 3))
            debug_info["steps"].append({"step": "embedding", "desc": "查询向量化", "vector_dim": len(query_vector), "time_s": round(embedding_time, 3)})

        with tracer.start_as_current_span("rag.retrieve.dense") as span:
            dense_start = time.time()
            results = self.vector_store.search_vectors(collection_name="chunks", query_vector=query_vector, top_k=top_k)
            dense_time = time.time() - dense_start
            span.set_attribute("retrieve.dense_count", len(results))
            span.set_attribute("retrieve.dense_time_s", round(dense_time, 3))

        dense_results = []
        dense_by_type = {"small": [], "medium": [], "large": []}
        for r in results:
            chunk_type = r.get("chunk_type", "small")
            r["dense_score"] = r.get("score", 0)
            dense_by_type.setdefault(chunk_type, []).append({"content": r.get("content", "")[:100], "dense_score": round(r.get("dense_score", 0), 4), "chunk_type": chunk_type})
            debug_info["chunks_by_type"][chunk_type] = debug_info["chunks_by_type"].get(chunk_type, 0) + 1
        dense_results.extend(results)

        debug_info["dense_results"] = len(dense_results)
        debug_info["detail"]["dense_by_type"] = dense_by_type
        debug_info["steps"].append({"step": "dense_search", "desc": "密集向量检索（所有块类型）", "count": len(dense_results), "time_s": round(dense_time, 3)})

        # 2. 稀疏检索（BM25）
        with tracer.start_as_current_span("rag.retrieve.sparse") as span:
            sparse_start = time.time()
            sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k)
            sparse_time = time.time() - sparse_start
            span.set_attribute("retrieve.sparse_count", len(sparse_results))
            span.set_attribute("retrieve.sparse_time_s", round(sparse_time, 3))
        for r in sparse_results:
            r["sparse_score"] = r.pop("score", 0)
            # chunk_type 在 metadata 里，提到顶层方便后续 RRF 合并和晋升逻辑读取
            if "chunk_type" not in r:
                meta = r.get("metadata") or {}
                if meta.get("chunk_type"):
                    r["chunk_type"] = meta["chunk_type"]
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

        # 4. 去重（精确内容 + 父子块晋升）
        with tracer.start_as_current_span("rag.retrieve.dedup") as span:
            dedup_start = time.time()
            # 4a. 精确内容去重
            exact_dedup = []
            seen_contents = set()
            for result in sorted(all_results, key=lambda x: x.get("rrf_score", 0), reverse=True):
                content = result.get("content", "")
                if content not in seen_contents:
                    seen_contents.add(content)
                    exact_dedup.append(result)
            # 4b. 父子块晋升：2+ smalls 同属一个 medium -> 丢 smalls 保 medium；mediums->large 同理
            unique_results = self._dedup_by_containment(exact_dedup)
            dedup_time = time.time() - dedup_start
            promo_stats = getattr(self, "_last_promotion_stats", {"small_to_medium": 0, "medium_to_large": 0})
            span.set_attribute("retrieve.dedup_count", len(unique_results))
            span.set_attribute("retrieve.promoted_small_to_medium", promo_stats["small_to_medium"])
            span.set_attribute("retrieve.promoted_medium_to_large", promo_stats["medium_to_large"])
            span.set_attribute("retrieve.containment_dropped", len(exact_dedup) - len(unique_results))
        debug_info["detail"]["deduped_results"] = [{"content": r.get("content", "")[:200], "rrf_score": round(r.get("rrf_score", 0), 4), "chunk_type": r.get("chunk_type", "unknown")} for r in unique_results]
        debug_info["steps"].append({"step": "dedup", "desc": f"去重+晋升（small->medium: {promo_stats['small_to_medium']} 次, medium->large: {promo_stats['medium_to_large']} 次, 丢弃 {len(exact_dedup) - len(unique_results)} 个子块）", "count": len(unique_results), "time_s": round(dedup_time, 3)})

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

    def _dedup_by_containment(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        父子块晋升：同一父块下 >=2 个子块被检索到时，丢弃子块保留父块

        - 2+ smalls 内容包含于同一 medium -> 丢 smalls，保留 medium
        - 2+ mediums 内容包含于同一 large -> 丢 mediums，保留 large
        - 单个 small/medium 不晋升（保留精确匹配）
        - 父子关系通过内容包含判断（归一化空白后做子串匹配，避免全角/半角空格差异导致漏判）

        Args:
            results: 已精确去重的结果列表

        Returns:
            晋升后的结果列表（按 RRF 分数降序）
        """
        if len(results) <= 1:
            return results

        def _normalize(s: str) -> str:
            return re.sub(r"\s+", "", s)

        indexed = list(enumerate(results))
        smalls = [(i, r) for i, r in indexed if r.get("chunk_type") == "small"]
        mediums = [(i, r) for i, r in indexed if r.get("chunk_type") == "medium"]
        larges = [(i, r) for i, r in indexed if r.get("chunk_type") == "large"]

        to_drop: set = set()
        self._last_promotion_stats = {"small_to_medium": 0, "medium_to_large": 0}

        # small -> medium 晋升
        medium_to_smalls: Dict[int, List[int]] = defaultdict(list)
        for s_idx, s in smalls:
            s_norm = _normalize(s.get("content", ""))
            if not s_norm:
                continue
            for m_idx, m in mediums:
                m_norm = _normalize(m.get("content", ""))
                if s_norm != m_norm and s_norm in m_norm:
                    medium_to_smalls[m_idx].append(s_idx)
                    break  # small 只可能属于一个 medium
        for m_idx, s_indices in medium_to_smalls.items():
            if len(s_indices) >= 2:
                to_drop.update(s_indices)
                self._last_promotion_stats["small_to_medium"] += 1

        # medium -> large 晋升
        large_to_mediums: Dict[int, List[int]] = defaultdict(list)
        for m_idx, m in mediums:
            m_norm = _normalize(m.get("content", ""))
            if not m_norm:
                continue
            for l_idx, l in larges:
                l_norm = _normalize(l.get("content", ""))
                if m_norm != l_norm and m_norm in l_norm:
                    large_to_mediums[l_idx].append(m_idx)
                    break
        for l_idx, m_indices in large_to_mediums.items():
            if len(m_indices) >= 2:
                to_drop.update(m_indices)
                self._last_promotion_stats["medium_to_large"] += 1

        promoted = [r for i, r in enumerate(results) if i not in to_drop]
        promoted.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return promoted

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
