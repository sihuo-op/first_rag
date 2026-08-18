"""
检索器

包含稀疏检索（BM25）、重排序、混合检索
"""

import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

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

    def add_documents(self, documents: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> None:
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

    def retrieve(self, query: str, top_k: int = 10, chunk_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.bm25 or not self.documents:
                return []
            scores = self.bm25.get_scores(self._tokenize(query))
            # 先按 chunk_type 过滤，再按分数排序取 top_k（避免非 small 块占满 top_k 漏掉 small）
            candidates = []
            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                if chunk_type_filter:
                    meta = self.metadata_list[idx] or {}
                    if meta.get("chunk_type") != chunk_type_filter:
                        continue
                candidates.append((idx, score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [
                {"content": self.documents[idx], "metadata": self.metadata_list[idx], "score": float(score), "sparse_score": float(score)}
                for idx, score in candidates[:top_k]
            ]

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

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
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
                        print("Quantizing to int8 (AVX2 dynamic)...")
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
                pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()
        for doc, score in zip(documents, scores, strict=False):
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
        top_n: int = 5,
        rerank_max_candidates: int = 10,
        rerank_max_length: int = 256,
        kg_retriever=None,  # KGRetriever 实例或 None（None 时为旧两路行为）
    ):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k  # RRF 参数
        self.use_reranker = use_reranker
        self.top_n = top_n
        self.rerank_max_candidates = rerank_max_candidates
        self.rerank_max_length = rerank_max_length
        self.kg_retriever = kg_retriever

        if use_reranker:
            self.reranker = Reranker(model_name=reranker_model, max_length=rerank_max_length)
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

        # 1. 密集向量检索（只检索 small 块，按需升级到 parent）
        with tracer.start_as_current_span("rag.retrieve.embedding") as span:
            embedding_start = time.time()
            query_vector = self.vector_store.embed_query(query)
            embedding_time = time.time() - embedding_start
            span.set_attribute("retrieve.vector_dim", len(query_vector))
            span.set_attribute("retrieve.embedding_time_s", round(embedding_time, 3))
            debug_info["steps"].append({"step": "embedding", "desc": "查询向量化", "vector_dim": len(query_vector), "time_s": round(embedding_time, 3)})

        with tracer.start_as_current_span("rag.retrieve.dense") as span:
            dense_start = time.time()
            results = self.vector_store.search_vectors(
                collection_name="chunks",
                query_vector=query_vector,
                top_k=top_k,
                filter_expr="chunk_type == 'small' and status == 'active'"
            )
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
        debug_info["steps"].append({"step": "dense_search", "desc": "密集向量检索（仅 small 块）", "count": len(dense_results), "time_s": round(dense_time, 3)})

        # 2. 稀疏检索（BM25，仅 small 块）
        with tracer.start_as_current_span("rag.retrieve.sparse") as span:
            sparse_start = time.time()
            sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k, chunk_type_filter="small")
            sparse_time = time.time() - sparse_start
            span.set_attribute("retrieve.sparse_count", len(sparse_results))
            span.set_attribute("retrieve.sparse_time_s", round(sparse_time, 3))
        for r in sparse_results:
            r["sparse_score"] = r.pop("score", 0)
            # chunk_type 在 metadata 里，提到顶层方便后续 RRF 合和晋升逻辑读取
            if "chunk_type" not in r:
                meta = r.get("metadata") or {}
                if meta.get("chunk_type"):
                    r["chunk_type"] = meta["chunk_type"]
        debug_info["sparse_results"] = len(sparse_results)
        debug_info["detail"]["sparse_results"] = [{"content": r.get("content", ""), "sparse_score": round(r.get("sparse_score", 0), 4)} for r in sparse_results]
        debug_info["steps"].append({"step": "sparse_search", "desc": "稀疏检索（BM25，仅 small 块）", "count": len(sparse_results), "time_s": round(sparse_time, 3)})

        # 3. KG 检索（第三路，复用已算好的查询向量，失败回退为空结果）
        kg_results = []
        if self.kg_retriever is not None:
            with tracer.start_as_current_span("rag.retrieve.kg") as kg_span:
                kg_start = time.time()
                try:
                    kg_results = self.kg_retriever.retrieve(query=query, query_embedding=query_vector)
                    kg_span.set_attribute("retrieve.kg_count", len(kg_results))
                except Exception as e:
                    kg_span.record_exception(e)
                    kg_span.set_attribute("retrieve.kg_failed", True)
                    kg_results = []
                kg_time = time.time() - kg_start
                kg_span.set_attribute("retrieve.kg_time_s", round(kg_time, 3))
            # RRF 按排名融合，KG 结果先按 kg_score 降序保证排名有意义
            kg_results = sorted(kg_results, key=lambda r: r.get("kg_score", 0), reverse=True)
            debug_info["kg_results"] = len(kg_results)
            debug_info["detail"]["kg_results"] = [{"content": r.get("content", "")[:100], "kg_score": round(r.get("kg_score", 0), 4)} for r in kg_results]
            debug_info["steps"].append({"step": "kg_search", "desc": "知识图谱检索", "count": len(kg_results), "time_s": round(kg_time, 3)})

        # 4. RRF 排名融合（dense + sparse + KG）
        with tracer.start_as_current_span("rag.retrieve.merge") as span:
            merge_start = time.time()
            all_results = self._merge_with_rrf(dense_results, sparse_results, kg_results, top_k=top_k)
            merge_time = time.time() - merge_start
            span.set_attribute("retrieve.merged_count", len(all_results))
            span.set_attribute("retrieve.rrf_k", self.rrf_k)
        debug_info["detail"]["merged_results"] = [
            {"content": r.get("content", ""), "rrf_score": round(r.get("rrf_score", 0), 4)}
            for r in sorted(all_results, key=lambda x: x.get("rrf_score", 0), reverse=True)[:10]
        ]
        debug_info["steps"].append({"step": "merge", "desc": f"RRF 排名融合（k={self.rrf_k}）", "count": len(all_results), "time_s": round(merge_time, 3)})

        # 4. 去重（精确内容）+ 按需升级到 parent（small-only 检索，KG 结果并入）
        with tracer.start_as_current_span("rag.retrieve.dedup") as span:
            dedup_start = time.time()
            # 5a. 精确内容去重
            exact_dedup = []
            seen_contents = set()
            for result in sorted(all_results, key=lambda x: x.get("rrf_score", 0), reverse=True):
                content = result.get("content", "")
                if content not in seen_contents:
                    seen_contents.add(content)
                    exact_dedup.append(result)
            # 4b. 按需升级：2+ smalls 同属一个 medium -> 丢 smalls 加 medium；2+ mediums 同属一个 large -> 丢 mediums 加 large
            #（KG 检索可能返回 medium/large 块：它们不参与晋升，但保留精确匹配结果）
            unique_results = self._promote_smalls_to_parents(exact_dedup)
            dedup_time = time.time() - dedup_start
            promo_stats = getattr(self, "_last_promotion_stats", {"small_to_medium": 0, "medium_to_large": 0})
            span.set_attribute("retrieve.dedup_count", len(unique_results))
            span.set_attribute("retrieve.promoted_small_to_medium", promo_stats["small_to_medium"])
            span.set_attribute("retrieve.promoted_medium_to_large", promo_stats["medium_to_large"])
            span.set_attribute("retrieve.containment_dropped", len(exact_dedup) + promo_stats["small_to_medium"] + promo_stats["medium_to_large"] - len(unique_results))
        # 重新统计 chunks_by_type（按 promotion 后的实际类型分布）
        final_by_type = {"small": 0, "medium": 0, "large": 0}
        for r in unique_results:
            ct = r.get("chunk_type", "small")
            final_by_type[ct] = final_by_type.get(ct, 0) + 1
        debug_info["chunks_by_type"] = final_by_type
        debug_info["detail"]["deduped_results"] = [{"content": r.get("content", "")[:200], "rrf_score": round(r.get("rrf_score", 0), 4), "chunk_type": r.get("chunk_type", "unknown")} for r in unique_results]
        debug_info["steps"].append({"step": "dedup", "desc": f"去重+升级（s->m: {promo_stats['small_to_medium']}, m->l: {promo_stats['medium_to_large']}, 输出 {len(unique_results)} 个 chunk）", "count": len(unique_results), "time_s": round(dedup_time, 3)})

        # 6. 重排序
        rerank_time = 0
        if self.use_reranker and self.reranker and len(unique_results) > 0:
            with tracer.start_as_current_span("rag.retrieve.rerank") as span:
                rerank_start = time.time()
                # ONNX CPU 推理耗时随候选数线性增长，KG 三路融合后候选可达 17+，
                # 只送 rrf_score 前 N 个进 reranker（结果已按 rrf_score 降序，直接切片）
                if self.rerank_max_candidates and len(unique_results) > self.rerank_max_candidates:
                    unique_results = unique_results[:self.rerank_max_candidates]
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

    def _promote_smalls_to_parents(self, smalls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Small-only 检索结果按需升级到 parent：

        - 2+ smalls 内容包含于同一 medium -> 丢 smalls，加 medium
        - 2+ mediums（晋升来的）内容包含于同一 large -> 丢 mediums，加 large
        - 单个 small 不晋升（保留精确匹配）
        - Parent 通过 document_id 查 Milvus，再用内容包含（归一化空白后子串匹配）确认父子关系

        Args:
            smalls: 已精确去重的 small 块结果列表

        Returns:
            升级后的结果列表（按 rrf_score 降序）
        """
        self._last_promotion_stats = {"small_to_medium": 0, "medium_to_large": 0}
        if not smalls:
            return []

        def _normalize(s: str) -> str:
            return re.sub(r"\s+", "", s)

        # 收集所有 document_id，查 parent medium+large
        doc_ids = {r.get("document_id") for r in smalls if r.get("document_id") is not None}
        if not doc_ids:
            return smalls

        doc_ids_str = ", ".join(str(d) for d in doc_ids)
        parent_filter = f"document_id in [{doc_ids_str}] and chunk_type in ['medium', 'large'] and status == 'active'"
        try:
            parents = self.vector_store.query_chunks("chunks", filter_expr=parent_filter)
        except Exception as e:
            print(f"[_promote_smalls_to_parents] query_chunks failed: {e}")
            return smalls

        mediums = [p for p in parents if p.get("chunk_type") == "medium"]
        larges = [p for p in parents if p.get("chunk_type") == "large"]

        # small -> medium 映射（small 只属于一个 medium）
        small_to_medium: Dict[int, Dict] = {}
        for s_idx, s in enumerate(smalls):
            s_norm = _normalize(s.get("content", ""))
            if not s_norm:
                continue
            for m in mediums:
                m_norm = _normalize(m.get("content", ""))
                if s_norm != m_norm and s_norm in m_norm:
                    small_to_medium[s_idx] = m
                    break

        # 按 medium 分组 smalls
        medium_content_to_smalls: Dict[str, List[int]] = defaultdict(list)
        for s_idx, m in small_to_medium.items():
            medium_content_to_smalls[m["content"]].append(s_idx)

        # 决定要丢的 smalls 和要加的 mediums
        smalls_to_drop: set = set()
        promoted_mediums: List[Dict] = []
        for s_indices in medium_content_to_smalls.values():
            if len(s_indices) >= 2:
                smalls_to_drop.update(s_indices)
                m_dict = small_to_medium[s_indices[0]]
                max_score = max(smalls[i].get("rrf_score", 0) for i in s_indices)
                promoted_mediums.append({
                    "content": m_dict["content"],
                    "document_id": m_dict.get("document_id"),
                    "chunk_type": "medium",
                    "id": m_dict.get("id"),
                    "rrf_score": max_score,
                    "dense_score": 0,
                    "sparse_score": 0,
                })
                self._last_promotion_stats["small_to_medium"] += 1

        # 检查 promoted_mediums 是否能进一步晋升到 large
        medium_to_large: Dict[int, Dict] = {}
        for pm_idx, pm in enumerate(promoted_mediums):
            pm_norm = _normalize(pm.get("content", ""))
            if not pm_norm:
                continue
            for large in larges:
                l_norm = _normalize(large.get("content", ""))
                if pm_norm != l_norm and pm_norm in l_norm:
                    medium_to_large[pm_idx] = large
                    break

        # 按 large 分组 mediums
        large_content_to_mediums: Dict[str, List[int]] = defaultdict(list)
        for pm_idx, large in medium_to_large.items():
            large_content_to_mediums[large["content"]].append(pm_idx)

        mediums_to_drop: set = set()
        promoted_larges: List[Dict] = []
        for pm_indices in large_content_to_mediums.values():
            if len(pm_indices) >= 2:
                mediums_to_drop.update(pm_indices)
                l_dict = medium_to_large[pm_indices[0]]
                max_score = max(promoted_mediums[i].get("rrf_score", 0) for i in pm_indices)
                promoted_larges.append({
                    "content": l_dict["content"],
                    "document_id": l_dict.get("document_id"),
                    "chunk_type": "large",
                    "id": l_dict.get("id"),
                    "rrf_score": max_score,
                    "dense_score": 0,
                    "sparse_score": 0,
                })
                self._last_promotion_stats["medium_to_large"] += 1

        # 组装最终结果：保留未丢弃的 smalls + 未丢弃的 promoted mediums + promoted larges
        result = []
        for i, s in enumerate(smalls):
            if i not in smalls_to_drop:
                result.append(s)
        for pm_idx, pm in enumerate(promoted_mediums):
            if pm_idx not in mediums_to_drop:
                result.append(pm)
        result.extend(promoted_larges)

        result.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return result

    def _merge_with_rrf(self, *result_lists, top_k: int = 10) -> List[Dict]:
        """
        使用 RRF（Reciprocal Rank Fusion）融合多路检索器的结果（2-3 路）

        RRF 公式：score(d) = Σ(1 / (k + rank_i(d)))

        按内容去重合并（保持与既有两路行为一致）：稀疏检索结果没有 id 字段
        （metadata 只有 document_id/chunk_type），content 是唯一可靠的跨路合并键。

        Args:
            *result_lists: 各路检索结果列表（各自已按分数降序），如
                (dense_results, sparse_results[, kg_results])
            top_k: 每路取前 top_k 个结果参与融合

        Returns:
            融合后的结果列表（带 rrf_score，降序）
        """
        # 创建内容到文档的映射
        content_map: Dict[str, Dict] = {}

        # 逐路累加 RRF 分数（按各路内排名）
        for results in result_lists:
            for rank, result in enumerate(results[:top_k], start=1):
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
