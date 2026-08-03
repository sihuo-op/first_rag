"""诊断检索问题"""
import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')

from dotenv import load_dotenv
load_dotenv('/app/.env')

from app.core.config import get_settings
settings = get_settings()

print(f"SENTENCE_TRANSFORMER_MODEL: {settings.SENTENCE_TRANSFORMER_MODEL}")
print(f"MILVUS_HOST: {settings.MILVUS_HOST}")
print(f"MILVUS_PORT: {settings.MILVUS_PORT}")

# 1. 测试 embedding 模型加载
print("\n--- 测试 Embedding 模型 ---")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
    query_emb = model.encode(["劳动合同试用期最长是多久"])
    print(f"Embedding shape: {query_emb.shape}")
    print("Embedding OK!")
except Exception as e:
    print(f"Embedding FAILED: {e}")

# 2. 测试 VectorStore 的 search 方法
print("\n--- 测试 VectorStore.search ---")
try:
    from app.core.dependencies import get_vector_store
    vs = get_vector_store()
    results = vs.search("chunks", "劳动合同试用期最长是多久", top_k=5)
    print(f"Search results: {len(results)}")
    for r in results[:3]:
        content = str(r.get("content", ""))[:100]
        score = r.get("score", 0)
        print(f"  Score={score:.4f}, Content={content}")
except Exception as e:
    print(f"Search FAILED: {e}")

# 3. 测试 HybridRetriever
print("\n--- 测试 HybridRetriever ---")
try:
    from app.core.dependencies import get_vector_store
    from app.rag.retriever import HybridRetriever, SparseRetriever
    vs = get_vector_store()
    sr = SparseRetriever()
    hr = HybridRetriever(
        vector_store=vs,
        sparse_retriever=sr,
        rrf_k=60,
        use_reranker=settings.RERANKER_ENABLED,
        reranker_model=settings.RERANKER_MODEL,
        top_n=settings.RERANKER_TOP_N
    )
    results = hr.retrieve("劳动合同试用期最长是多久", top_k=5)
    print(f"HybridRetriever results: {len(results)}")
    for r in results[:3]:
        content = str(r.get("content", ""))[:100]
        score = r.get("score", 0)
        print(f"  Score={score}, Content={content}")
except Exception as e:
    print(f"HybridRetriever FAILED: {e}")
