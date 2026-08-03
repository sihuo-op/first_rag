import sys
sys.path.insert(0, '/app')

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.rag.retriever import HybridRetriever, SparseRetriever

settings = get_settings()
print(f"MILVUS_HOST: {settings.MILVUS_HOST}")
print(f"MILVUS_PORT: {settings.MILVUS_PORT}")
print(f"RERANKER_ENABLED: {settings.RERANKER_ENABLED}")

vector_store = get_vector_store()
print(f"VectorStore connected: {vector_store._connected}")

# 测试直接搜索
results = vector_store.search('rag_chunks', '劳动合同', top_k=5)
print(f"\nDirect search results: {len(results)}")
for r in results:
    print(f"  score={r['score']:.4f} content={r['content'][:80]}...")

# 测试 HybridRetriever
sparse_retriever = SparseRetriever()
retriever = HybridRetriever(
    vector_store=vector_store,
    sparse_retriever=sparse_retriever,
    rrf_k=60,
    use_reranker=settings.RERANKER_ENABLED,
    reranker_model=settings.RERANKER_MODEL,
    top_n=settings.RERANKER_TOP_N
)
result = retriever.retrieve('劳动合同', top_k=5)
print(f"\nHybridRetriever results: {len(result.documents)}")
for doc in result.documents:
    print(f"  {doc.get('content', '')[:80]}...")
