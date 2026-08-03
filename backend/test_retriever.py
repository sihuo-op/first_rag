import sys
sys.path.insert(0, '/app')
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pymilvus import connections
connections.connect(alias='default', host='milvus', port='19530')
from pymilvus import Collection
col = Collection('rag_chunks')
col.load()
print('Milvus entities:', col.num_entities)

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.rag.retriever import HybridRetriever, SparseRetriever

settings = get_settings()
vector_store = get_vector_store()
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
print(f'检索到 {len(result.documents)} 个文档')
for doc in result.documents[:3]:
    print(f'  {doc.get("content", "")[:80]}...')
