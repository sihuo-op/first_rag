"""检查 Milvus 数据和 embedding 模型"""
import sys
sys.path.insert(0, '/app')

from pymilvus import connections, Collection

# 连接 Milvus
connections.connect(alias='default', host='milvus', port='19530')

# 检查 collection
coll = Collection('labor_law_chunks')
print(f"Collection entities: {coll.num_entities}")
print(f"Schema fields: {[f.name for f in coll.schema.fields]}")

# 查看索引
for idx in coll.indexes:
    print(f"Index: field={idx.field_name}, params={idx.params}")

# 加载并查询
coll.load()
results = coll.query(expr='pk >= 0', limit=3, output_fields=['content', 'source'])
for r in results:
    pk = r.get('pk', 'N/A')
    source = r.get('source', 'N/A')
    content = str(r.get('content', ''))[:100]
    print(f"PK={pk}, Source={source}, Content={content}")

# 测试 embedding
print("\n--- 测试 Embedding ---")
try:
    from app.core.config import get_settings
    settings = get_settings()
    print(f"SENTENCE_TRANSFORMER_MODEL: {settings.SENTENCE_TRANSFORMER_MODEL}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
except Exception as e:
    print(f"Config error: {e}")

import os
from dotenv import load_dotenv
load_dotenv('/app/.env')
print(f"HF_ENDPOINT from .env: {os.environ.get('HF_ENDPOINT', 'not set')}")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
    emb = model.encode(["劳动合同试用期"])
    print(f"Embedding shape: {emb.shape}")
    print("Embedding model loaded OK!")
except Exception as e:
    print(f"Embedding error: {e}")

# 测试向量搜索
print("\n--- 测试向量搜索 ---")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
    query_emb = model.encode(["劳动合同试用期最长是多久"])
    
    from pymilvus import connections
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = coll.search(
        data=[query_emb.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["content", "source"]
    )
    print(f"Search results count: {len(results[0])}")
    for hit in results[0]:
        print(f"  Score={hit.score:.4f}, Source={hit.entity.get('source', 'N/A')}, Content={str(hit.entity.get('content', ''))[:80]}")
except Exception as e:
    print(f"Search error: {e}")
