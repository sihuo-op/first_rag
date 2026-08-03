"""检查 Milvus 数据"""
import sys
sys.path.insert(0, '/app')

from pymilvus import connections, utility

connections.connect(alias='default', host='milvus', port='19530')

# 列出所有 collections
collections = utility.list_collections()
print(f"Collections: {collections}")

for name in collections:
    from pymilvus import Collection
    coll = Collection(name)
    print(f"\nCollection: {name}")
    print(f"  Entities: {coll.num_entities}")
    print(f"  Schema fields: {[f.name for f in coll.schema.fields]}")
