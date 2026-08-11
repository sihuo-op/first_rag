"""
向量存储

Milvus 向量数据库封装，集成了 embedding 模型，负责：
1. 文本向量化
2. 存储向量和原始文本
3. 相似度搜索
"""

from typing import List, Dict, Any, Optional
import threading
import uuid
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer


class MilvusStore:
    """
    Milvus 向量存储（集成 embedding）

    集成了 SentenceTransformer embedding 模型，可直接处理文本到搜索的完整流程
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_prefix: str = "rag_",
        embedding_model: str = "BAAI/bge-m3",
        dimension: int = 1024,
        alias: str = "default"
    ):
        """
        初始化 Milvus 连接参数和 embedding 模型

        Args:
            host: Milvus 服务地址
            port: Milvus 端口
            collection_prefix: 集合名称前缀，用于区分不同用途的集合
            embedding_model: embedding 模型名称
            dimension: 向量维度，需与 embedding 模型输出维度一致
            alias: 连接别名，支持多连接
        """
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.alias = alias

        self._embedding: Optional[SentenceTransformer] = None
        self._connected = False
        self._collections: Dict[str, Collection] = {}
        self._embedding_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._collections_lock = threading.Lock()

    def _get_embedding(self) -> SentenceTransformer:
        """懒加载 embedding 模型"""
        with self._embedding_lock:
            if self._embedding is None:
                self._embedding = SentenceTransformer(self.embedding_model)
            return self._embedding

    def connect(self) -> None:
        """建立与 Milvus 的连接（懒加载）"""
        with self._connection_lock:
            if not self._connected:
                connections.connect(alias=self.alias, host=self.host, port=self.port)
                self._connected = True

    def disconnect(self) -> None:
        """断开与 Milvus 的连接"""
        with self._connection_lock:
            if self._connected:
                connections.disconnect(alias=self.alias)
                self._connected = False
                with self._collections_lock:
                    self._collections.clear()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        with self._embedding_lock:
            embedding = self._embedding or SentenceTransformer(self.embedding_model)
            self._embedding = embedding
            return embedding.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转为向量

        Args:
            text: 查询文本

        Returns:
            查询向量
        """
        with self._embedding_lock:
            embedding = self._embedding or SentenceTransformer(self.embedding_model)
            self._embedding = embedding
            return embedding.encode([text], convert_to_numpy=True)[0].tolist()

    @property
    def embedding_dimension(self) -> int:
        """获取 embedding 向量维度"""
        return self.dimension

    def create_collection(self, collection_name: str, dimension: int = None) -> None:
        """
        创建向量集合（如果已存在则跳过）

        集合包含以下字段：
        - id: 唯一标识符
        - document_id: 所属文档 ID
        - chunk_type: 切分类型（LARGE/MEDIUM/SMALL）
        - content: 原始文本内容
        - content_hash: 内容哈希，用于去重和冲突检测
        - status: 状态（active/draft/archived/cold）
        - embedding: 向量

        Args:
            collection_name: 集合名称
            dimension: 向量维度，不指定则使用默认值
        """
        self.connect()
        full_name = self._get_full_name(collection_name)
        if utility.has_collection(full_name, using=self.alias):
            return

        dim = dimension or self.dimension
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description=f"{collection_name} collection")
        collection = Collection(name=full_name, schema=schema, using=self.alias)
        collection.create_index(field_name="embedding", index_params={
            "index_type": "IVF_FLAT",  # IVF_FLAT 是最基础的聚类索引，适合小规模数据
            "metric_type": "COSINE",    # 余弦相似度，最适合文本 embedding 检索
            "params": {"nlist": 128}    # 聚类中心数量，影响索引精度和性能
        })
        collection.load()
        with self._collections_lock:
            self._collections[full_name] = collection

    def drop_collection(self, collection_name: str) -> None:
        """删除指定的向量集合"""
        self.connect()
        full_name = self._get_full_name(collection_name)
        if utility.has_collection(full_name, using=self.alias):
            utility.drop_collection(full_name, using=self.alias)
            with self._collections_lock:
                self._collections.pop(full_name, None)

    def add_texts(self, collection_name: str, texts: List[str], metadata_list: List[Dict] = None) -> List[str]:
        """
        批量添加文本（自动向量化）

        这是最常用的方法，只需传入文本列表，自动完成向量化并存入 Milvus

        Args:
            collection_name: 目标集合名
            texts: 原始文本列表
            metadata_list: 元数据列表（如 document_id、chunk_type、content_hash）

        Returns:
            插入后生成的 ID 列表
        """
        vectors = self.embed_texts(texts)
        return self.insert_vectors(collection_name, vectors, texts, metadata_list)

    def insert_vectors(self, collection_name: str, vectors: List[List[float]], documents: List[str], metadata_list: List[Dict] = None) -> List[str]:
        """
        批量插入向量和对应文档（不经过 embedding）

        Args:
            collection_name: 目标集合名
            vectors: 向量列表，每个向量是浮点数列表
            documents: 原始文本列表，与 vectors 一一对应
            metadata_list: 元数据列表（document_id、chunk_type、content_hash）

        Returns:
            插入后生成的 ID 列表
        """
        import hashlib
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)

        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        document_ids = [meta.get("document_id", 0) for meta in metadata_list] if metadata_list else [0] * len(vectors)
        chunk_types = [meta.get("chunk_type", "small") for meta in metadata_list] if metadata_list else ["small"] * len(vectors)
        content_hashes = [
            meta.get("content_hash") or hashlib.sha256(doc.encode("utf-8")).hexdigest()
            for meta, doc in zip(metadata_list or [{}] * len(documents), documents)
        ]
        statuses = ["active"] * len(vectors)

        collection.insert([ids, document_ids, chunk_types, documents, content_hashes, statuses, vectors])
        collection.flush()
        return ids

    def search(self, collection_name: str, query: str, top_k: int = 10, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        检索相关内容（输入文本，自动向量化）

        一站式检索：传入查询文本 → 自动转向量 → Milvus 搜索 → 返回结果

        Args:
            collection_name: 要搜索的集合名
            query: 查询文本
            top_k: 返回最相似的 top_k 条结果
            filter_expr: 可选的过滤表达式（如 "document_id == 1"）

        Returns:
            检索结果列表，每项包含 id、document_id、chunk_type、content、score
        """
        query_vector = self.embed_query(query)
        return self.search_vectors(collection_name, query_vector, top_k, filter_expr)

    def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 10, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        基于向量进行检索（不经过 embedding）

        Args:
            collection_name: 要搜索的集合名
            query_vector: 查询向量
            top_k: 返回最相似的 top_k 条结果
            filter_expr: 可选过滤表达式。不传则默认过滤 status == 'active'，
                         传 None 字符串可禁用过滤（如 admin 查全部）

        Returns:
            检索结果列表
        """
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)

        # 默认只检索 active chunk
        effective_filter = filter_expr if filter_expr is not None else "status == 'active'"

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                },
                limit=top_k,
                expr=effective_filter,
                output_fields=["id", "document_id", "chunk_type", "content", "content_hash", "status"],
                timeout=5
            )
        except MilvusException as e:
            # pymilvus 2.6.x 客户端在 0-hit 搜索时触发 "Unsupported field type: 0" bug
            # （服务端返回 type=0 的空 field_data，HybridHits 解析失败）
            # 0-hit 即无结果，等价于返回 []
            if "Unsupported field type: 0" in (str(e.code) + str(e.message)):
                return []
            raise

        return [
            {
                "id": hit.id,
                "document_id": hit.entity.get("document_id"),
                "chunk_type": hit.entity.get("chunk_type"),
                "content": hit.entity.get("content"),
                "content_hash": hit.entity.get("content_hash"),
                "status": hit.entity.get("status"),
                "score": hit.score
            }
            for hit in results[0]
        ]

    def create_memory_collection(self, collection_name: str, dimension: int = None) -> None:
        """创建用户长期记忆向量集合。"""
        self.connect()
        full_name = self._get_full_name(collection_name)
        if utility.has_collection(full_name, using=self.alias):
            return

        dim = dimension or self.dimension
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="memory_id", dtype=DataType.INT64),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="conversation_id", dtype=DataType.INT64),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="importance", dtype=DataType.FLOAT),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description=f"{collection_name} memory collection")
        collection = Collection(name=full_name, schema=schema, using=self.alias)
        collection.create_index(field_name="embedding", index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        })
        collection.load()
        with self._collections_lock:
            self._collections[full_name] = collection

    def add_memory_texts(self, collection_name: str, texts: List[str], metadata_list: List[Dict]) -> List[str]:
        vectors = self.embed_texts(texts)
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)

        ids = [str(uuid.uuid4()) for _ in texts]
        memory_ids = [int(meta.get("memory_id", 0)) for meta in metadata_list]
        user_ids = [int(meta.get("user_id", 0)) for meta in metadata_list]
        conversation_ids = [int(meta.get("conversation_id", 0)) for meta in metadata_list]
        memory_types = [str(meta.get("memory_type", "")) for meta in metadata_list]
        importances = [float(meta.get("importance", 0.5)) for meta in metadata_list]

        collection.insert([ids, memory_ids, user_ids, conversation_ids, memory_types, importances, texts, vectors])
        collection.flush()
        return ids

    def search_memories(self, collection_name: str, query: str, user_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embed_query(query)
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=f"user_id == {int(user_id)}",
            output_fields=["id", "memory_id", "user_id", "conversation_id", "memory_type", "importance", "content"],
            timeout=5,
        )
        return [
            {
                "id": hit.id,
                "memory_id": hit.entity.get("memory_id"),
                "user_id": hit.entity.get("user_id"),
                "conversation_id": hit.entity.get("conversation_id"),
                "memory_type": hit.entity.get("memory_type"),
                "importance": hit.entity.get("importance"),
                "content": hit.entity.get("content"),
                "score": hit.score,
            }
            for hit in results[0]
        ]

    def delete_memory_vectors(self, collection_name: str, memory_id: int) -> None:
        self.delete_vectors(collection_name, filter_expr=f"memory_id == {int(memory_id)}")

    def query_chunks(self, collection_name: str, filter_expr: str, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        按 filter 表达式查询 chunk（非向量搜索），用于按 document_id / chunk_type 取 parent chunks。

        Args:
            collection_name: 集合名
            filter_expr: Milvus 过滤表达式（如 "document_id in [1,2] and chunk_type in ['medium','large']"）
            output_fields: 返回字段，默认 id/document_id/chunk_type/content/content_hash/status

        Returns:
            chunk dict 列表
        """
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)
        fields = output_fields or ["id", "document_id", "chunk_type", "content", "content_hash", "status"]
        try:
            results = collection.query(
                expr=filter_expr,
                output_fields=fields,
                timeout=10
            )
        except MilvusException as e:
            if "Unsupported field type: 0" in (str(e.code) + str(e.message)):
                return []
            raise
        return [
            {field: row.get(field) for field in fields}
            for row in results
        ]

    def upsert_status(self, collection_name: str, chunk_id: str, status: str) -> None:
        """
        更新指定 chunk 的 status（低频操作，用于冲突作废/归档/回滚）

        Milvus upsert 需要提供完整行，所以先 query 拿到原数据，改 status 后整体 upsert。

        Args:
            collection_name: 集合名
            chunk_id: chunk 的 id（VARCHAR 主键）
            status: 新状态（active / superseded / pending_review / archived）
        """
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)

        # 查出完整行
        results = collection.query(
            expr=f'id == "{chunk_id}"',
            output_fields=["id", "document_id", "chunk_type", "content", "content_hash", "status", "embedding"],
            timeout=5
        )
        if not results:
            raise MilvusException(f"Chunk {chunk_id} not found in {full_name}")

        row = results[0]
        # upsert（delete + insert）
        collection.upsert([{
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_type": row["chunk_type"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "status": status,
            "embedding": row["embedding"]
        }])
        collection.flush()

    def delete_vectors(self, collection_name: str, ids: Optional[List[str]] = None, filter_expr: Optional[str] = None) -> None:
        """
        删除向量（根据 ID 或过滤条件）

        Args:
            collection_name: 集合名
            ids: 要删除的向量 ID 列表
            filter_expr: 删除条件表达式，与 ids 二选一
        """
        self.connect()
        full_name = self._get_full_name(collection_name)
        collection = self._get_collection(full_name)
        if ids:
            collection.delete(f'id in {ids}')
        elif filter_expr:
            collection.delete(filter_expr)

    def has_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        self.connect()
        return utility.has_collection(self._get_full_name(collection_name), using=self.alias)

    def _get_full_name(self, collection_name: str) -> str:
        """生成带前缀的完整集合名称"""
        return f"{self.collection_prefix}{collection_name}"

    def _get_collection(self, full_name: str) -> Collection:
        """
        获取集合实例（带缓存）

        如果集合已在本地缓存中直接返回，否则从 Milvus 加载
        """
        with self._collections_lock:
            if full_name not in self._collections:
                if not utility.has_collection(full_name, using=self.alias):
                    raise MilvusException(f"Collection {full_name} does not exist")
                collection = Collection(name=full_name, using=self.alias)
                collection.load()
                self._collections[full_name] = collection
            return self._collections[full_name]

    def __enter__(self):
        """支持 with 语句自动连接"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句自动断开"""
        self.disconnect()
