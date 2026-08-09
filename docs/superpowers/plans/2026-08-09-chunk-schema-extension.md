# Chunk Schema 扩容实施计划：更新与冷知识识别

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 RAG 系统的 chunk 数据增加"语义冲突检测作废"和"冷知识识别清理"两项能力，扩容 Milvus 和 PostgreSQL schema。

**Architecture:** 方案 B 分层设计--Milvus 只放检索期过滤字段（`content_hash` + `status`），PostgreSQL 放高频统计与审核元数据。冲突检测走 LLM 判定 + 自动/人工双通道；冷知识走 APScheduler 周期扫描 + 两段式清理（归档 90 天后硬删）。

**Tech Stack:** FastAPI + SQLAlchemy + Milvus (pymilvus) + APScheduler + LangChain LLM + pytest

## Global Constraints

- Milvus collection 前缀：`rag_`（chunks collection 全名 `rag_chunks`）
- Embedding 模型：`BAAI/bge-m3`，向量维度 1024
- 配置文件：`backend/app/core/config.py`（Pydantic Settings），`.env.example` 同步更新
- 依赖注入：`backend/app/core/dependencies.py` 单例 + 双检锁
- LLM 调用：`from app.llm.providers import get_rewrite_llm, invoke_llm_threadsafe`（项目统一封装）
- 测试：pytest，`backend/tests/unit/` 下，`sys.path.insert` 导入 `app.*`
- 提交规范：`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:` 前缀
- ORM 模型：`backend/app/entities/database.py`，Pydantic schema 在 `backend/app/entities/schemas.py`
- 注意：`document_service.py` 现有代码与 ORM 有字段名漂移（`filename`/`chunk_index`/`DocumentStatus.ACTIVE`），实施时以 `database.py` 为准，不要"修复"已有漂移（超出本计划范围）

---

## 文件结构总览

### 新增文件

| 文件 | 职责 |
|---|---|
| `backend/scripts/migrate_pg.sql` | PG ALTER TABLE 迁移脚本 |
| `backend/scripts/migrate_milvus.py` | Milvus drop + 重建 + 重处理脚本 |
| `backend/app/services/conflict_service.py` | 冲突检测管道（LLM 判冲突 + 自动/人工分流） |
| `backend/app/services/cold_knowledge_service.py` | 冷知识扫描 + 硬删除 |
| `backend/app/core/scheduler.py` | APScheduler 初始化与任务注册 |
| `backend/tests/unit/test_config_chunk_lifecycle.py` | 配置项字段测试 |
| `backend/tests/unit/test_conflict_service.py` | 冲突检测逻辑测试（mock LLM） |
| `backend/tests/unit/test_cold_knowledge_service.py` | 冷知识扫描规则测试 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `backend/app/core/config.py` | 加冲突检测阈值 + 冷知识规则 + cron 配置 |
| `.env.example` | 同步新配置项 |
| `backend/app/entities/database.py` | `DocumentChunk` / `Document` 加新字段 |
| `backend/app/entities/schemas.py` | 加 `ChunkResponse` / `ChunkPatchRequest` 等 |
| `backend/app/rag/vector_store.py` | `create_collection` 加字段；新增 `upsert_status`；`search_vectors` 加默认过滤；`insert_vectors`/`add_texts` 写 content_hash + status |
| `backend/app/rag/retriever.py` | `search` 接收 `background_tasks`，注册统计更新 |
| `backend/app/services/document_service.py` | `process_document` 写 content_hash + status；上传后触发冲突检测 |
| `backend/app/services/chat_service.py` | 透传 `background_tasks` |
| `backend/app/api/chat.py` | 注入 `BackgroundTasks` |
| `backend/app/api/admin.py` | 扩展 `/admin/documents` 返回字段；新增 `/admin/chunks` 列表与 PATCH |
| `backend/app/core/dependencies.py` | 注册 `conflict_service` / `cold_knowledge_service` / `scheduler` |
| `backend/main.py` | startup 事件启动 scheduler |

---

## Phase 1: Config & Schema Foundation

### Task 1: 添加配置项

**Files:**
- Modify: `backend/app/core/config.py:111-125`（在 `ALLOWED_EXTENSIONS` 后、`UPLOAD_DIR` 前插入）
- Modify: `.env.example`
- Test: `backend/tests/unit/test_config_chunk_lifecycle.py`

**Interfaces:**
- Produces: `settings.CONFLICT_DETECTION_HIGH_CONFIDENCE` / `CONFLICT_DETECTION_LOW_CONFIDENCE` / `COLD_KNOWLEDGE_TIMEOUT_DAYS` / `COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD` / `COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS` / `COLD_KNOWLEDGE_LOW_QUALITY_SCORE` / `COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS` / `COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS` / `COLD_KNOWLEDGE_SWEEP_CRON` / `HARD_DELETE_SWEEP_CRON`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_config_chunk_lifecycle.py`：

```python
"""验证 chunk 生命周期相关配置字段存在且有正确默认值。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import get_settings


def test_conflict_detection_settings_exist_with_defaults():
    settings = get_settings()
    assert hasattr(settings, "CONFLICT_DETECTION_HIGH_CONFIDENCE")
    assert hasattr(settings, "CONFLICT_DETECTION_LOW_CONFIDENCE")
    assert settings.CONFLICT_DETECTION_HIGH_CONFIDENCE == 0.85
    assert settings.CONFLICT_DETECTION_LOW_CONFIDENCE == 0.5


def test_cold_knowledge_settings_exist_with_defaults():
    settings = get_settings()
    assert settings.COLD_KNOWLEDGE_TIMEOUT_DAYS == 90
    assert settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD == 2
    assert settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS == 30
    assert settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE == 0.3
    assert settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS == 5
    assert settings.COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS == 90
    assert settings.COLD_KNOWLEDGE_SWEEP_CRON == "0 3 * * *"
    assert settings.HARD_DELETE_SWEEP_CRON == "0 4 * * *"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_config_chunk_lifecycle.py -v`
Expected: FAIL with `AttributeError` 或 assert 错误

- [ ] **Step 3: 在 config.py 添加配置项**

在 `backend/app/core/config.py` 的 `ALLOWED_EXTENSIONS` 后插入：

```python
    # Chunk 生命周期配置
    # 冲突检测：高置信度自动作废，低置信度转人工审核
    CONFLICT_DETECTION_HIGH_CONFIDENCE: float = 0.85
    CONFLICT_DETECTION_LOW_CONFIDENCE: float = 0.5

    # 冷知识识别规则
    COLD_KNOWLEDGE_TIMEOUT_DAYS: int = 90              # last_accessed_at 超过此天数归档
    COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD: int = 2         # access_count 低于此值
    COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS: int = 30         # 上传超过此天数才适用频次规则
    COLD_KNOWLEDGE_LOW_QUALITY_SCORE: float = 0.3      # avg_score 低于此值
    COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS: int = 5       # hit_count 达到此值才适用质量规则
    COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS: int = 90    # 归档后保留天数，过期硬删

    # 调度 cron
    COLD_KNOWLEDGE_SWEEP_CRON: str = "0 3 * * *"       # 冷知识扫描（每天 3 点）
    HARD_DELETE_SWEEP_CRON: str = "0 4 * * *"          # 硬删除扫描（每天 4 点）
```

- [ ] **Step 4: 同步 .env.example**

在 `.env.example` 末尾追加：

```env
# Chunk 生命周期配置（可选，不设置则使用默认值）
# CONFLICT_DETECTION_HIGH_CONFIDENCE=0.85
# CONFLICT_DETECTION_LOW_CONFIDENCE=0.5
# COLD_KNOWLEDGE_TIMEOUT_DAYS=90
# COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD=2
# COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS=30
# COLD_KNOWLEDGE_LOW_QUALITY_SCORE=0.3
# COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS=5
# COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS=90
# COLD_KNOWLEDGE_SWEEP_CRON=0 3 * * *
# HARD_DELETE_SWEEP_CRON=0 4 * * *
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_config_chunk_lifecycle.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/app/core/config.py .env.example backend/tests/unit/test_config_chunk_lifecycle.py
git commit -m "feat(config): add chunk lifecycle config for conflict detection and cold knowledge"
```

---

### Task 2: PG 迁移 SQL 脚本

**Files:**
- Create: `backend/scripts/migrate_pg.sql`

**Interfaces:**
- Produces: `backend/scripts/migrate_pg.sql`（手动执行：`sqlite3 data/sqlite/app.db < scripts/migrate_pg.sql` 或 PG 客户端执行）

> 注意：项目 `DATABASE_URL` 默认是 SQLite (`sqlite:///./data/sqlite/app.db`)，但 SQL 兼容 PG。SQLite 支持 `ALTER TABLE ADD COLUMN` 和 `CREATE INDEX`，但 `REFERENCES` 子句在 SQLite 中只是语法糖（默认不强制）。脚本两种库都能跑。

- [ ] **Step 1: 创建迁移脚本**

创建 `backend/scripts/migrate_pg.sql`：

```sql
-- Chunk 生命周期扩容迁移脚本
-- 执行方式：sqlite3 data/sqlite/app.db < scripts/migrate_pg.sql
-- 或在 PG 客户端中执行

-- ============ document_chunks 表 ============
ALTER TABLE document_chunks ADD COLUMN content_hash VARCHAR(64);
ALTER TABLE document_chunks ADD COLUMN status VARCHAR(20) DEFAULT 'active' NOT NULL;
ALTER TABLE document_chunks ADD COLUMN conflict_with_chunk_id VARCHAR(100);
ALTER TABLE document_chunks ADD COLUMN conflict_detected_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN confidence FLOAT;
ALTER TABLE document_chunks ADD COLUMN review_reason TEXT;
ALTER TABLE document_chunks ADD COLUMN superseded_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN reviewed_by INTEGER REFERENCES users(id);
ALTER TABLE document_chunks ADD COLUMN reviewed_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN access_count INTEGER DEFAULT 0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN last_accessed_at TIMESTAMP;
ALTER TABLE document_chunks ADD COLUMN hit_count INTEGER DEFAULT 0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN total_score FLOAT DEFAULT 0.0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN avg_score FLOAT DEFAULT 0.0 NOT NULL;
ALTER TABLE document_chunks ADD COLUMN archived_reason VARCHAR(30);
ALTER TABLE document_chunks ADD COLUMN archived_at TIMESTAMP;

CREATE INDEX idx_document_chunks_milvus_id ON document_chunks(milvus_id);
CREATE INDEX idx_document_chunks_status ON document_chunks(status);

-- ============ documents 表 ============
ALTER TABLE documents ADD COLUMN conflict_check_status VARCHAR(20) DEFAULT 'completed' NOT NULL;
ALTER TABLE documents ADD COLUMN conflict_check_started_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_completed_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_progress VARCHAR(20);
```

- [ ] **Step 2: 验证脚本能跑通（在测试库上）**

```bash
cd backend
cp data/sqlite/app.db data/sqlite/app.db.bak
sqlite3 data/sqlite/app.db < scripts/migrate_pg.sql
sqlite3 data/sqlite/app.db ".schema document_chunks" | head -30
sqlite3 data/sqlite/app.db ".schema documents" | grep conflict_check
# 验证字段都在后恢复
cp data/sqlite/app.db.bak data/sqlite/app.db
```

Expected: 新字段出现在 schema 输出中

- [ ] **Step 3: 提交**

```bash
git add backend/scripts/migrate_pg.sql
git commit -m "feat(migration): add PG schema migration script for chunk lifecycle"
```

---

### Task 3: 更新 ORM 模型

**Files:**
- Modify: `backend/app/entities/database.py:126-156`（`DocumentChunk` 类）
- Modify: `backend/app/entities/database.py:98-123`（`Document` 类）

**Interfaces:**
- Produces: `DocumentChunk` 模型新增字段（`content_hash` / `status` / `conflict_with_chunk_id` 等）；`Document` 模型新增 `conflict_check_status` 等

- [ ] **Step 1: 更新 DocumentChunk 模型**

在 `backend/app/entities/database.py` 的 `DocumentChunk` 类中，在 `milvus_id` 字段后追加新字段（保留现有字段不动）：

```python
class DocumentChunk(Base):
    """文档片段模型"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_type = Column(SQLEnum(ChunkType), nullable=False)
    parent_chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    content = Column(Text, nullable=False)
    metadata_ = Column(JSON, name="metadata")
    position = Column(Integer)
    token_count = Column(Integer)
    milvus_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ============ 新增：chunk 生命周期字段 ============
    # 增量更新 & 状态镜像
    content_hash = Column(String(64), nullable=True, index=True)
    status = Column(String(20), default="active", nullable=False, index=True)

    # 冲突作废元数据
    conflict_with_chunk_id = Column(String(100), nullable=True)
    conflict_detected_at = Column(DateTime(timezone=True), nullable=True)
    confidence = Column(Float, nullable=True)
    review_reason = Column(Text, nullable=True)
    superseded_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    # 命中统计（高频写）
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    hit_count = Column(Integer, default=0, nullable=False)
    total_score = Column(Float, default=0.0, nullable=False)
    avg_score = Column(Float, default=0.0, nullable=False)

    # 冷知识归档
    archived_reason = Column(String(30), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="chunks")
    parent_chunk = relationship("DocumentChunk", remote_side=[id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
```

- [ ] **Step 2: 更新 Document 模型**

在 `Document` 类的 `updated_at` 后追加：

```python
    # ============ 新增：冲突检测状态 ============
    conflict_check_status = Column(String(20), default="completed", nullable=False)
    conflict_check_started_at = Column(DateTime(timezone=True), nullable=True)
    conflict_check_completed_at = Column(DateTime(timezone=True), nullable=True)
    conflict_check_progress = Column(String(20), nullable=True)
```

- [ ] **Step 3: 验证模型能加载**

```bash
cd backend
python -c "from app.entities.database import DocumentChunk, Document; print([c.name for c in DocumentChunk.__table__.columns]); print([c.name for c in Document.__table__.columns])"
```

Expected: 输出包含 `content_hash` / `status` / `conflict_check_status` 等新字段

- [ ] **Step 4: 提交**

```bash
git add backend/app/entities/database.py
git commit -m "feat(orm): extend DocumentChunk and Document models with lifecycle fields"
```

---

### Task 4: 更新 Pydantic 响应 Schema

**Files:**
- Modify: `backend/app/entities/schemas.py`（在 `ChunkResponse` 附近扩展，或新增）

**Interfaces:**
- Produces: `ChunkDetailResponse`（带冲突/统计字段）、`ChunkPatchRequest`（admin PATCH body）、`DocumentWithConflictStatusResponse`（admin 文档列表返回）

- [ ] **Step 1: 在 schemas.py 末尾追加响应模型**

```python
class ChunkDetailResponse(BaseModel):
    """chunk 详情响应（含冲突与统计字段）"""
    id: int
    document_id: int
    chunk_type: str
    content: str
    milvus_id: Optional[str] = None
    content_hash: Optional[str] = None
    status: str = "active"
    # 冲突
    conflict_with_chunk_id: Optional[str] = None
    conflict_with_content: Optional[str] = None  # join 出来的新 chunk 内容
    conflict_detected_at: Optional[datetime] = None
    confidence: Optional[float] = None
    review_reason: Optional[str] = None
    superseded_at: Optional[datetime] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    # 统计
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    hit_count: int = 0
    avg_score: float = 0.0
    # 归档
    archived_reason: Optional[str] = None
    archived_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChunkPatchRequest(BaseModel):
    """admin PATCH chunk 动作"""
    action: str  # confirm | dismiss | archive | restore | hard_delete


class DocumentWithConflictStatusResponse(DocumentResponse):
    """文档响应（带冲突检测状态）"""
    conflict_check_status: str = "completed"
    conflict_check_progress: Optional[str] = None
    conflict_check_completed_at: Optional[datetime] = None
```

- [ ] **Step 2: 验证 schema 能加载**

```bash
cd backend
python -c "from app.entities.schemas import ChunkDetailResponse, ChunkPatchRequest, DocumentWithConflictStatusResponse; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add backend/app/entities/schemas.py
git commit -m "feat(schema): add Pydantic response models for chunk lifecycle and admin API"
```

---

## Phase 2: Milvus Schema & Vector Store

### Task 5: 更新 MilvusStore.create_collection 加新字段

**Files:**
- Modify: `backend/app/rag/vector_store.py:116-153`（`create_collection` 方法）

**Interfaces:**
- Produces: `MilvusStore.create_collection` 创建的 collection 包含 `content_hash` 和 `status` 字段

- [ ] **Step 1: 修改 create_collection 方法的 fields 列表**

在 `backend/app/rag/vector_store.py` 的 `create_collection` 方法中，把 `fields` 列表替换为：

```python
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
```

- [ ] **Step 2: 验证创建逻辑（手动）**

```bash
cd backend
python -c "
from app.rag.vector_store import MilvusStore
vs = MilvusStore()
vs.drop_collection('chunks_test')
vs.create_collection('chunks_test')
from pymilvus import Collection, connections
vs.connect()
c = Collection('rag_chunks_test')
print([f.name for f in c.schema.fields])
vs.drop_collection('chunks_test')
"
```

Expected: 输出包含 `content_hash` 和 `status`

- [ ] **Step 3: 提交**

```bash
git add backend/app/rag/vector_store.py
git commit -m "feat(vector_store): add content_hash and status fields to chunks collection schema"
```

---

### Task 6: 更新 insert_vectors / add_texts 写入新字段

**Files:**
- Modify: `backend/app/rag/vector_store.py:164-204`（`add_texts` 和 `insert_vectors` 方法）

**Interfaces:**
- Consumes: metadata_list 里支持 `content_hash` 字段
- Produces: `insert_vectors` 写入 `content_hash` 和 `status='active'`

- [ ] **Step 1: 修改 insert_vectors 方法**

在 `backend/app/rag/vector_store.py` 的 `insert_vectors` 方法中，替换为：

```python
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
```

- [ ] **Step 2: 修改 add_texts 透传 content_hash**

`add_texts` 方法已经调用 `insert_vectors`，无需改动（metadata_list 会透传）。确认 `add_texts` 的 docstring 提到 `content_hash`：

```python
    def add_texts(self, collection_name: str, texts: List[str], metadata_list: List[Dict] = None) -> List[str]:
        """
        批量添加文本（自动向量化）

        Args:
            collection_name: 目标集合名
            texts: 原始文本列表
            metadata_list: 元数据列表（如 document_id、chunk_type、content_hash）

        Returns:
            插入后生成的 ID 列表
        """
        vectors = self.embed_texts(texts)
        return self.insert_vectors(collection_name, vectors, texts, metadata_list)
```

- [ ] **Step 3: 验证插入（手动）**

```bash
cd backend
python -c "
from app.rag.vector_store import MilvusStore
vs = MilvusStore()
vs.drop_collection('chunks_test')
vs.create_collection('chunks_test')
ids = vs.add_texts('chunks_test', ['测试内容1', '测试内容2'], [
    {'document_id': 1, 'chunk_type': 'small'},
    {'document_id': 1, 'chunk_type': 'small'}
])
print('inserted ids:', ids)
results = vs.search('chunks_test', '测试', top_k=2)
for r in results:
    print(r)
vs.drop_collection('chunks_test')
"
```

Expected: 搜索结果包含 `content_hash` 和 `status`（注：search 默认输出字段需更新，见 Task 7）

- [ ] **Step 4: 提交**

```bash
git add backend/app/rag/vector_store.py
git commit -m "feat(vector_store): write content_hash and status on insert"
```

---

### Task 7: 更新 search_vectors 加默认 status 过滤

**Files:**
- Modify: `backend/app/rag/vector_store.py:224-263`（`search_vectors` 方法）

**Interfaces:**
- Produces: `search_vectors` 默认 `filter_expr="status == 'active'"`，调用方可覆盖

- [ ] **Step 1: 修改 search_vectors 方法**

替换 `search_vectors` 方法为：

```python
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
```

- [ ] **Step 2: 验证默认过滤生效（手动）**

```bash
cd backend
python -c "
from app.rag.vector_store import MilvusStore
vs = MilvusStore()
vs.drop_collection('chunks_test')
vs.create_collection('chunks_test')
vs.add_texts('chunks_test', ['active chunk'], [{'document_id': 1, 'chunk_type': 'small'}])
# 手动 upsert 一条 status=archived 的（等 Task 8 实现 upsert_status 后再做完整测试）
results = vs.search('chunks_test', 'active', top_k=5)
print('default filter results:', len(results))
results_all = vs.search_vectors('chunks_test', vs.embed_query('active'), top_k=5, filter_expr='')
print('no filter results:', len(results_all))
vs.drop_collection('chunks_test')
"
```

Expected: 默认过滤返回 1 条；无过滤也返回 1 条（此时还没 archived 数据）

- [ ] **Step 3: 更新现有调用方传入 status 过滤**

`backend/app/rag/retriever.py:184` 现有调用传了 `filter_expr='chunk_type == "small"'`，会**覆盖**默认的 status 过滤。需要改成组合过滤：

把：
```python
filter_expr = 'chunk_type == "small"'
```
改为：
```python
filter_expr = 'chunk_type == "small" && status == "active"'
```

- [ ] **Step 4: 提交**

```bash
git add backend/app/rag/vector_store.py backend/app/rag/retriever.py
git commit -m "feat(vector_store): default status==active filter; update retriever to combine filters"
```

---

### Task 8: 新增 upsert_status 方法

**Files:**
- Modify: `backend/app/rag/vector_store.py`（在 `delete_vectors` 方法前新增）

**Interfaces:**
- Produces: `MilvusStore.upsert_status(collection_name: str, chunk_id: str, status: str) -> None`

> 注意：Milvus upsert 需要提供完整行（包括向量）。本方法先 query 拿到完整行，再 upsert 修改 status。低频操作可接受。

- [ ] **Step 1: 添加 upsert_status 方法**

在 `backend/app/rag/vector_store.py` 的 `delete_vectors` 方法前插入：

```python
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
```

- [ ] **Step 2: 验证 upsert（手动）**

```bash
cd backend
python -c "
from app.rag.vector_store import MilvusStore
vs = MilvusStore()
vs.drop_collection('chunks_test')
vs.create_collection('chunks_test')
ids = vs.add_texts('chunks_test', ['测试 chunk'], [{'document_id': 1, 'chunk_type': 'small'}])
chunk_id = ids[0]
vs.upsert_status('chunks_test', chunk_id, 'archived')
# 默认过滤应该查不到
r1 = vs.search('chunks_test', '测试', top_k=5)
# 关闭过滤应该能查到，且 status=archived
r2 = vs.search_vectors('chunks_test', vs.embed_query('测试'), top_k=5, filter_expr='')
print('after archive, default filter:', len(r1))
print('after archive, no filter status:', r2[0]['status'] if r2 else None)
vs.drop_collection('chunks_test')
"
```

Expected: 默认过滤返回 0 条；无过滤返回 1 条且 status='archived'

- [ ] **Step 3: 提交**

```bash
git add backend/app/rag/vector_store.py
git commit -m "feat(vector_store): add upsert_status method for chunk lifecycle updates"
```

---

### Task 9: Milvus 迁移脚本（drop + 重建 + 重处理）

**Files:**
- Create: `backend/scripts/migrate_milvus.py`

**Interfaces:**
- Produces: `python scripts/migrate_milvus.py` 执行 drop + recreate + 重处理所有 COMPLETED 文档

- [ ] **Step 1: 创建迁移脚本**

创建 `backend/scripts/migrate_milvus.py`：

```python
"""
Milvus 迁移脚本：drop 旧 chunks collection，用新 schema 重建，重处理所有 COMPLETED 文档。

执行方式：
    cd backend
    python scripts/migrate_milvus.py

注意：会重新 embedding，耗时取决于文档数量。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.db.session import SessionLocal
from app.entities.database import Document, DocumentChunk, DocumentStatus
from app.entities.schemas import DocumentCreate
from app.rag.parsers import get_parser
from app.rag.splitter import ThreeLayerSplitter
from app.services.document_service import DocumentService
import hashlib


def main():
    settings = get_settings()
    vector_store = get_vector_store()
    db = SessionLocal()
    splitter = ThreeLayerSplitter()

    try:
        print("=" * 50)
        print("Milvus 迁移：drop + 重建 + 重处理")
        print("=" * 50)

        # 1. drop 旧 collection
        print("[1/3] Dropping chunks collection...")
        if vector_store.has_collection("chunks"):
            vector_store.drop_collection("chunks")
        print("Dropped")

        # 2. 用新 schema 重建
        print("[2/3] Creating chunks collection with new schema...")
        vector_store.create_collection("chunks")
        print("Created")

        # 3. 重处理所有 COMPLETED 文档
        print("[3/3] Reprocessing documents...")
        docs = db.query(Document).filter(Document.status == DocumentStatus.COMPLETED).all()
        print(f"Found {len(docs)} documents to reprocess")

        for i, doc in enumerate(docs, 1):
            print(f"  [{i}/{len(docs)}] Doc {doc.id}: {doc.title}")
            try:
                # 解析 + 切分
                parser = get_parser(doc.file_type)
                content = parser.parse(doc.file_path)
                chunks = splitter.split_text(content)

                # 删除旧 chunk 记录
                db.query(DocumentChunk).filter_by(document_id=doc.id).delete()
                db.commit()

                # 写入 Milvus（带 content_hash）
                texts = []
                metadata_list = []
                for chunk in chunks:
                    for chunk_type in ['large', 'medium', 'small']:
                        texts.append(chunk[chunk_type])
                        metadata_list.append({
                            "document_id": doc.id,
                            "chunk_type": chunk_type,
                            "content_hash": hashlib.sha256(chunk[chunk_type].encode("utf-8")).hexdigest()
                        })
                milvus_ids = vector_store.add_texts("chunks", texts, metadata_list)

                # 写入 PG
                idx = 0
                for chunk_idx, chunk in enumerate(chunks):
                    for chunk_type in ['large', 'medium', 'small']:
                        db_chunk = DocumentChunk(
                            document_id=doc.id,
                            content=chunk[chunk_type],
                            chunk_type=getattr(__import__('app.entities.database', fromlist=['ChunkType']).ChunkType, chunk_type.upper()),
                            position=chunk_idx,
                            milvus_id=milvus_ids[idx],
                            content_hash=metadata_list[idx]["content_hash"],
                            status="active"
                        )
                        db.add(db_chunk)
                        idx += 1
                db.commit()
                print(f"    -> {len(texts)} chunks reinserted")
            except Exception as e:
                print(f"    ERROR: {e}")
                db.rollback()

        print("=" * 50)
        print("Migration complete!")
        print("=" * 50)
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本能 import**

```bash
cd backend
python -c "import sys; sys.path.insert(0, 'scripts'); from migrate_milvus import main; print('OK')"
```

Expected: 输出 `OK`（不实际执行，只验证语法和 import）

- [ ] **Step 3: 提交**

```bash
git add backend/scripts/migrate_milvus.py
git commit -m "feat(migration): add Milvus drop+recreate+reprocess migration script"
```

---

## Phase 3: Document Service & Retrieval

### Task 10: 更新 document_service.process_document 写 content_hash + status

**Files:**
- Modify: `backend/app/services/document_service.py:125-178`（`process_document` 方法）

**Interfaces:**
- Produces: `process_document` 写入 `content_hash` 和 `status='active'` 到 PG + Milvus

- [ ] **Step 1: 修改 process_document 方法**

在 `backend/app/services/document_service.py` 的 `process_document` 方法中，替换 chunks 入库逻辑（参考实际代码，下面是关键改动点）：

在文件顶部加 import：
```python
import hashlib
```

在 `process_document` 方法中，把构造 `texts` / `metadata_list` 的循环改为：

```python
            texts = []
            metadata_list = []
            for i, chunk in enumerate(chunks):
                for chunk_type in ['large', 'medium', 'small']:
                    text = chunk[chunk_type]
                    texts.append(text)
                    metadata_list.append({
                        "document_id": doc_id,
                        "chunk_type": chunk_type,
                        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
                    })

            milvus_ids = self.vector_store.add_texts("chunks", texts, metadata_list)
```

把构造 `DocumentChunk` 的循环改为（用 milvus_ids[idx] 关联）：

```python
            idx = 0
            for i, chunk in enumerate(chunks):
                for chunk_type in ['large', 'medium', 'small']:
                    db_chunk = DocumentChunk(
                        document_id=doc_id,
                        content=chunk[chunk_type],
                        chunk_type=getattr(ChunkType, chunk_type.upper()),
                        position=i,
                        milvus_id=milvus_ids[idx],
                        content_hash=metadata_list[idx]["content_hash"],
                        status="active"
                    )
                    self.db.add(db_chunk)
                    idx += 1
```

> 注意：现有代码用 `chunk_index=i` 和 `filename` 等字段与 ORM 不一致。本步骤只新增 `content_hash` / `status` / `position`，**不要**顺手修复其他漂移字段（超出范围）。如果现有代码能跑，保持原样；如果跑不通，由实施者按 ORM 实际字段名调整。

- [ ] **Step 2: 手动验证（上传一个测试文档）**

```bash
cd backend
# 启动服务后用 curl 上传一个文档，或直接调 service
python -c "
from app.db.session import SessionLocal
from app.entities.database import Document, DocumentStatus
from app.services.document_service import DocumentService
from app.core.dependencies import get_vector_store
db = SessionLocal()
doc = db.query(Document).first()
if doc:
    svc = DocumentService(db, get_vector_store())
    svc.process_document(doc.id)
    db.commit()
    # 检查 chunk 是否有 content_hash
    from app.entities.database import DocumentChunk
    chunks = db.query(DocumentChunk).filter_by(document_id=doc.id).all()
    for c in chunks[:3]:
        print(f'chunk {c.id}: content_hash={c.content_hash[:16] if c.content_hash else None}, status={c.status}')
else:
    print('No document found')
db.close()
"
```

Expected: chunk 的 content_hash 有值，status='active'

- [ ] **Step 3: 提交**

```bash
git add backend/app/services/document_service.py
git commit -m "feat(document_service): write content_hash and status on process_document"
```

---

### Task 11: 检索路径注入 BackgroundTasks 写统计

**Files:**
- Modify: `backend/app/rag/retriever.py`（`HybridRetriever.search` 方法）
- Modify: `backend/app/services/chat_service.py`（透传 background_tasks）
- Modify: `backend/app/api/chat.py`（注入 BackgroundTasks）

**Interfaces:**
- Consumes: `fastapi.BackgroundTasks`（从 API 层透传）
- Produces: `HybridRetriever.search(query, top_k, background_tasks=None)`；统计更新函数 `update_chunk_stats(db, hits)`

- [ ] **Step 1: 在 retriever.py 添加统计更新工具函数**

在 `backend/app/rag/retriever.py` 顶部加 import：
```python
from datetime import datetime
```

在 `HybridRetriever` 类外（文件末尾）添加工具函数：

```python
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
```

- [ ] **Step 2: 修改 HybridRetriever.retrieve 接收 background_tasks**

`backend/app/rag/retriever.py:161` 的 `HybridRetriever.retrieve` 方法当前签名：

```python
def retrieve(self, query: str, top_k: int = 10) -> tuple:
```

返回 `tuple`：`(unique_results, debug_info)`，其中 `unique_results` 是最终结果列表（每项含 `id`、`content`、`rrf_score` 等）。

修改为接收 `background_tasks=None`，在 `return` 前注册统计更新：

```python
    def retrieve(self, query: str, top_k: int = 10, background_tasks=None) -> tuple:
        # ... 现有检索逻辑不动，最后得到 unique_results, debug_info ...
        
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
            update_chunk_stats(db, hits)
        finally:
            db.close()
```

> `unique_results` 里每项的 `id` 字段就是 Milvus chunk id，`score` 字段名取决于是否经过 reranker：rerank 后是 `rerank_score`，否则是 `rrf_score`。`update_chunk_stats` 函数里用 `hit.get("score", 0.0)` 兜底，但实际可能没有 `score` 键。建议在 wrapper 里先把分数归一化：

```python
@staticmethod
def _update_stats_wrapper(hits):
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
```

- [ ] **Step 3: 修改 chat_service 透传 background_tasks**

在 `backend/app/services/chat_service.py` 找到调用 `retriever.retrieve(...)` 的地方（如 `chat` 方法），加 `background_tasks` 参数透传。先读文件确认方法签名，然后在调用 `self.retriever.retrieve(...)` 时传入 `background_tasks=background_tasks`，并把方法签名加 `background_tasks=None`。

- [ ] **Step 4: 修改 chat API 注入 BackgroundTasks**

在 `backend/app/api/chat.py` 的 chat 接口（`/chat` 或 `/conversations/{id}/messages`，按实际）加 `background_tasks: BackgroundTasks` 参数，传给 `chat_service`：

```python
from fastapi import BackgroundTasks

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    return chat_service.chat(request, background_tasks=background_tasks, user_id=current_user.id)
```

> 实施者需要读 `chat.py` 和 `chat_service.py` 找到实际的 chat 方法签名（可能是 `chat`、`send_message` 等）。如果 ChatService 走的是 `MainAgent`（agentic 模式），需要确认 agent 内部调用 retriever 的位置，把 `background_tasks` 透传进去。如果 agent 是独立线程/异步任务执行，BackgroundTasks 可能无法直接透传，此时需要改成在 agent 完成后手动触发统计更新（用 `asyncio.create_task` 或直接同步调用 `update_chunk_stats`）。

- [ ] **Step 5: 手动验证（发起一次问答，检查统计写入）**

```bash
cd backend
# 启动服务，发起一次问答，然后检查 chunk 统计
python -c "
from app.db.session import SessionLocal
from app.entities.database import DocumentChunk
db = SessionLocal()
chunks = db.query(DocumentChunk).filter(DocumentChunk.access_count > 0).all()
print(f'chunks with access_count > 0: {len(chunks)}')
for c in chunks[:3]:
    print(f'chunk {c.id}: access_count={c.access_count}, last_accessed_at={c.last_accessed_at}, avg_score={c.avg_score}')
db.close()
"
```

Expected: 问答后至少有一个 chunk 的 access_count > 0

- [ ] **Step 6: 提交**

```bash
git add backend/app/rag/retriever.py backend/app/services/chat_service.py backend/app/api/chat.py
git commit -m "feat(retrieval): update chunk stats via BackgroundTasks after retrieval"
```

---

## Phase 4: Conflict Detection

### Task 12: ConflictService - LLM 判冲突核心

**Files:**
- Create: `backend/app/services/conflict_service.py`
- Test: `backend/tests/unit/test_conflict_service.py`

**Interfaces:**
- Produces: `ConflictService.judge_conflicts(new_chunk_content: str, candidates: List[Dict]) -> List[Dict]`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_conflict_service.py`：

```python
"""测试 ConflictService 的 LLM 冲突判定逻辑（mock LLM）。"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.conflict_service import ConflictService


def test_judge_conflicts_returns_empty_when_no_candidates():
    """没有候选时返回空列表"""
    svc = ConflictService.__new__(ConflictService)
    result = svc.judge_conflicts("新内容", [])
    assert result == []


def test_judge_conflicts_parses_llm_response():
    """能正确解析 LLM 返回的 JSON 判定"""
    svc = ConflictService.__new__(ConflictService)
    candidates = [
        {"id": "old1", "content": "电池不可退货退款"},
        {"id": "old2", "content": "退款到原支付账户"},
    ]
    fake_llm_response = MagicMock()
    fake_llm_response.content = '''[
        {"old_id": "old1", "conflict": true, "confidence": 0.9, "reason": "结论矛盾"},
        {"old_id": "old2", "conflict": false, "confidence": 0.1, "reason": "讲不同事"}
    ]'''
    with patch.object(svc, "_invoke_llm", return_value=fake_llm_response):
        result = svc.judge_conflicts("所有商品都可7天无理由退款", candidates)
    
    assert len(result) == 2
    assert result[0]["old_id"] == "old1"
    assert result[0]["conflict"] is True
    assert result[0]["confidence"] == 0.9
    assert result[1]["conflict"] is False


def test_judge_conflicts_returns_empty_on_llm_failure():
    """LLM 异常时返回空列表（不阻塞）"""
    svc = ConflictService.__new__(ConflictService)
    with patch.object(svc, "_invoke_llm", side_effect=Exception("LLM down")):
        result = svc.judge_conflicts("新内容", [{"id": "x", "content": "旧内容"}])
    assert result == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_conflict_service.py -v`
Expected: FAIL with `ModuleNotFoundError` 或 `AttributeError`

- [ ] **Step 3: 实现 ConflictService**

创建 `backend/app/services/conflict_service.py`：

```python
"""
冲突检测服务

新文档入库后，对每个新 chunk 检索相似旧 chunk，用 LLM 判定是否语义冲突。
高置信度自动作废，低置信度转人工审核。
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import DocumentChunk
from app.llm.providers import get_rewrite_llm, invoke_llm_threadsafe
from app.rag.vector_store import MilvusStore

tracer = get_tracer("conflict")


class ConflictService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()
        self.llm = get_rewrite_llm()

    def judge_conflicts(self, new_content: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        用 LLM 判断 new_content 与每个 candidate 是否冲突。

        Args:
            new_content: 新 chunk 的文本内容
            candidates: 候选旧 chunk 列表，每项含 id 和 content

        Returns:
            判定结果列表，每项含 old_id / conflict / confidence / reason
        """
        if not candidates:
            return []

        prompt = self._build_prompt(new_content, candidates)
        try:
            response = self._invoke_llm(prompt)
            return self._parse_response(response, candidates)
        except Exception as e:
            print(f"[ConflictService] LLM 判冲突失败: {e}")
            return []

    def _invoke_llm(self, prompt: str):
        return invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])

    def _build_prompt(self, new_content: str, candidates: List[Dict]) -> str:
        cand_text = "\n".join(
            f"[{i}] id={c['id']}\n内容: {c['content']}"
            for i, c in enumerate(candidates)
        )
        return f"""判断新内容与下列每条旧内容是否语义冲突（讲同一件事但结论不同/已过期）。

新内容：{new_content}

旧内容列表：
{cand_text}

要求：
1. 对每条旧内容判断是否与新内容冲突
2. conflict=true 表示讲同一事但结论矛盾，新内容应取代旧内容
3. conflict=false 表示讲不同事、互补关系、或无关
4. confidence 范围 0-1

只输出 JSON 数组，不要其他文字：
[{{"old_id": "<id>", "conflict": true/false, "confidence": 0.0, "reason": "<简短原因>"}}]"""

    def _parse_response(self, response, candidates: List[Dict]) -> List[Dict]:
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON 数组（容错：LLM 可能加额外文字）
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group(0))
            # 校验结构
            valid_ids = {c["id"] for c in candidates}
            return [
                {
                    "old_id": item.get("old_id"),
                    "conflict": bool(item.get("conflict", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "reason": item.get("reason", "")
                }
                for item in result
                if item.get("old_id") in valid_ids
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[ConflictService] 解析 LLM 响应失败: {e}")
            return []
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_conflict_service.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/conflict_service.py backend/tests/unit/test_conflict_service.py
git commit -m "feat(conflict): add ConflictService with LLM-based conflict judgment"
```

---

### Task 13: ConflictService - 检测管道编排

**Files:**
- Modify: `backend/app/services/conflict_service.py`（加 `detect_for_document` 方法）

**Interfaces:**
- Produces: `ConflictService.detect_for_document(doc_id: int) -> None`（异步后台调用）

- [ ] **Step 1: 在 ConflictService 加 detect_for_document 方法**

在 `backend/app/services/conflict_service.py` 的 `ConflictService` 类中追加：

```python
    def detect_for_document(self, doc_id: int) -> None:
        """
        对指定文档的所有 chunk 跑冲突检测管道。
        作为后台任务调用，不抛异常（失败只记日志）。
        """
        from app.entities.database import Document
        with tracer.start_as_current_span("conflict.detect_for_document") as span:
            span.set_attribute("conflict.doc_id", doc_id)
            try:
                document = self.db.query(Document).filter_by(id=doc_id).first()
                if not document:
                    print(f"[ConflictService] document {doc_id} not found")
                    return

                # 标记检测中
                document.conflict_check_status = "in_progress"
                document.conflict_check_started_at = datetime.utcnow()
                self.db.commit()

                new_chunks = self.db.query(DocumentChunk).filter_by(
                    document_id=doc_id, status="active"
                ).all()
                total = len(new_chunks)
                print(f"[ConflictService] doc {doc_id}: detecting conflicts for {total} chunks")

                processed = 0
                for chunk in new_chunks:
                    try:
                        self._detect_for_single_chunk(chunk, doc_id)
                    except Exception as e:
                        print(f"[ConflictService] chunk {chunk.id} detect failed: {e}")
                    processed += 1
                    document.conflict_check_progress = f"{processed}/{total}"
                    self.db.commit()

                document.conflict_check_status = "completed"
                document.conflict_check_completed_at = datetime.utcnow()
                self.db.commit()
                print(f"[ConflictService] doc {doc_id}: detection complete")

            except Exception as e:
                print(f"[ConflictService] detect_for_document failed: {e}")
                try:
                    document = self.db.query(Document).filter_by(id=doc_id).first()
                    if document:
                        document.conflict_check_status = "failed"
                        self.db.commit()
                except Exception:
                    pass

    def _detect_for_single_chunk(self, new_chunk: DocumentChunk, new_doc_id: int) -> None:
        """对单个新 chunk 检测冲突"""
        if not new_chunk.milvus_id or not new_chunk.content:
            return

        # Step 1: Milvus 检索相似旧 chunk（排除本文档）
        query_vector = self.vector_store.embed_query(new_chunk.content)
        candidates = self.vector_store.search_vectors(
            "chunks", query_vector, top_k=5,
            filter_expr=f"status == 'active' && document_id != {new_doc_id}"
        )
        if not candidates:
            return

        # Step 2: LLM 判冲突
        cand_for_llm = [{"id": c["id"], "content": c["content"]} for c in candidates]
        judgments = self.judge_conflicts(new_chunk.content, cand_for_llm)

        # Step 3: 按置信度分流
        high_threshold = self.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE
        low_threshold = self.settings.CONFLICT_DETECTION_LOW_CONFIDENCE

        for j in judgments:
            if not j["conflict"]:
                continue
            old_milvus_id = j["old_id"]
            old_chunk = self.db.query(DocumentChunk).filter_by(milvus_id=old_milvus_id).first()
            if not old_chunk or old_chunk.status != "active":
                continue

            confidence = j["confidence"]
            now = datetime.utcnow()

            if confidence >= high_threshold:
                # 自动作废
                old_chunk.status = "superseded"
                old_chunk.superseded_at = now
                old_chunk.conflict_with_chunk_id = new_chunk.milvus_id
                old_chunk.conflict_detected_at = now
                old_chunk.confidence = confidence
                old_chunk.review_reason = f"auto:conflict_with:{new_chunk.milvus_id}:{j['reason']}"
                self.db.commit()
                # 同步 Milvus
                self.vector_store.upsert_status("chunks", old_milvus_id, "superseded")
                print(f"[ConflictService] auto-superseded chunk {old_chunk.id} (confidence={confidence:.2f})")
            elif confidence >= low_threshold:
                # 转人工
                old_chunk.status = "pending_review"
                old_chunk.conflict_with_chunk_id = new_chunk.milvus_id
                old_chunk.conflict_detected_at = now
                old_chunk.confidence = confidence
                old_chunk.review_reason = f"review:conflict_with:{new_chunk.milvus_id}:{j['reason']}"
                self.db.commit()
                self.vector_store.upsert_status("chunks", old_milvus_id, "pending_review")
                print(f"[ConflictService] pending_review chunk {old_chunk.id} (confidence={confidence:.2f})")
```

- [ ] **Step 2: 写管道集成测试（mock vector_store + LLM）**

在 `backend/tests/unit/test_conflict_service.py` 追加：

```python
def test_detect_for_single_chunk_auto_supersede_on_high_confidence():
    """高置信度时自动作废旧 chunk"""
    from unittest.mock import MagicMock
    from app.services.conflict_service import ConflictService
    from datetime import datetime

    svc = ConflictService.__new__(ConflictService)
    svc.settings = MagicMock()
    svc.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE = 0.85
    svc.settings.CONFLICT_DETECTION_LOW_CONFIDENCE = 0.5

    new_chunk = MagicMock()
    new_chunk.milvus_id = "new1"
    new_chunk.content = "所有商品都可7天无理由退款"

    old_chunk = MagicMock()
    old_chunk.id = 100
    old_chunk.milvus_id = "old1"
    old_chunk.status = "active"

    # mock db query chain
    svc.db = MagicMock()
    svc.db.query.return_value.filter_by.return_value.first.return_value = old_chunk

    # mock vector_store
    svc.vector_store = MagicMock()
    svc.vector_store.embed_query.return_value = [0.1] * 1024
    svc.vector_store.search_vectors.return_value = [
        {"id": "old1", "content": "电池不可退货退款"}
    ]

    # mock judge_conflicts
    svc.judge_conflicts = MagicMock(return_value=[
        {"old_id": "old1", "conflict": True, "confidence": 0.92, "reason": "结论矛盾"}
    ])

    svc._detect_for_single_chunk(new_chunk, new_doc_id=999)

    assert old_chunk.status == "superseded"
    assert old_chunk.confidence == 0.92
    svc.vector_store.upsert_status.assert_called_once_with("chunks", "old1", "superseded")


def test_detect_for_single_chunk_pending_review_on_medium_confidence():
    """中置信度转人工审核"""
    from unittest.mock import MagicMock
    from app.services.conflict_service import ConflictService

    svc = ConflictService.__new__(ConflictService)
    svc.settings = MagicMock()
    svc.settings.CONFLICT_DETECTION_HIGH_CONFIDENCE = 0.85
    svc.settings.CONFLICT_DETECTION_LOW_CONFIDENCE = 0.5

    new_chunk = MagicMock()
    new_chunk.milvus_id = "new1"
    new_chunk.content = "新内容"

    old_chunk = MagicMock()
    old_chunk.id = 100
    old_chunk.milvus_id = "old1"
    old_chunk.status = "active"

    svc.db = MagicMock()
    svc.db.query.return_value.filter_by.return_value.first.return_value = old_chunk
    svc.vector_store = MagicMock()
    svc.vector_store.embed_query.return_value = [0.1] * 1024
    svc.vector_store.search_vectors.return_value = [{"id": "old1", "content": "旧内容"}]
    svc.judge_conflicts = MagicMock(return_value=[
        {"old_id": "old1", "conflict": True, "confidence": 0.6, "reason": "可能冲突"}
    ])

    svc._detect_for_single_chunk(new_chunk, new_doc_id=999)

    assert old_chunk.status == "pending_review"
    svc.vector_store.upsert_status.assert_called_once_with("chunks", "old1", "pending_review")
```

- [ ] **Step 3: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_conflict_service.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 4: 提交**

```bash
git add backend/app/services/conflict_service.py backend/tests/unit/test_conflict_service.py
git commit -m "feat(conflict): add detection pipeline with auto-supersede and pending-review"
```

---

### Task 14: document_service 上传后触发冲突检测

**Files:**
- Modify: `backend/app/services/document_service.py`
- Modify: `backend/app/api/documents.py`（透传 BackgroundTasks）

**Interfaces:**
- Consumes: `ConflictService`
- Produces: `DocumentService` 上传完成后注册冲突检测后台任务

- [ ] **Step 1: 在 DocumentService 注入 ConflictService**

在 `backend/app/services/document_service.py` 顶部加 import：
```python
from app.services.conflict_service import ConflictService
```

修改 `DocumentService.__init__` 加可选参数：
```python
    def __init__(
        self,
        db: Session,
        vector_store: MilvusStore,
        retriever: HybridRetriever = None,
        conflict_service: ConflictService = None
    ):
        self.db = db
        self.vector_store = vector_store
        self.retriever = retriever
        self.conflict_service = conflict_service
        self.splitter = ThreeLayerSplitter()
```

- [ ] **Step 2: 在 process_document 末尾触发冲突检测**

在 `process_document` 方法最后（`return True` 前）加：

```python
            # 触发冲突检测（后台任务）
            if self.conflict_service and background_tasks:
                # 标记文档待检测
                document.conflict_check_status = "pending"
                self.db.commit()
                background_tasks.add_task(self._run_conflict_detection, doc_id)
            
            return True
```

注意 `process_document` 签名需要接收 `background_tasks`。当前 `process_document(self, doc_id: int)` 没有。修改为：

```python
    def process_document(self, doc_id: int, background_tasks=None) -> bool:
```

并在 `upload_document` 调用时传入：
```python
        if background_tasks:
            background_tasks.add_task(self.process_document, db_document.id, background_tasks)
        else:
            self.process_document(db_document.id)
```

> 这里有个细节：`background_tasks.add_task` 注册的函数会在响应发出后执行，但该函数内部再 `background_tasks.add_task` 注册冲突检测，是否可行？答：可行，FastAPI 的 BackgroundTasks 在响应后执行时，内部再 add_task 会追加到队列末尾继续执行。

在类中加 `_run_conflict_detection` 辅助方法：

```python
    def _run_conflict_detection(self, doc_id: int):
        """后台执行冲突检测（独立 db session）"""
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            svc = ConflictService(db, self.vector_store)
            svc.detect_for_document(doc_id)
        except Exception as e:
            print(f"[DocumentService] conflict detection failed for doc {doc_id}: {e}")
        finally:
            db.close()
```

- [ ] **Step 3: 在 core/dependencies.py 注册 ConflictService 工厂**

在 `backend/app/core/dependencies.py` 末尾加：

```python
def get_conflict_service(
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store)
) -> ConflictService:
    return ConflictService(db, vector_store)
```

并在文件顶部加 import：
```python
from app.services.conflict_service import ConflictService
from app.db.session import get_db
from fastapi import Depends
```

- [ ] **Step 4: 修改 admin.py 和 documents.py 注入 conflict_service**

在 `backend/app/api/admin.py` 的 `list_all_documents` 和 `admin_delete_document` 中，构造 `DocumentService` 时传 `conflict_service`：

```python
from app.services.conflict_service import ConflictService
from app.core.dependencies import get_conflict_service

@router.get("/documents", response_model=List[DocumentWithConflictStatusResponse])
async def list_all_documents(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store),
    conflict_service: ConflictService = Depends(get_conflict_service)
):
    doc_service = DocumentService(db, vector_store, conflict_service=conflict_service)
    return doc_service.get_documents(skip=skip, limit=limit)
```

同样修改 `documents.py` 的上传接口，注入 `conflict_service`。

- [ ] **Step 5: 手动验证（上传文档后看 conflict_check_status 变化）**

```bash
cd backend
# 上传一个文档后立即查
python -c "
from app.db.session import SessionLocal
from app.entities.database import Document
db = SessionLocal()
doc = db.query(Document).order_by(Document.id.desc()).first()
print(f'latest doc {doc.id}: conflict_check_status={doc.conflict_check_status}, progress={doc.conflict_check_progress}')
db.close()
"
```

Expected: 上传后 `conflict_check_status` 从 `pending` -> `in_progress` -> `completed`

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/document_service.py backend/app/core/dependencies.py backend/app/api/admin.py backend/app/api/documents.py
git commit -m "feat(document_service): trigger conflict detection after document processing"
```

---

### Task 15: Admin API - GET /admin/chunks 列表

**Files:**
- Modify: `backend/app/api/admin.py`

**Interfaces:**
- Produces: `GET /admin/chunks?status=pending_review|superseded|archived|active` 返回 `List[ChunkDetailResponse]`

- [ ] **Step 1: 在 admin.py 加 chunks 列表接口**

```python
from app.entities.database import DocumentChunk
from app.entities.schemas import ChunkDetailResponse, DocumentWithConflictStatusResponse

@router.get("/chunks", response_model=List[ChunkDetailResponse])
async def list_chunks(
    status: Optional[str] = None,
    archived_reason: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """列出 chunks，可按 status / archived_reason 过滤"""
    query = db.query(DocumentChunk)
    if status:
        query = query.filter(DocumentChunk.status == status)
    if archived_reason:
        query = query.filter(DocumentChunk.archived_reason == archived_reason)
    chunks = query.order_by(DocumentChunk.id.desc()).offset(skip).limit(limit).all()
    
    # join conflict_with_chunk 的 content（用于 pending_review / superseded 展示新 chunk 内容）
    result = []
    for c in chunks:
        item = ChunkDetailResponse.model_validate(c)
        if c.conflict_with_chunk_id:
            new_chunk = db.query(DocumentChunk).filter_by(milvus_id=c.conflict_with_chunk_id).first()
            if new_chunk:
                item.conflict_with_content = new_chunk.content
        result.append(item)
    return result
```

> 需要在 admin.py 顶部加 `from typing import Optional` 和 `from app.entities.schemas import ChunkDetailResponse`

- [ ] **Step 2: 修改 /admin/documents 返回冲突检测状态**

把 `list_all_documents` 的 `response_model` 改为 `List[DocumentWithConflictStatusResponse]`（如果 Task 14 已改则跳过）。

- [ ] **Step 3: 手动验证**

```bash
cd backend
# 启动服务后
curl -H "Authorization: Bearer <admin_token>" http://localhost:8000/api/v1/admin/chunks?status=pending_review
```

Expected: 返回 pending_review 的 chunk 列表，每条含 conflict_with_content

- [ ] **Step 4: 提交**

```bash
git add backend/app/api/admin.py
git commit -m "feat(admin): add GET /admin/chunks with status filter and conflict join"
```

---

### Task 16: Admin API - PATCH /admin/chunks/{id} 动作

**Files:**
- Modify: `backend/app/api/admin.py`

**Interfaces:**
- Produces: `PATCH /admin/chunks/{id}` 接收 `ChunkPatchRequest`，执行 confirm/dismiss/archive/restore/hard_delete

- [ ] **Step 1: 在 admin.py 加 PATCH 接口**

```python
from app.entities.schemas import ChunkPatchRequest
from datetime import datetime
from app.core.dependencies import get_vector_store

@router.patch("/chunks/{chunk_id}")
async def patch_chunk(
    chunk_id: int,
    data: ChunkPatchRequest,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store)
):
    """对 chunk 执行动作：confirm / dismiss / archive / restore / hard_delete"""
    chunk = db.query(DocumentChunk).filter_by(id=chunk_id).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    action = data.action
    now = datetime.utcnow()

    if action == "confirm":
        # pending_review -> superseded
        if chunk.status != "pending_review":
            raise HTTPException(status_code=400, detail="Only pending_review can be confirmed")
        chunk.status = "superseded"
        chunk.superseded_at = now
        chunk.reviewed_by = current_user.id
        chunk.reviewed_at = now
        db.commit()
        if chunk.milvus_id:
            vector_store.upsert_status("chunks", chunk.milvus_id, "superseded")

    elif action == "dismiss":
        # pending_review -> active（驳回，清空冲突字段）
        if chunk.status != "pending_review":
            raise HTTPException(status_code=400, detail="Only pending_review can be dismissed")
        chunk.status = "active"
        chunk.conflict_with_chunk_id = None
        chunk.conflict_detected_at = None
        chunk.confidence = None
        chunk.review_reason = None
        chunk.reviewed_by = current_user.id
        chunk.reviewed_at = now
        db.commit()
        if chunk.milvus_id:
            vector_store.upsert_status("chunks", chunk.milvus_id, "active")

    elif action == "archive":
        # active -> archived（manual）
        if chunk.status != "active":
            raise HTTPException(status_code=400, detail="Only active can be archived")
        chunk.status = "archived"
        chunk.archived_reason = "manual"
        chunk.archived_at = now
        chunk.reviewed_by = current_user.id
        chunk.reviewed_at = now
        db.commit()
        if chunk.milvus_id:
            vector_store.upsert_status("chunks", chunk.milvus_id, "archived")

    elif action == "restore":
        # archived -> active（保留统计字段）
        if chunk.status != "archived":
            raise HTTPException(status_code=400, detail="Only archived can be restored")
        chunk.status = "active"
        chunk.archived_reason = None
        chunk.archived_at = None
        chunk.reviewed_by = current_user.id
        chunk.reviewed_at = now
        db.commit()
        if chunk.milvus_id:
            vector_store.upsert_status("chunks", chunk.milvus_id, "active")

    elif action == "hard_delete":
        # 物理删除 PG + Milvus
        if chunk.milvus_id:
            vector_store.delete_vectors("chunks", ids=[chunk.milvus_id])
        db.delete(chunk)
        db.commit()

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    return {"message": f"Chunk {chunk_id} {action} done"}
```

- [ ] **Step 2: 手动验证（pending_review -> confirm）**

```bash
# 假设有 pending_review 的 chunk id=10
curl -X PATCH -H "Authorization: Bearer <admin_token>" \
     -H "Content-Type: application/json" \
     -d '{"action":"confirm"}' \
     http://localhost:8000/api/v1/admin/chunks/10
```

Expected: 返回 `{"message": "Chunk 10 confirm done"}`，且 chunk status 变为 superseded

- [ ] **Step 3: 提交**

```bash
git add backend/app/api/admin.py
git commit -m "feat(admin): add PATCH /admin/chunks/{id} for confirm/dismiss/archive/restore/hard_delete"
```

---

## Phase 5: Cold Knowledge

### Task 17: ColdKnowledgeService - 扫描逻辑

**Files:**
- Create: `backend/app/services/cold_knowledge_service.py`
- Test: `backend/tests/unit/test_cold_knowledge_service.py`

**Interfaces:**
- Produces: `ColdKnowledgeService.sweep() -> Dict[str, int]`（返回各类归档数量）

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_cold_knowledge_service.py`：

```python
"""测试冷知识扫描规则。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.cold_knowledge_service import ColdKnowledgeService


def _make_chunk(status="active", last_accessed_at=None, access_count=0, hit_count=0, avg_score=0.0, created_at=None, archived_reason=None):
    c = MagicMock()
    c.status = status
    c.last_accessed_at = last_accessed_at
    c.access_count = access_count
    c.hit_count = hit_count
    c.avg_score = avg_score
    c.created_at = created_at or datetime.utcnow()
    c.archived_reason = archived_reason
    c.milvus_id = "milvus_x"
    return c


def test_timeout_rule_archives_old_unaccessed_chunk():
    """90 天未被访问的 chunk 归档为 timeout"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    old = _make_chunk(last_accessed_at=datetime.utcnow() - timedelta(days=100))
    svc.db.query.return_value.filter.return_value.all.return_value = [old]

    stats = svc.sweep()
    assert stats["timeout"] == 1
    assert old.status == "archived"
    assert old.archived_reason == "timeout"


def test_low_freq_rule_archives_unpopular_old_chunk():
    """上传 30+ 天且命中 < 2 次的归档为 low_freq"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=10),  # 不触发 timeout
        access_count=1,
        created_at=datetime.utcnow() - timedelta(days=40)  # 上传 40 天
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["low_freq"] == 1
    assert chunk.archived_reason == "low_freq"


def test_low_quality_rule_archives_low_score_chunk():
    """hit_count >= 5 且 avg_score < 0.3 的归档为 low_quality"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=1),
        access_count=10,
        hit_count=6,
        avg_score=0.2,
        created_at=datetime.utcnow() - timedelta(days=10)
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["low_quality"] == 1
    assert chunk.archived_reason == "low_quality"


def test_no_archive_for_fresh_chunk():
    """新 chunk（上传 < 30 天）即使命中少也不归档"""
    svc = ColdKnowledgeService.__new__(ColdKnowledgeService)
    svc.settings = MagicMock()
    svc.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS = 90
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2
    svc.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3
    svc.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5
    svc.db = MagicMock()
    svc.vector_store = MagicMock()

    chunk = _make_chunk(
        last_accessed_at=datetime.utcnow() - timedelta(days=1),
        access_count=1,
        hit_count=1,
        avg_score=0.5,
        created_at=datetime.utcnow() - timedelta(days=5)
    )
    svc.db.query.return_value.filter.return_value.all.return_value = [chunk]

    stats = svc.sweep()
    assert stats["timeout"] + stats["low_freq"] + stats["low_quality"] == 0
    assert chunk.status == "active"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_cold_knowledge_service.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 ColdKnowledgeService**

创建 `backend/app/services/cold_knowledge_service.py`：

```python
"""
冷知识识别与清理服务

定期扫描 active chunk，按四类信号判定是否归档：
- timeout: last_accessed_at > 90 天
- low_freq: 上传 > 30 天且 access_count < 2
- low_quality: hit_count >= 5 且 avg_score < 0.3
- manual: admin 手动归档（走 PATCH 接口，不走本服务）

归档后 90 天由 hard_delete_sweep 物理删除。
"""
from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import DocumentChunk
from app.rag.vector_store import MilvusStore

tracer = get_tracer("cold_knowledge")


class ColdKnowledgeService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()

    def sweep(self) -> Dict[str, int]:
        """扫描所有 active chunk，按规则归档。返回各类归档数量。"""
        with tracer.start_as_current_span("cold_knowledge.sweep") as span:
            stats = {"timeout": 0, "low_freq": 0, "low_quality": 0}
            now = datetime.utcnow()

            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.status == "active"
            ).all()

            print(f"[ColdKnowledge] scanning {len(chunks)} active chunks")
            for chunk in chunks:
                reason = self._classify(chunk, now)
                if reason:
                    self._archive(chunk, reason, now)
                    stats[reason] += 1

            self.db.commit()
            span.set_attribute("cold_knowledge.archived_total", sum(stats.values()))
            for k, v in stats.items():
                span.set_attribute(f"cold_knowledge.archived.{k}", v)
            print(f"[ColdKnowledge] done: {stats}")
            return stats

    def _classify(self, chunk: DocumentChunk, now: datetime) -> str:
        """判定 chunk 应归档的原因，返回 None 表示不归档"""
        # 规则 1: timeout
        if chunk.last_accessed_at:
            days_since_access = (now - chunk.last_accessed_at).days
            if days_since_access > self.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS:
                return "timeout"
        else:
            # 从未访问过，看 created_at
            if chunk.created_at:
                days_since_create = (now - chunk.created_at).days
                if days_since_create > self.settings.COLD_KNOWLEDGE_TIMEOUT_DAYS:
                    return "timeout"

        # 规则 2: low_freq
        if chunk.created_at:
            days_since_create = (now - chunk.created_at).days
            if (days_since_create > self.settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS
                    and chunk.access_count < self.settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD):
                return "low_freq"

        # 规则 3: low_quality
        if (chunk.hit_count >= self.settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS
                and chunk.avg_score < self.settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE):
            return "low_quality"

        return None

    def _archive(self, chunk: DocumentChunk, reason: str, now: datetime) -> None:
        """归档单个 chunk（PG + Milvus）"""
        chunk.status = "archived"
        chunk.archived_reason = reason
        chunk.archived_at = now
        if chunk.milvus_id:
            try:
                self.vector_store.upsert_status("chunks", chunk.milvus_id, "archived")
            except Exception as e:
                print(f"[ColdKnowledge] upsert_status failed for chunk {chunk.id}: {e}")

    def hard_delete_sweep(self) -> int:
        """扫描归档超过保留期的 chunk，物理删除。返回删除数量。"""
        with tracer.start_as_current_span("cold_knowledge.hard_delete") as span:
            retention_days = self.settings.COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS
            cutoff = datetime.utcnow() - timedelta(days=retention_days)

            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.status == "archived",
                DocumentChunk.archived_at < cutoff
            ).all()

            print(f"[ColdKnowledge] hard deleting {len(chunks)} chunks archived before {cutoff}")
            for chunk in chunks:
                if chunk.milvus_id:
                    try:
                        self.vector_store.delete_vectors("chunks", ids=[chunk.milvus_id])
                    except Exception as e:
                        print(f"[ColdKnowledge] milvus delete failed for chunk {chunk.id}: {e}")
                self.db.delete(chunk)

            self.db.commit()
            span.set_attribute("cold_knowledge.hard_deleted", len(chunks))
            return len(chunks)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_cold_knowledge_service.py -v`
Expected: PASS（4 个测试）

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/cold_knowledge_service.py backend/tests/unit/test_cold_knowledge_service.py
git commit -m "feat(cold_knowledge): add sweep and hard_delete_sweep with rule-based classification"
```

---

### Task 18: 注册 ColdKnowledgeService 依赖

**Files:**
- Modify: `backend/app/core/dependencies.py`

**Interfaces:**
- Produces: `get_cold_knowledge_service(db, vector_store) -> ColdKnowledgeService`

- [ ] **Step 1: 在 dependencies.py 加工厂函数**

```python
from app.services.cold_knowledge_service import ColdKnowledgeService

def get_cold_knowledge_service(
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store)
) -> ColdKnowledgeService:
    return ColdKnowledgeService(db, vector_store)
```

- [ ] **Step 2: 提交**

```bash
git add backend/app/core/dependencies.py
git commit -m "feat(deps): register ColdKnowledgeService factory"
```

---

### Task 19: APScheduler 集成与启动

**Files:**
- Create: `backend/app/core/scheduler.py`
- Modify: `backend/main.py:58-97`（startup 事件）

**Interfaces:**
- Produces: `setup_scheduler(app)` 在 startup 调用；调度 `cold_knowledge_sweep` 和 `hard_delete_sweep`

- [ ] **Step 1: 添加 apscheduler 依赖**

```bash
cd backend
pip install APScheduler
```

并在 `backend/requirements.txt`（如果有）加 `APScheduler>=3.10`。如果没有 requirements.txt，跳过。

- [ ] **Step 2: 创建 scheduler.py**

创建 `backend/app/core/scheduler.py`：

```python
"""
APScheduler 调度器

注册两个定时任务：
- cold_knowledge_sweep: 每天 3 点扫描归档冷知识
- hard_delete_sweep: 每天 4 点硬删除过期归档

使用 SQLAlchemyJobStore 持久化到 PG/SQLite，重启不丢。
多副本部署时通过 PG 行锁保证单实例执行（TODO: 后续如需多副本再加）。
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.core.observability import get_tracer

tracer = get_tracer("scheduler")

_scheduler: BackgroundScheduler = None


def _run_cold_knowledge_sweep():
    """定时任务：冷知识扫描"""
    from app.db.session import SessionLocal
    from app.core.dependencies import get_vector_store
    from app.services.cold_knowledge_service import ColdKnowledgeService
    db = SessionLocal()
    try:
        with tracer.start_as_current_span("scheduler.cold_knowledge_sweep"):
            svc = ColdKnowledgeService(db, get_vector_store())
            stats = svc.sweep()
            print(f"[Scheduler] cold_knowledge_sweep: {stats}")
    except Exception as e:
        print(f"[Scheduler] cold_knowledge_sweep failed: {e}")
    finally:
        db.close()


def _run_hard_delete_sweep():
    """定时任务：硬删除过期归档"""
    from app.db.session import SessionLocal
    from app.core.dependencies import get_vector_store
    from app.services.cold_knowledge_service import ColdKnowledgeService
    db = SessionLocal()
    try:
        with tracer.start_as_current_span("scheduler.hard_delete_sweep"):
            svc = ColdKnowledgeService(db, get_vector_store())
            count = svc.hard_delete_sweep()
            print(f"[Scheduler] hard_delete_sweep: deleted {count} chunks")
    except Exception as e:
        print(f"[Scheduler] hard_delete_sweep failed: {e}")
    finally:
        db.close()


def setup_scheduler():
    """初始化并启动调度器（应用启动时调用一次）"""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    settings = get_settings()
    from app.db.session import engine
    # 复用现有 engine 的 URL（避免硬编码）
    jobstore_url = str(engine.url).replace("sqlite:///", "sqlite:///")  # 兼容 SQLite/PG
    # 或者直接用 settings
    jobstore_url = settings.DATABASE_URL

    _scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=jobstore_url)},
        timezone="Asia/Shanghai"
    )

    # 冷知识扫描
    trigger = CronTrigger.from_crontab(settings.COLD_KNOWLEDGE_SWEEP_CRON)
    _scheduler.add_job(
        _run_cold_knowledge_sweep,
        trigger=trigger,
        id="cold_knowledge_sweep",
        replace_existing=True
    )

    # 硬删除扫描
    trigger2 = CronTrigger.from_crontab(settings.HARD_DELETE_SWEEP_CRON)
    _scheduler.add_job(
        _run_hard_delete_sweep,
        trigger=trigger2,
        id="hard_delete_sweep",
        replace_existing=True
    )

    _scheduler.start()
    print(f"[Scheduler] started: cold_sweep={settings.COLD_KNOWLEDGE_SWEEP_CRON}, hard_delete={settings.HARD_DELETE_SWEEP_CRON}")
    return _scheduler


def shutdown_scheduler():
    """关闭调度器（应用关闭时调用）"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
```

- [ ] **Step 3: 在 main.py startup 调用 setup_scheduler**

在 `backend/main.py` 的 `startup_event` 函数末尾（`print("Application startup complete!")` 前）加：

```python
    # 4. 启动定时任务调度器
    print("[4/4] Starting scheduler...")
    from app.core.scheduler import setup_scheduler
    setup_scheduler()
```

并在文件底部加 shutdown 事件：

```python
@app.on_event("shutdown")
async def shutdown_event():
    from app.core.scheduler import shutdown_scheduler
    shutdown_scheduler()
    print("Scheduler shut down")
```

- [ ] **Step 4: 手动验证调度器启动**

```bash
cd backend
python -c "
from app.core.scheduler import setup_scheduler, shutdown_scheduler
sched = setup_scheduler()
print('jobs:', [j.id for j in sched.get_jobs()])
shutdown_scheduler()
"
```

Expected: 输出 `jobs: ['cold_knowledge_sweep', 'hard_delete_sweep']`

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/scheduler.py backend/main.py
git commit -m "feat(scheduler): integrate APScheduler for cold knowledge and hard delete sweeps"
```

---

## Phase 6: Migration & Verification

### Task 20: 执行迁移 + 端到端验证

**Files:**
- 无新文件，执行已有脚本

- [ ] **Step 1: 执行 PG 迁移**

```bash
cd backend
sqlite3 data/sqlite/app.db < scripts/migrate_pg.sql
sqlite3 data/sqlite/app.db ".schema document_chunks" | grep -E "content_hash|status|conflict_with"
sqlite3 data/sqlite/app.db ".schema documents" | grep conflict_check
```

Expected: 新字段出现在 schema 中

- [ ] **Step 2: 执行 Milvus 迁移**

```bash
cd backend
python scripts/migrate_milvus.py
```

Expected: 输出"Migration complete!"，所有 COMPLETED 文档重处理完成

- [ ] **Step 3: 验证检索过滤生效**

```bash
cd backend
python -c "
from app.db.session import SessionLocal
from app.entities.database import DocumentChunk
from app.core.dependencies import get_vector_store
db = SessionLocal()
# 看现有 chunk 状态分布
from sqlalchemy import func
dist = db.query(DocumentChunk.status, func.count(DocumentChunk.id)).group_by(DocumentChunk.status).all()
print('chunk status distribution:', dist)
# 试检索
vs = get_vector_store()
results = vs.search('chunks', '测试查询', top_k=5)
print(f'search results (default active filter): {len(results)}')
db.close()
"
```

Expected: chunk 状态分布显示全 active；检索返回结果

- [ ] **Step 4: 端到端验证清单**

按以下顺序手动验证：

1. **上传新文档** -> 检查 `documents.conflict_check_status` 从 pending -> in_progress -> completed
2. **上传冲突文档**（包含与已有 chunk 矛盾的内容）-> 检查旧 chunk status 变为 superseded 或 pending_review
3. **发起问答** -> 检查命中 chunk 的 access_count / last_accessed_at 更新
4. **GET /admin/chunks?status=pending_review** -> 返回待审核列表
5. **PATCH /admin/chunks/{id} {action: confirm}** -> chunk 状态变 superseded
6. **PATCH /admin/chunks/{id} {action: archive}** -> chunk 状态变 archived
7. **PATCH /admin/chunks/{id} {action: restore}** -> chunk 状态恢复 active
8. **手动触发冷知识扫描** -> 检查老 chunk 被归档
9. **检索过滤** -> archived/superseded chunk 不出现在结果中

- [ ] **Step 5: 提交验证记录**

如果验证中发现 bug，分别修复并提交。无 bug 则无需提交。

```bash
# 如果有修复
git add <fixed files>
git commit -m "fix: <what was fixed>"
```

---

## Self-Review Checklist

实施者完成后自检：

- [ ] spec 第 3 节所有字段都在 ORM 和 Milvus schema 中体现
- [ ] spec 第 4 节冲突检测管道：检测中状态可见、pending_review 队列、自动作废审计都有 API
- [ ] spec 第 5 节冷知识：四类信号规则齐全、两段式清理（archive + hard delete）、回滚可用
- [ ] spec 第 6 节检索路径：status=='active' 过滤、统计写入走 BackgroundTasks
- [ ] spec 第 7 节迁移：PG SQL + Milvus 脚本都能跑通
- [ ] spec 第 8 节配置项都在 config.py 中
- [ ] 所有单元测试通过：`cd backend && python -m pytest tests/unit/ -v`
- [ ] 手动端到端验证清单全部通过
