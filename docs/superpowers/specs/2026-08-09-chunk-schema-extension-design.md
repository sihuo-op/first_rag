# Chunk Schema 扩容设计：更新与冷知识识别

**日期**：2026-08-09
**状态**：待审阅
**范围**：`backend/` 后端服务，Milvus `rag_chunks` collection，PostgreSQL `document_chunks` / `documents` 表

## 1. 目标

为 RAG 系统的 chunk 数据增加"更新"和"冷知识识别删除"两项能力。

**本期目标**：
- 支持整文档级增量更新：新文档上传后自动检测与旧 chunk 的语义冲突，按置信度自动作废或转人工审核
- 支持冷知识识别：基于时间、频次、质量、人工四类信号识别不再有价值的 chunk，软归档后定期硬删
- 检索路径按 `status` 过滤，保证 top_k 精度不被作废/归档 chunk 污染
- 管理后台可见：冲突检测进度、待审核队列、自动作废审计、归档列表

**本期不做**：
- 字节级完全相同 chunk 的引用计数（content_hash 相同直接跳过入库，不维护引用关系）
- 跨文档语义聚类（仅做 1对1 冲突判定，不做主题级合并）
- 消息队列引入（统计写入用 FastAPI BackgroundTasks 即可）/
- `rag_memories` collection 的 schema 改动

## 2. 核心设计原则

**字段放置策略（方案 B）**：Milvus 只放检索期必须过滤的字段，其余放 PostgreSQL。

- **Milvus**：低频写，存检索期过滤用的字段（`status`）和增量更新 diff 用的字段（`content_hash`）
- **PostgreSQL**：高频写（命中统计）+ 审核追溯元数据（冲突作废）+ 状态镜像（后台列表查询）
- **状态字段串联**：PG 是写入源，Milvus 是过滤消费方；状态变更时先改 PG，再 upsert Milvus（低频可接受）

理由：Milvus upsert 本质是"删 + 插"，每次重写整行包括 1024 维向量。命中统计是高频写（每次检索都更新），塞进 Milvus 会拖慢问答 P99。

## 3. Schema 设计

### 3.1 Milvus `rag_chunks` 新增字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `content_hash` | VARCHAR(64) | 内容 SHA256，增量更新 diff 用 |
| `status` | VARCHAR(20) | `active` / `superseded` / `pending_review` / `archived`，默认 `active` |

完整 schema：

```
id            VARCHAR(100)  PK
document_id   INT64
chunk_type    VARCHAR(20)
content       VARCHAR(65535)
content_hash  VARCHAR(64)     [NEW]
status        VARCHAR(20)     [NEW, default "active"]
embedding     FLOAT_VECTOR(1024)
```

索引不变（IVF_FLAT + COSINE，nlist=128，nprobe=10）。`status` 标量字段，Milvus 2.x 默认建标量索引。

检索时过滤表达式：`expr="status == 'active'"`

### 3.2 PostgreSQL `document_chunks` 表新增字段

按职责分四组：

#### A. 增量更新 & 状态镜像

| 字段 | 类型 | 说明 |
|---|---|---|
| `content_hash` | VARCHAR(64) | 与 Milvus 一致，diff 时只查 PG |
| `status` | VARCHAR(20) default 'active' | 镜像 Milvus status，后台查询用 |

#### B. 冲突作废元数据

| 字段 | 类型 | 说明 |
|---|---|---|
| `conflict_with_chunk_id` | VARCHAR(100) nullable | 触发冲突的新 chunk 的 milvus_id（pending_review 和 superseded 都填，作为唯一关联字段） |
| `conflict_detected_at` | DateTime nullable | 检测到冲突的时间 |
| `confidence` | FLOAT nullable | LLM 判冲突的置信度（0-1） |
| `review_reason` | Text nullable | 如 `conflict_with:<doc_id>:<chunk_hash>` |
| `superseded_at` | DateTime nullable | 被确认作废的时间（pending_review 阶段为空，confirm 后填） |
| `reviewed_by` | Integer FK -> users.id nullable | 审核人（自动作废时为空，人工 confirm/dismiss 后填） |
| `reviewed_at` | DateTime nullable | 审核时间 |

#### C. 命中统计（高频写）

| 字段 | 类型 | 说明 |
|---|---|---|
| `access_count` | Integer default 0 | 累计命中次数（每次检索 +1） |
| `last_accessed_at` | DateTime nullable | 最近一次被命中 |
| `hit_count` | Integer default 0 | 进入 top_k 的次数（access_count 子集，预留给"被采纳进 context"的语义） |
| `total_score` | FLOAT default 0.0 | 累计相似度分数 |
| `avg_score` | FLOAT default 0.0 | = total_score / hit_count |

#### D. 冷知识归档

| 字段 | 类型 | 说明 |
|---|---|---|
| `archived_reason` | VARCHAR(30) nullable | `timeout` / `low_freq` / `low_quality` / `manual` |
| `archived_at` | DateTime nullable | 进入 archived 的时间，用于硬删保留期判定 |

#### 索引

```sql
CREATE INDEX idx_document_chunks_milvus_id ON document_chunks(milvus_id);
CREATE INDEX idx_document_chunks_status ON document_chunks(status);
```

### 3.3 PostgreSQL `documents` 表新增字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `conflict_check_status` | VARCHAR(20) default 'completed' | `pending` / `in_progress` / `completed` / `failed` |
| `conflict_check_started_at` | DateTime nullable | |
| `conflict_check_completed_at` | DateTime nullable | |
| `conflict_check_progress` | VARCHAR(20) nullable | 如 `"5/50"` |

> 旧数据默认 `completed`，已存在的文档视为已检测过，不重跑。

> 文档状态机扩展为：`PENDING -> PROCESSING -> COMPLETED`，同时 `conflict_check_status` 独立走 `pending -> in_progress -> completed`。两条线解耦，文档入库完成不等于冲突检测完成。

## 4. 冲突检测流程

### 4.1 触发时机

新文档处理完成、chunk 入库后，作为后台任务异步执行（不阻塞上传响应）。

### 4.2 检测管道

```
新文档 N 个 chunk 入库（已 embed + insert 到 Milvus，status=active）
        │
        ▼
对每个新 chunk c_new：
   ┌─────────────────────────────────────────┐
   │ Step 1: 在 Milvus 检索同 document_id    │
   │         之外的相似 chunk                │
   │   search(c_new.embedding, top_k=5,     │
   │          expr="status == 'active' &&   │
   │                document_id != {new}")  │
   └─────────────────────────────────────────┘
        │ 候选集 C
        ▼
   ┌─────────────────────────────────────────┐
   │ Step 2: LLM 批量判冲突                  │
   │   prompt: 给 c_new 和候选 C，问哪些与   │
   │           c_new 语义冲突（讲同一事但    │
   │           结论不同/已过期）             │
   │   输出: [{old_id, conflict: bool,       │
   │          confidence: float, reason}]    │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌──────────────┬──────────────────────────┐
   │ 高置信 ≥0.85 │ 低置信 0.5~0.85          │
   │ -> 自动作废   │ -> pending_review         │
   │              │                          │
   │ status=      │ status=                  │
   │  superseded  │  pending_review          │
   │ superseded_  │ review_reason=           │
   │  at=now      │  conflict_with:...       │
   │ confidence=  │ confidence=...           │
   │ conflict_    │ conflict_with_chunk_id=  │
   │  with_chunk_ │  c_new.id                │
   │  id=c_new.id │ conflict_detected_at=    │
   │ conflict_    │  now                     │
   │  detected_   │                          │
   │  at=now      │                          │
   └──────────────┴──────────────────────────┘
        │
        ▼
   同步 upsert Milvus 的 status 字段（低频，可接受）
```

### 4.3 关键设计点

- **检索范围**：`document_id != new_doc_id`，避免新文档内部 chunk 互相判冲突（同文档内 chunk 本来就可能语义相近但不冲突）
- **LLM 判冲突**：用项目现有的 LLM 客户端，批量 prompt 一次判多个候选，省钱
- **阈值 0.85 / 0.5**：可配置，放到 `config.py`
- **同文档内冲突**：不做
- **字节级相同**：content_hash 相同直接跳过入库，不重复 embed

### 4.4 可见性 API（复用现有 `/admin` 路由）

不新增 `/admin/conflicts/*` 独立路径，扩展现有资源：

```
GET  /admin/documents
     -> 返回里带 conflict_check_status / conflict_check_progress
     -> 前端轮询（2-3s）展示"检测中 5/50"

GET  /admin/chunks?status=pending_review
     -> 待审核 chunk 列表
     -> 每条带：旧 chunk content + 新 chunk content（join conflict_with_chunk_id）+ confidence + review_reason

GET  /admin/chunks?status=superseded
     -> 自动作废审计列表
     -> 每条带：旧 content + 新 content + confidence + superseded_at

PATCH /admin/chunks/{id}
     -> body: {action: "confirm" | "dismiss" | "archive" | "restore" | "hard_delete"}
     -> confirm:  status -> superseded, superseded_at=now, reviewed_by, reviewed_at
                  （conflict_with_chunk_id 已在检测时填好，无需再写）
     -> dismiss:  status -> active, 清空 confidence / review_reason / conflict_with_chunk_id / conflict_detected_at, set reviewed_by / reviewed_at
     -> archive:  status -> archived, archived_reason='manual', archived_at=now, reviewed_by, reviewed_at
     -> restore:  status -> active, 清空 archived_reason / archived_at（统计字段 access_count 等保留）
     -> hard_delete: 从 PG + Milvus 物理删除
     -> 每次都同步 upsert Milvus status
```

## 5. 冷知识识别与清理

### 5.1 命中统计写入（检索热路径）

每次检索命中 chunk 后，用 FastAPI `BackgroundTasks` 异步更新 PG 统计（不阻塞问答响应）：

```
retriever 返回 top_k 结果
     │
     ▼
BackgroundTasks: update_chunk_stats(hit_chunks, scores)
     │
     ▼  对每个命中 chunk（按 milvus_id 批量更新 PG）：
     access_count   += 1
     last_accessed_at = now
     total_score    += score
     hit_count      += 1
     avg_score       = total_score / hit_count
```

统计不需要强一致，偶发丢失可接受。

### 5.2 冷知识识别（周期性扫描）

定时任务（默认每天凌晨 3 点）扫描 `status='active'` 的 chunk，按四类信号判定：

| 信号 | 规则 | archived_reason |
|---|---|---|
| 时间 | `last_accessed_at` 距今 > 90 天 | `timeout` |
| 频次 | 上传 > 30 天 且 `access_count` < 2 | `low_freq` |
| 质量 | `hit_count` >= 5 且 `avg_score` < 0.3 | `low_quality` |
| 人工 | admin 在后台直接点"归档" | `manual` |

任一规则命中即归档：`status -> archived`，写 `archived_reason` + `archived_at`，同步 upsert Milvus。

质量规则要求 `hit_count >= 5` 是为了避免样本太少误判。

### 5.3 清理策略（两段式）

```
active  ──(扫描命中/人工)──>  archived  ──(保留 90 天)──>  硬删除
                                 │
                                 └── 保留期内可回滚 / 审计
```

- **软归档**：`status=archived`，PG + Milvus 都保留，检索时被 `status=='active'` 过滤掉
- **硬删除**：归档后 90 天，定时任务从 PG + Milvus 物理删除
- **回滚**：归档保留期内，admin 可 PATCH 回 `active`（统计字段保留历史轨迹，避免再次被立刻归档）

### 5.4 调度器选型

**APScheduler + SQLAlchemy JobStore**：

- 复用现有 PG，无新基础设施
- 任务持久化，重启不丢
- 多副本部署时配 PG 行锁，保证单实例执行
- 现阶段不需要 Celery 的分布式能力

## 6. 检索路径改造

### 6.1 改造前

```
query -> embed_query -> Milvus search(top_k, 无过滤) -> 返回 chunks（含作废/归档的）
```

### 6.2 改造后

```
query
  │
  ▼
embed_query
  │
  ▼
Milvus search(top_k, 
              expr="status == 'active' && document_id != {excluded}",
              output_fields=[id, document_id, chunk_type, content])
  │
  ▼
返回 chunks（全是 active，top_k 精准）
  │
  ▼
BackgroundTasks: update_chunk_stats(hit_chunks, scores)
  │
  ▼
返回给 generator
```

### 6.3 改动点

| 文件 | 改动 |
|---|---|
| `vector_store.py:search_vectors` | 加默认 `filter_expr="status == 'active'"` |
| `retriever.py` | 检索后注册 BackgroundTasks 写统计 |
| `chat.py` / `chat_service.py` | 透传 `background_tasks: BackgroundTasks` 参数到 retriever |

### 6.4 边界情况

1. **冲突检测进行中**：新 chunk 入库时 status 直接是 `active`，检测期间该 chunk 仍可被检索（最多几秒到几分钟窗口，可接受）
2. **pending_review 的 chunk**：按非 active 过滤掉，不参与检索
3. **统计写入失败**：BackgroundTasks 异常不影响检索结果，记日志即可

### 6.5 依赖注入方式

采用参数透传（与项目现有 `documents.py:51-58` 上传文档的模式一致）：

```python
# chat.py
async def chat(request: ChatRequest, background_tasks: BackgroundTasks, ...):
    return chat_service.chat(request, background_tasks=background_tasks)

# chat_service.py
def chat(self, request, background_tasks):
    return self.retriever.search(request.query, background_tasks=background_tasks)

# retriever.py
def search(self, query, background_tasks):
    hits = self.vector_store.search_vectors(...)
    background_tasks.add_task(update_chunk_stats, hits, scores)
    return hits
```

理由：一致性、生命周期清晰、零新基础设施。

## 7. 数据迁移

### 7.1 PG 迁移（在线执行，非破坏性）

```sql
-- document_chunks 表
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

-- documents 表
ALTER TABLE documents ADD COLUMN conflict_check_status VARCHAR(20) DEFAULT 'completed' NOT NULL;
ALTER TABLE documents ADD COLUMN conflict_check_started_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_completed_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN conflict_check_progress VARCHAR(20);
```

### 7.2 Milvus 迁移（策略 A：drop 重建 + 重处理）

采用策略 A，适用于开发环境 / 数据可重新生成。

```python
# 迁移脚本 migrate_milvus.py
vector_store.drop_collection("chunks")
vector_store.create_collection("chunks")  # 新 schema（含 content_hash, status）

# 从 PG 拉所有 COMPLETED 文档，重新走 process_document
for doc in db.query(Document).filter(Document.status == DocumentStatus.COMPLETED):
    db.query(DocumentChunk).filter_by(document_id=doc.id).delete()
    doc_service.process_document(doc.id)
```

`process_document` 需要改造：插入时写 `content_hash` 和 `status='active'`。

### 7.3 部署顺序

```
1. 部署新代码（兼容旧 schema：新字段读写都加默认值兜底）
2. 执行 PG ALTER TABLE（在线，不锁表）
3. 执行 Milvus 迁移脚本（drop + 重建 + 重处理）
4. 重启服务
5. 验证：上传一个测试文档，检查新字段都有值
6. 跑一次冷知识扫描任务，确认无异常
```

### 7.4 不引入 Alembic

项目当前无迁移框架，PG 迁移用 SQL 脚本手动执行。本次改动字段多但都是 ADD COLUMN（非破坏性），手动执行风险可控。后续如需引入 Alembic 另行讨论。

## 8. 配置项

新增到 `backend/app/core/config.py`：

```python
# 冲突检测
CONFLICT_DETECTION_HIGH_CONFIDENCE = 0.85    # 高于此值自动作废
CONFLICT_DETECTION_LOW_CONFIDENCE = 0.5      # 低于此值忽略，介于两者之间转人工

# 冷知识识别
COLD_KNOWLEDGE_TIMEOUT_DAYS = 90             # last_accessed_at 超过此天数归档
COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD = 2        # access_count 低于此值
COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS = 30        # 上传超过此天数才适用频次规则
COLD_KNOWLEDGE_LOW_QUALITY_SCORE = 0.3       # avg_score 低于此值
COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS = 5      # hit_count 达到此值才适用质量规则
COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS = 90   # 归档后保留天数，过期硬删

# 调度
COLD_KNOWLEDGE_SWEEP_CRON = "0 3 * * *"      # 冷知识扫描 cron
HARD_DELETE_SWEEP_CRON = "0 4 * * *"         # 硬删除扫描 cron
```

## 9. 文件改动清单

### 9.1 新增文件

- `backend/app/services/conflict_service.py` - 冲突检测管道
- `backend/app/services/cold_knowledge_service.py` - 冷知识扫描与硬删除
- `backend/app/scheduler.py` - APScheduler 初始化与任务注册
- `backend/scripts/migrate_milvus.py` - Milvus 迁移脚本
- `backend/scripts/migrate_pg.sql` - PG 迁移 SQL

### 9.2 修改文件

- `backend/app/rag/vector_store.py` - `create_collection` 加 content_hash/status 字段；`search_vectors` 加默认过滤；新增 `upsert_status` 方法
- `backend/app/rag/retriever.py` - 检索后注册 BackgroundTasks 写统计；接收 `background_tasks` 参数
- `backend/app/services/document_service.py` - `process_document` 写 content_hash 和 status；上传后触发冲突检测后台任务
- `backend/app/services/chat_service.py` - 透传 `background_tasks`
- `backend/app/api/chat.py` - 注入 `BackgroundTasks`
- `backend/app/api/admin.py` - 扩展 `/admin/documents` 返回 conflict_check_status；新增 `/admin/chunks` 列表与 PATCH 动作
- `backend/app/entities/database.py` - `DocumentChunk` / `Document` 模型加新字段
- `backend/app/entities/schemas.py` - 加 Pydantic 响应模型
- `backend/app/core/config.py` - 加配置项
- `backend/app/core/dependencies.py` - 注册 conflict_service / cold_knowledge_service / scheduler 依赖

## 10. 非目标 / 后续工作

- 字节级相同 chunk 的引用计数（多文档共享同一 chunk 时去重存储）
- 跨文档主题聚类与合并
- 冲突检测的 WebSocket 实时推送（当前用轮询）
- 引入 Alembic 迁移框架
- `rag_memories` collection 的 schema 演进
- 冷知识识别的 ML 模型化（当前是规则阈值，未来可学习每个文档的分布）
