# 知识图谱设计：实体-关系-属性 KG 与三路融合检索

**日期**：2026-08-11
**状态**：待审阅
**范围**：`backend/app/knowledge_graph/`（新增），`backend/app/rag/retriever.py`、`backend/app/rag/splitter.py`、`backend/app/rag/vector_store.py`、`backend/app/llm/providers.py`（改造），`docker-compose.yml`（加 Neo4j 服务）

## 1. 目标

为 RAG 系统增加**真·知识图谱**（实体-关系-属性）作为 HybridRetriever 的第三路检索源，与 dense / BM25 三路融合。

**本期目标**：
- 节点：Law / Article / Concept / Party / Region / Document 六类
- 边：CITES / IS_A / CONFLICTS_WITH / APPLIES_TO / CONTAINS / EXPLAINS 六类
- 抽取管道：规则解析（法名/条款/日期/地域）+ LLM 抽取（Concept/Party/关系）混合
- 演化：增量 + 冲突检测，冲突走人工审核
- 检索：query 实体抽取 + 2 跳 Cypher + RRF 三路融合 + 失败回退
- 可观测性：OTel tracing 全链路；管理后台用 Neo4j Browser + admin API

**生产级标准保留**：冲突检测审核流程、失败回退、严格实体合并、OTel tracing、错误处理与重试。

**不做**（迁移/部署考虑，个人项目不涉及）：
- 渐进式迁移 / 灰度发布 / feature flag
- 向后兼容旧接口
- 现有 chunks 增量回填策略（直接全量重跑）

**Non-goals**（明确不做）：
- 实体 linking 到外部 KG（Wikidata / CN-DBpedia）
- 图可视化前端（用 Neo4j Browser）
- 自动学习新 schema（节点/边类型预定义死）
- KG 失败率监控告警
- 跨语言支持（只中文劳动法）
- 多租户隔离

## 2. 架构总览

### 写入路径（异步 BackgroundTasks）

```
文档上传 -> chunk 入库（已有）
        -> KG 抽取管道（异步触发）
            ├─ 规则解析：法名/条款编号/生效日期/地域
            ├─ LLM 抽取：Concept / Party / 关系
            ├─ 实体合并：embedding + LLM 验证
            └─ 冲突检测 -> active 或 pending_review（人工审核）
```

跟 `chunk-schema-extension` spec 的冲突检测一致走 BackgroundTasks，不引入消息队列。

### 检索路径（同步，加在 HybridRetriever 里）

```
用户 query
   ├─ query 实体抽取（Concept/Party/Region）
   └─ 三路并行检索:
        ├─ dense (Milvus)        ── 已有
        ├─ BM25 (sparse)         ── 已有
        └─ KG (Neo4j): 实体 -> 2跳 Cypher -> 关联 chunks  ── 新增
                                  └─ 失败回退：异常时跳过，不影响另两路
   -> RRF 融合（已有，扩展到三路）
   -> CrossEncoder rerank（已有）
```

### 模块划分（`backend/app/knowledge_graph/`）

| 文件 | 职责 |
|---|---|
| `schema.py` | 节点/边类型枚举、Pydantic 模型、Cypher 标签常量 |
| `extractor.py` | 规则解析 + LLM 抽取管道入口 |
| `rule_parser.py` | 法律文档结构化解析（法名/条款/日期/地域） |
| `llm_extractor.py` | LLM 抽 Concept/Party/关系，few-shot prompt |
| `graph_store.py` | Neo4j 连接、Cypher 读写封装 |
| `entity_resolver.py` | 实体合并：embedding 相似度 + LLM 验证 |
| `conflict_detector.py` | 关系冲突检测 + pending_review 流转 |
| `kg_retriever.py` | KG 检索路径：query 实体抽取 + 多跳查询 |
| `kg_admin.py` | 后台接口：图查询、审核队列 |
| `exceptions.py` | KG 异常类型（用于失败回退） |

### 集成点

- `backend/app/rag/retriever.py` `HybridRetriever.retrieve` 加 KG 路径 + RRF 三路融合
- `backend/main.py` lifespan 初始化 `Neo4jStore` 单例，跟 `MilvusStore` 一致
- 文档上传 API（`backend/app/api/`）触发 BackgroundTasks 调 `extractor.run(document_id)`
- OTel tracer 复用 `app.core.observability.get_tracer("kg")`

## 3. Neo4j 数据模型

### 节点类型（6 种）

| 标签 | 关键属性 | 说明 |
|---|---|---|
| `Law` | `id`, `name`, `level` (法律/法规/规章/司法解释), `effective_date`, `issuer`, `region_id` | 法名，如《劳动合同法》 |
| `Article` | `id`, `law_id`, `article_no`, `content_hash`, `chunk_ids` (list), `status` | 具体法条；`chunk_ids` 关联 Milvus chunks |
| `Concept` | `id`, `name`, `aliases` (list), `embedding` | 法律概念，如"试用期"；embedding 用于 query 实体匹配 |
| `Party` | `id`, `name`, `aliases` (list) | 主体，如"用人单位""劳动者" |
| `Region` | `id`, `name`, `level` (国家/省/市) | 地域 |
| `Document` | `id`, `source_file`, `uploaded_at`, `doc_type` | 上传的原始文档 |

### 边类型（6 种）

| 类型 | from -> to | 属性 |
|---|---|---|
| `CITES` | `Article` -> `Article` | `context`, `confidence` |
| `IS_A` | `Concept` -> `Concept` | `confidence` |
| `CONFLICTS_WITH` | `Article` -> `Article` | `reason`, `confidence`, `status` (pending_review/confirmed/dismissed), `detected_at`, `reviewed_by`, `reviewed_at`, `review_note` |
| `APPLIES_TO` | `Article` -> `Region` / `Party` | `confidence` |
| `CONTAINS` | `Law` -> `Article` / `Document` -> `Article` | - |
| `EXPLAINS` | `Article` -> `Concept` | `confidence` |

### Article ↔ chunks 关联策略

**Article 节点是图上的"指针"，不存原文**。原文在 Milvus chunks 里。

- `Article.chunk_ids: list[str]` 记录关联的 Milvus chunk_id 列表
- Article 解析时记录字符 range `(char_start, char_end)`，与 chunks 的 `char_start` / `char_end` 字段做 overlap 匹配，填入 `chunk_ids`
- KG 检索路径：query 抽实体 -> 多跳查询拿 Article -> 用 `chunk_ids` 反查 Milvus -> 拿到 chunks -> 进 RRF 融合
- 三路检索返回的都是 chunks，同构可直接 RRF；不重复存储原文

### chunks schema 扩展（跨 spec 改动）

`backend/app/rag/splitter.py` 切分时记录每个 chunk 在原文中的字符 offset：

```
char_start   INT
char_end     INT
```

`ThreeLayerSplitter.split` 输出的每个 chunk dict 加 `char_start` / `char_end`。三层切分时父层 offset 累加传给子层。

Milvus `rag_chunks` collection 加两个标量字段（不参与检索过滤，只为 KG 关联用）：

```
char_start   INT64
char_end     INT64
```

PostgreSQL `document_chunks` 表加两个 INT 列（镜像 Milvus，跟 chunk-schema-extension 的方案 B 一致）。

**对 `chunk-schema-extension` spec 的影响**：该 spec 已经在改 chunk schema，把这两个字段一并加进去。本期 KG spec 依赖此字段就位。

### 约束与索引

```cypher
CREATE CONSTRAINT law_unique IF NOT EXISTS
  FOR (n:Law) REQUIRE (n.name, n.region_id) IS UNIQUE;
CREATE CONSTRAINT article_unique IF NOT EXISTS
  FOR (n:Article) REQUIRE (n.law_id, n.article_no) IS UNIQUE;
CREATE CONSTRAINT concept_unique IF NOT EXISTS
  FOR (n:Concept) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT party_unique IF NOT EXISTS
  FOR (n:Party) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT region_unique IF NOT EXISTS
  FOR (n:Region) REQUIRE n.name IS UNIQUE;

CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
  FOR (n:Concept) ON (n.embedding)
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};
```

启动时 `Neo4jStore._init_constraints()` 跑上述语句（幂等）。

### 示例子图

```
(Law:劳动合同法)-[:CONTAINS]->(Article:第19条)-[:EXPLAINS]->(Concept:试用期)
                                       |
                                       ├─[:APPLIES_TO]->(Party:劳动者)
                                       ├─[:APPLIES_TO]->(Party:用人单位)
                                       └─[:CITES]->(Article:劳动法第21条)-[:EXPLAINS]->(Concept:试用期)

(Concept:试用期)-[:IS_A]->(Concept:合同期限)
```

## 4. 抽取管道

### 流程

```
document_id 输入
  ├─ 1. 加载 chunks（从 PG/Milvus）+ 文档全文
  ├─ 2. rule_parser.py（规则解析，全文档一次）
  │     ├─ 法名/层级/生效日期/发文机关/地域 -> Law 节点
  │     ├─ 第X条正则切分 -> Article 节点列表 + chunk_ids 关联
  │     └─ Document 节点
  ├─ 3. llm_extractor.py（per chunk，并发）
  │     └─ LLM 抽 Concept/Party/关系，输出 JSON
  ├─ 4. entity_resolver.py
  │     └─ 三级合并：精确 name -> alias -> embedding+LLM 验证
  ├─ 5. conflict_detector.py
  │     └─ 同 Concept 已有 Articles 的矛盾判定 -> CONFLICTS_WITH 边
  └─ 6. graph_store.py
        └─ MERGE 节点 + MERGE 关系 + OTel span
```

### 规则解析边界（`rule_parser.py`）

**走规则**：法名（《...法》《...条例》正则）、条款编号（`第[一二三四...百千\d]+条`）、生效日期（"自X年X月X日起施行"）、发文机关、地域（法名含地域名 or 文末适用范围段）。

**不走规则**：概念、主体、引用关系、冲突关系 - 全部交给 LLM。

**Article.chunk_ids 填法**：
- 规则解析器在文档全文中找 Article 的字符 range `(article_start, article_end)`
- 查 Milvus / PG：`SELECT chunk_id WHERE document_id = ? AND char_start < article_end AND char_end > article_start`
- 字符 range overlap 是精确匹配，不依赖内容相等

### LLM 抽取（`llm_extractor.py`）

Per chunk 调一次 LLM。输出 JSON：

```json
{
  "entities": [
    {"type": "Concept", "name": "试用期", "aliases": ["试用期间"]},
    {"type": "Party", "name": "用人单位"}
  ],
  "relations": [
    {"type": "EXPLAINS", "from": "article:劳动合同法:19", "to": "concept:试用期", "confidence": 0.9},
    {"type": "APPLIES_TO", "from": "article:劳动合同法:19", "to": "party:劳动者", "confidence": 0.95}
  ]
}
```

`from`/`to` 用 `type:name` 引用，由 `entity_resolver` 阶段解析成 Neo4j 节点 ID。LLM 不需要知道图上现有的 ID，只描述语义关系。

**LLM 调用**：复用 `app.llm.providers.invoke_llm_threadsafe`，新增 `get_extraction_llm()` provider。抽取对输出 JSON 格式要求严，prompt 跟评估不同，独立 provider 便于调优。

### 实体合并三级策略（`entity_resolver.py`）

| 级别 | 触发条件 | 动作 |
|---|---|---|
| 1 精确 | name + type 完全一致 | 直接合并，新 alias 加到已有节点 |
| 2 alias | 新 name 在已有节点 aliases 里 | 合并到已有节点 |
| 3 模糊 | name embedding cosine > 0.92 | LLM 二次确认（"试用期" vs "试用期间" 是否同一概念）-> 合并或新建 |

合并时不修改已有节点的 name（首次写入者占主导），记录 `source_chunk_ids` 作为来源。

### OTel 集成

每步开 span：
- `kg.extract.rule_parse`
- `kg.extract.llm_extract`（per chunk）
- `kg.extract.entity_resolve`
- `kg.extract.conflict_detect`（详见 Section 6）
- `kg.extract.graph_write`

属性记录：`document_id`、`chunk_count`、`entities_extracted`、`relations_extracted`、`merge_count`、`conflict_count`、`duration_ms`。

## 5. 检索集成

### KG 检索路径（`kg_retriever.py`）

```
query 输入
  ├─ 1. query 实体抽取
  │     ├─ Concept: query embedding -> Neo4j vector index -> top-K (cosine > 0.7)
  │     └─ Party/Region: 词表精确匹配（O(1)）
  ├─ 2. 多跳 Cypher 查询（2 跳）
  │     └─ Concept <-[:EXPLAINS|IS_A*1..2]- Article
  ├─ 3. 收集 Article 节点 + concept_hit_count
  ├─ 4. 反查 chunks：Article.chunk_ids -> Milvus 拿 chunks
  └─ 5. 返回 chunks 列表（带 kg_score）
```

### Query 实体抽取：零 LLM 调用

**Concept**：query embedding 在 Neo4j Concept 向量索引上找 top-K（cosine > `KG_CONCEPT_SIMILARITY_THRESHOLD`）。复用 dense 路径已经算过的 query embedding，不重复计算。

**Party/Region**：维护常用词表（"用人单位""劳动者""工会""北京""上海"…），query 里直接字符串匹配。词表扩容时改代码（不做 admin 接口）。

**为什么不用 LLM 抽 query 实体**：检索是热路径，每次问答都调 LLM 拖 P99。Concept 走 embedding、Party/Region 走词表，零 LLM 调用。

### 多跳 Cypher 查询

```cypher
MATCH (c:Concept)-[:EXPLAINS|IS_A*1..2]-(a:Article)
WHERE c.id IN $concept_ids
  AND a.status = 'active'
RETURN DISTINCT a,
       collect(c.name) AS matched_concepts,
       count(DISTINCT c) AS concept_hit_count
ORDER BY concept_hit_count DESC
LIMIT 20
```

返回 Article + 命中 Concept 数。命中数作为 `kg_score` 基础。

### kg_score 计算

```python
kg_score = concept_hit_count / max_concept_hit_count  # 归一化到 [0, 1]
```

路径长度衰减（1 跳 1.0 / 2 跳 0.5）YAGNI，先不做。

### RRF 三路融合（改 `HybridRetriever.retrieve`）

```python
# 当前（两路）
dense_results = self.vector_store.search(...)
sparse_results = self.bm25_search(...)
merged = rrf_fuse([dense_results, sparse_results])

# 改造后（三路）
dense_results = self.vector_store.search(...)
sparse_results = self.bm25_search(...)
if self.kg_enabled:
    try:
        kg_results = self.kg_retriever.retrieve(query, query_embedding)
    except KGError as e:
        span.record_exception(e)
        kg_results = []
else:
    kg_results = []
merged = rrf_fuse([dense_results, sparse_results, kg_results])
```

RRF 公式不变：`score = sum(1 / (k + rank))` 跨三路。三路等权（1:1:1）。`rerank_score` 字段仍由 CrossEncoder rerank 阶段写入，RRF 阶段写 `kg_score` 作为辅助。

### 失败回退

- `kg_retriever` 内部所有 Neo4j 调用 try/except
- 异常时记 OTel span 错误，返回空列表
- HybridRetriever 主流程不感知，RRF 自然降权（少一路）
- 不做"KG 失败率告警"（监控是后续事）

### OTel 集成

```
kg.retrieve (父 span)
  ├─ kg.retrieve.entity_extract
  ├─ kg.retrieve.cypher_query
  └─ kg.retrieve.chunk_fetch
```

属性：`query`、`concept_count`、`article_count`、`chunk_count`、`kg_score_max`、`duration_ms`、`error`（如有）。

## 6. 冲突检测 + 审核流程

### 冲突检测场景

**只检测一种**：新 Article 抽取时发现 `A -[:EXPLAINS]-> Concept C`，查 Neo4j 已有 `A' -[:EXPLAINS]-> C`，对每对 `(A, A')` 让 LLM 判内容是否矛盾。

```
新 Article A 入库
  ├─ A -[:EXPLAINS]-> C
  ├─ 查 Neo4j: MATCH (a':Article)-[:EXPLAINS]->(C) WHERE a'.id <> A.id
  ├─ 对每个 A':
  │   └─ LLM 判定 (A.content, A'.content) 是否矛盾
  │       ├─ 矛盾 -> 写 A -[:CONFLICTS_WITH {status: pending_review, ...}]-> A'
  │       └─ 不矛盾 -> 跳过
  └─ 不阻塞主流程，异步触发
```

**矛盾判定 prompt 示例**：

```
你是劳动法冲突检测专家。判断两条法条对同一概念的规定是否矛盾。

概念：{concept_name}
法条A（{law_a_name}第{article_a_no}条）：{article_a_content}
法条B（{law_b_name}第{article_b_no}条）：{article_b_content}

判断：
- 矛盾：两条对同一概念给出互斥规定（如"最长6个月" vs "最长3个月"）
- 互补：两条从不同角度规定，不互斥（如"试用期包含在合同期内" vs "试用期最长6个月"）
- 不相关：虽然都讲同一概念但内容无交集

输出 JSON: {"is_conflict": bool, "reason": "...", "confidence": 0.0-1.0}
```

**不检测的场景**（明确排除）：
- `CITES`：引用是事实关系，无矛盾概念
- `IS_A`：上下位是事实关系
- `APPLIES_TO`：适用关系（除非同一 Article 适用矛盾地域，属于数据错误）
- `CONTAINS`：包含关系

### `CONFLICTS_WITH` 边状态机

```
pending_review (初始)
   ├─ confirmed   -> 旧 Article 节点 status 改 superseded（与 chunk-schema-extension 对齐）
   └─ dismissed   -> 边保留 status=dismissed（审计用），Articles 不变
```

边属性：
- `status`: pending_review / confirmed / dismissed
- `reason`: LLM 判冲突的理由
- `confidence`: LLM 置信度
- `detected_at`: 检测时间
- `reviewed_by` / `reviewed_at` / `review_note`: 审核元数据

### 审核接口（`kg_admin.py`）

```
GET  /api/v1/admin/kg/conflicts?status=pending_review
     -> 列出 pending 边，含两条 Article 全文

POST /api/v1/admin/kg/conflicts/{edge_id}/confirm
     body: { review_note: "..." }
     -> 边 status=confirmed；旧 Article status=superseded

POST /api/v1/admin/kg/conflicts/{edge_id}/dismiss
     body: { review_note: "..." }
     -> 边 status=dismissed

GET  /api/v1/admin/kg/graph?concept_id=...
     -> 返回 Concept 周围子图（用于可视化/调试）

GET  /api/v1/admin/kg/stats
     -> 节点/边/pending 数统计
```

### 后台可视化

**不写专门前端**：
- 看图用 Neo4j Browser（`http://localhost:7474`），直接跑 Cypher
- 审核用 API + curl/Postman
- 简易 stats 页面可选（FastAPI 直接返回 HTML，挂 `/admin/kg/stats.html`），不做也行

### 与 chunk-schema-extension 审核流程的关系

两套审核**互相独立**：

| 维度 | chunk 冲突（chunk-schema-extension） | Article 冲突（KG） |
|---|---|---|
| 粒度 | chunk ↔ chunk | Article ↔ Article |
| 触发 | 新 chunk 与旧 chunk 语义冲突 | 新 Article 与旧 Article 解释同 Concept 矛盾 |
| 接口 | `/api/v1/admin/chunks/conflicts` | `/api/v1/admin/kg/conflicts` |
| 状态字段位置 | chunk 节点（PG + Milvus status） | CONFLICTS_WITH 边（Neo4j） |

两者可能同时发生（chunk 级冲突 + Article 级冲突），但分别处理。

## 7. 部署、配置、错误处理

### Neo4j 部署（`docker-compose.yml`）

```yaml
neo4j:
  image: neo4j:5.20
  ports:
    - "7474:7474"  # Browser UI
    - "7687:7687"  # Bolt protocol
  environment:
    - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    - NEO4J_PLUGINS=["apoc"]
  volumes:
    - neo4j_data:/data
```

跟现有 milvus / postgres / redis 并列，docker-compose 加一个 service。

### Neo4j 连接管理（`graph_store.py`）

跟 `MilvusStore` 一致的模式：
- 启动时 `main.py` lifespan 初始化 `Neo4jStore` 单例
- 启动时跑 `_init_constraints()` 建约束 + vector index（Section 3 的 Cypher，幂等）
- Lifespan shutdown 时 `driver.close()`
- 全局 `get_graph_store()` 拿单例

```python
from neo4j import GraphDatabase

class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_constraints()

    def _init_constraints(self) -> None:
        # 跑 Section 3 的 CONSTRAINT 和 VECTOR INDEX 语句（幂等）
        ...

    def session(self):
        return self.driver.session()

    def close(self) -> None:
        self.driver.close()
```

### 配置（`.env`）

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
KG_ENABLED=true                       # 总开关，false 时 retriever 跳过 KG 路径
KG_CONCEPT_SIMILARITY_THRESHOLD=0.7   # query 实体匹配
KG_MULTI_HOP_DEPTH=2
```

### 异常类型（`exceptions.py`）

```python
class KGError(Exception): ...
class KGConnectionError(KGError): ...   # Neo4j 连接失败
class KGExtractionError(KGError): ...   # 抽取管道失败
class KGQueryError(KGError): ...        # Cypher 查询失败
```

`kg_retriever` 内部捕获 `KGError`，返回空列表，HybridRetriever 自动降级到两路。

### OTel tracer 命名汇总

| Span | 来源 |
|---|---|
| `kg.extract.rule_parse` | Section 4 |
| `kg.extract.llm_extract` | Section 4 |
| `kg.extract.entity_resolve` | Section 4 |
| `kg.extract.conflict_detect` | Section 6 |
| `kg.extract.graph_write` | Section 4 |
| `kg.retrieve`（父） | Section 5 |
| `kg.retrieve.entity_extract` | Section 5 |
| `kg.retrieve.cypher_query` | Section 5 |
| `kg.retrieve.chunk_fetch` | Section 5 |

tracer 用 `get_tracer("kg")`，跟现有 `get_tracer("rag.tools")` / `get_tracer("rag.retrieve")` 一致。

## 8. 测试策略

### 单元测试（`backend/tests/test_knowledge_graph/`，mock LLM / Neo4j）

- `test_rule_parser.py`：法名/条款/日期/地域解析
- `test_llm_extractor.py`：mock LLM 返回，测 JSON 解析
- `test_entity_resolver.py`：三级合并
- `test_conflict_detector.py`：mock LLM 判矛盾 + 状态机

### 集成测试（testcontainers Neo4j）

- `test_graph_store.py`：跑真 Neo4j 容器，测 Cypher 读写 + 约束
- `test_extraction_pipeline.py`：mock LLM + 真 Neo4j，跑完整抽取
- `test_kg_retriever_integration.py`：mock LLM + 真 Neo4j + 真 Milvus，测三路融合 + 失败回退

### E2E

用《劳动合同法》第19-21条跑完整流程：上传 -> 抽取 -> 入图 -> 检索 -> 验证返回 chunks。

## 9. 现有文档回填

CLI 脚本（不做 API）：

```bash
python -m app.knowledge_graph.backfill --all-documents
python -m app.knowledge_graph.backfill --document-id 1
```

遍历现有 documents 跑抽取管道。个人项目全量重跑即可。

## 10. 性能预算

KG 路径 P99 增量：
- query 实体抽取（Concept vector 检索）：10-30ms
- 2 跳 Cypher：20-50ms
- 反查 Milvus chunks：10-20ms
- 总计：40-100ms

跟 dense + BM25 并行，理论 P99 增量是 `max(三路) - max(两路)`。失败回退时不增加延迟。

## 11. 跨 spec 依赖

- **`chunk-schema-extension` spec**：需要 chunks 加 `char_start` / `char_end` 字段。该 spec 已经在改 chunk schema，把这两个字段一并加进去。本期 KG spec 依赖此字段就位。
- **`opentelemetry-integration` spec**：复用 `get_tracer` 和 `invoke_llm_threadsafe` 的 tracing 基础设施。

## 12. 实施顺序

1. **基础设施**：Neo4j docker-compose、`Neo4jStore` 单例、`_init_constraints()`、`.env` 配置
2. **chunks schema 扩展**：splitter 加 offset、Milvus + PG schema 加字段、回填现有 chunks
3. **数据模型**：`schema.py` Pydantic 模型、节点/边类型枚举
4. **抽取管道**：`rule_parser.py` -> `llm_extractor.py` -> `entity_resolver.py` -> `graph_store.py` 写入
5. **冲突检测**：`conflict_detector.py` + `kg_admin.py` 审核接口
6. **检索集成**：`kg_retriever.py` + `HybridRetriever` 改造三路融合
7. **OTel + 错误处理**：全链路 span、异常类型、失败回退
8. **测试**：单元 + 集成 + E2E
9. **回填脚本**：`backfill.py` 跑全量文档
