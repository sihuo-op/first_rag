# 在所有导入之前设置临时目录（用于 jieba 等库的缓存）
import os
from pathlib import Path

JIEBA_CACHE_DIR = Path(__file__).parent / "data" / "jieba_cache"
JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMPDIR"] = str(JIEBA_CACHE_DIR)

import asyncio
import time

from app.api import admin, auth, chat, documents
from app.core.config import get_settings
from app.db.init_db import init_db
from app.entities.schemas import HealthResponse
from app.knowledge_graph import kg_admin  # KG 审核/后台 API
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    print(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)

if settings.KG_ENABLED:
    app.include_router(kg_admin.router)

# OTel 初始化（必须在所有 middleware 和 router 注册之后，使 OTel 成为最外层 middleware）
from app.core.observability import instrument_app, setup_otel
from app.db.session import engine as db_engine

setup_otel(settings)
instrument_app(app, engine=db_engine)


@app.on_event("startup")
async def startup_event():
    """
    应用启动初始化（精简版，避免阻塞）
    """
    print("=" * 50)
    print("Starting application...")
    print("=" * 50)

    # 1. 初始化数据库
    print("[1/5] Initializing database...")
    init_db()
    print("Database initialized")

    # 2. 预加载 MilvusStore（包含 embedding 模型）
    print("[2/5] Preloading MilvusStore...")
    from app.core.dependencies import get_vector_store
    vector_store = get_vector_store()
    print(f"MilvusStore loaded: embedding_model={vector_store.embedding_model}, dimension={vector_store.dimension}")

    # 3. 初始化检索器（异步加载 BM25 索引）
    print("[3/5] Initializing retriever...")
    from app.core.dependencies import get_retriever
    retriever = get_retriever()
    print("Retriever initialized")

    # 后台加载 BM25 索引（不阻塞启动）
    asyncio.create_task(_load_bm25_index(retriever))

    # 预加载 Reranker 模型
    if settings.RERANKER_ENABLED:
        print(f"Preloading reranker model: {settings.RERANKER_MODEL}")
        from app.rag.retriever import Reranker
        reranker = Reranker(model_name=settings.RERANKER_MODEL)
        reranker._load_model()
        print("Reranker model preloaded")

    # 首次推理预热（后台）：模型加载 ≠ 推理就绪，torch/ONNX 首次前向另有秒级开销
    asyncio.create_task(_warmup_inference())

    # 4. 启动定时任务调度器
    print("[4/5] Starting scheduler...")
    from app.core.scheduler import setup_scheduler
    setup_scheduler()

    # 5. 初始化 Neo4jStore（KG 检索路径）
    # Neo4j 不可用时降级继续（get_retriever 查询期同样会捕获并降级），
    # 绝不因 KG 基础设施故障阻止应用启动 —— 与 KG 全局"失败回退"约束一致。
    if settings.KG_ENABLED:
        print("[5/5] Initializing Neo4jStore...")
        from app.knowledge_graph.graph_store import get_graph_store
        try:
            get_graph_store()
            print("Neo4jStore initialized")
        except Exception as exc:
            print(f"Neo4jStore unavailable, KG disabled for this run: {exc}")

    print("=" * 50)
    print("Application startup complete!")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    from app.core.scheduler import shutdown_scheduler
    shutdown_scheduler()
    print("Scheduler shut down")

    if settings.KG_ENABLED:
        from app.knowledge_graph.graph_store import reset_graph_store
        reset_graph_store()
        print("Neo4jStore closed")


async def _warmup_inference():
    """后台预热首次推理路径，消除首请求 ~4s 的隐性 warm-up 延迟。

    覆盖四类一次性开销：jieba 词典初始化、embedding 首次前向、
    reranker ONNX 首次前向（arena 分配）、LLM HTTPS 连接建立。
    任何一步失败都不影响应用功能。
    """
    def _run() -> None:
        try:
            import time as _time
            t0 = _time.time()

            import jieba
            jieba.initialize()

            from app.core.dependencies import get_retriever, get_vector_store
            get_vector_store().embed_query("劳动合同 试用期 预热")

            retriever = get_retriever()
            if getattr(retriever, "reranker", None):
                retriever.reranker.rerank("预热", [{"content": "劳动合同的试用期规定"}], 1)

            from app.llm.providers import get_rewrite_llm
            from langchain_core.messages import HumanMessage
            get_rewrite_llm().invoke([HumanMessage(content="ping")])

            print(f"Inference warmup done in {round(_time.time() - t0, 1)}s")
        except Exception as e:
            print(f"Inference warmup skipped (non-fatal): {e}")

    try:
        await asyncio.to_thread(_run)
    except Exception as e:
        print(f"Inference warmup failed (non-fatal): {e}")


async def _load_bm25_index(retriever):
    """后台加载 BM25 索引"""
    try:
        print("Loading existing documents into BM25 index (background)...")
        from app.db.session import SessionLocal
        from app.entities.database import Document, DocumentChunk, DocumentStatus
        db = SessionLocal()
        try:
            chunks = db.query(DocumentChunk).join(Document).filter(
                Document.status == DocumentStatus.COMPLETED
            ).all()

            if chunks:
                contents = [chunk.content for chunk in chunks]
                metadata_list = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_type": chunk.chunk_type.value,
                        "db_chunk_id": chunk.id
                    }
                    for chunk in chunks
                ]
                retriever.add_to_sparse_index(contents, metadata_list)
                print(f"Loaded {len(chunks)} chunks into BM25 index")
            else:
                print("No existing documents found for BM25 index")
        finally:
            db.close()
    except Exception as e:
        print(f"Error loading BM25 index: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        milvus_connected=True,
        database_connected=True
    )


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "0.1.0",
        "docs": "/docs"
    }
