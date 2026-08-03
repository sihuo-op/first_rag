"""
RAG Search MCP Server — 将 RAGGraph 封装为 MCP 工具

对外暴露两个工具：
1. search_knowledge_base — 只检索，返回文档片段
2. ask_knowledge_base — 完整问答，返回最终答案

RAGGraph 内部逻辑不动，只在外面包壳。
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from dotenv import dotenv_values

# ===== 设置临时目录（jieba 缓存需要） =====
JIEBA_CACHE_DIR = Path(__file__).parent / "data" / "jieba_cache"
JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMPDIR"] = str(JIEBA_CACHE_DIR)

# ===== 加载配置 =====
PROJECT_ROOT = Path(__file__).parent
ENV_FILE = PROJECT_ROOT / ".env"

for k, v in dotenv_values(ENV_FILE).items():
    if k not in os.environ:
        os.environ[k] = v

# 将项目根目录加入 sys.path，确保能 import app 包
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

# ===== 延迟初始化组件 =====

_initialized = False
_retriever = None
_rag_graph = None
_generation_llm = None
_rewrite_llm = None


def _init_components():
    """延迟初始化所有组件（首次调用工具时才连接数据库、加载模型）"""
    global _initialized, _retriever, _generation_llm, _rewrite_llm

    if _initialized:
        return

    from app.core.config import get_settings
    from app.rag.vector_store import MilvusStore
    from app.rag.retriever import HybridRetriever, SparseRetriever, Reranker
    from app.agent.tools import get_generation_llm, get_rewrite_llm

    settings = get_settings()

    print("[MCP Server] 初始化组件...")

    # 1. 向量存储
    vector_store = MilvusStore(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        embedding_model=settings.SENTENCE_TRANSFORMER_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
    )

    # 2. 稀疏检索器 + 加载 BM25 索引
    sparse_retriever = _load_bm25_index(settings)

    # 3. 混合检索器
    _retriever = HybridRetriever(
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        use_reranker=settings.RERANKER_ENABLED,
        reranker_model=settings.RERANKER_MODEL,
        top_n=settings.RERANKER_TOP_N,
    )

    # 4. 预加载 Reranker
    if settings.RERANKER_ENABLED:
        print(f"[MCP Server] 预加载 Reranker: {settings.RERANKER_MODEL}")
        reranker = Reranker(model_name=settings.RERANKER_MODEL)
        reranker._load_model()

    # 5. LLM
    _generation_llm = get_generation_llm()
    _rewrite_llm = get_rewrite_llm()

    _initialized = True
    print("[MCP Server] 初始化完成")


def _load_bm25_index(settings):
    """从数据库加载 BM25 索引（复用 main.py 中的逻辑）"""
    from app.rag.retriever import SparseRetriever

    sparse_retriever = SparseRetriever()

    try:
        from app.db.session import SessionLocal
        from app.entities.database import DocumentChunk, DocumentStatus, Document

        db = SessionLocal()
        try:
            chunks = db.query(DocumentChunk).join(Document).filter(
                Document.status == DocumentStatus.COMPLETED
            ).all()

            if chunks:
                contents = [chunk.content for chunk in chunks]
                metadata_list = [
                    {"document_id": chunk.document_id, "chunk_type": chunk.chunk_type}
                    for chunk in chunks
                ]
                sparse_retriever.add_documents(contents, metadata_list)
                print(f"[MCP Server] BM25 索引加载完成: {len(chunks)} 个文档片段")
            else:
                print("[MCP Server] BM25 索引: 数据库中无文档")
        finally:
            db.close()
    except Exception as e:
        print(f"[MCP Server] BM25 索引加载失败（仅向量检索可用）: {e}")

    return sparse_retriever


def _get_rag_graph():
    """获取 RAGGraph 实例（延迟创建）"""
    global _rag_graph

    _init_components()

    if _rag_graph is None:
        from app.agent.graph import RAGGraph

        _rag_graph = RAGGraph(
            retriever=_retriever,
            generation_llm=_generation_llm,
            rewrite_llm=_rewrite_llm,
            evaluation_llm=_generation_llm,  # 评估用主LLM
            max_attempts=2,
            top_k=8,
        )
        print("[MCP Server] RAGGraph 创建完成")

    return _rag_graph


# ===== 创建 MCP Server =====

mcp = FastMCP(
    "rag-search-server",
    instructions="劳动法知识库检索与问答服务。支持两种使用方式："
                 "1. search_knowledge_base 只检索文档片段，适合需要自行分析的场景；"
                 "2. ask_knowledge_base 完整问答流程，自动检索+评估+改写+生成答案。"
)


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 8) -> str:
    """从劳动法知识库中检索相关文档片段

    只返回检索到的文档片段，不生成答案。
    适合需要查看原文、自行分析的Agent使用。

    Args:
        query: 搜索查询文本，如"劳动合同终止的情形"
        top_k: 返回结果数量，默认8
    """
    _init_components()

    try:
        chunks, _ = _retriever.retrieve(query=query, top_k=top_k)
    except Exception as e:
        return json.dumps({"error": f"检索失败: {str(e)}"}, ensure_ascii=False)

    if not chunks:
        return json.dumps({"documents": [], "message": "未检索到相关内容"}, ensure_ascii=False)

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "index": i + 1,
            "content": chunk.get("content", ""),
            "score": round(chunk.get("rrf_score", 0) or chunk.get("rerank_score", 0) or 0, 4),
            "chunk_type": chunk.get("chunk_type", "unknown"),
            "document_id": chunk.get("document_id"),
        })

    return json.dumps({
        "documents": results,
        "query": query,
        "count": len(results),
        "message": f"检索到 {len(results)} 个相关文档片段"
    }, ensure_ascii=False)


@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """向劳动法知识库提问，返回完整答案

    自动经过：检索 → 评估 → 改写 → 生成 的完整 Agentic RAG 流程。
    适合只想得到最终答案的Agent使用。

    Args:
        question: 用户的问题，如"劳动合同到期不续签有补偿吗"
    """
    rag_graph = _get_rag_graph()

    try:
        result = rag_graph.run(question)
    except Exception as e:
        return json.dumps({"error": f"问答失败: {str(e)}"}, ensure_ascii=False)

    # 只返回核心信息，不返回内部调试细节（太大了）
    return json.dumps({
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0),
        "attempt_count": result.get("attempt_count", 0),
        "query_history": result.get("query_history", []),
        "documents_count": len(result.get("documents", [])),
    }, ensure_ascii=False)


# ===== 资源和提示（可选，增值功能） =====

@mcp.resource("rag://stats")
def get_kb_stats() -> str:
    """获取知识库统计信息"""
    _init_components()
    from app.core.config import get_settings
    settings = get_settings()
    return json.dumps({
        "service": "劳动法知识库",
        "retriever": "混合检索（Milvus向量 + BM25稀疏）",
        "reranker": f"CrossEncoder({settings.RERANKER_MODEL})" if settings.RERANKER_ENABLED else "未启用",
        "top_n": settings.RERANKER_TOP_N,
    }, ensure_ascii=False)


@mcp.prompt()
def labor_law_qa(question: str) -> str:
    """劳动法知识问答提示模板

    Args:
        question: 用户的问题
    """
    return f"""你是一个专业的劳动法知识助手。用户提出了以下问题：

{question}

请先使用 search_knowledge_base 工具检索相关法律条款，
然后基于检索结果给出准确、有条理的回答。
如果检索结果不够充分，可以使用 ask_knowledge_base 工具获取完整答案。"""


if __name__ == "__main__":
    print("[MCP Server] rag-search-server 启动中...")
    mcp.run()  # 默认 stdio 模式