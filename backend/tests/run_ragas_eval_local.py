"""RAGAS 评估脚本 - 在 Docker 容器内运行"""
import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, '/app')
os.chdir('/app')

# 猴子补丁：绕过 ragas 对 vertexai 的硬导入
import tests.ragas_patch

from dotenv import load_dotenv
load_dotenv('/app/.env')

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.rag.retriever import HybridRetriever, SparseRetriever
from app.db.session import SessionLocal
from app.services.chat_service import ChatService
from tests.testset_labor_law_1994 import get_testset


def init_services():
    settings = get_settings()
    print(f"Model: {settings.CHAT_MODEL}")
    print(f"API Base: {settings.CHAT_API_BASE}")
    print(f"Reranker: {settings.RERANKER_ENABLED}")

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

    db = SessionLocal()
    chat_service = ChatService(db, retriever)
    return chat_service, db


async def call_rag(chat_service, question):
    """调用 RAG 系统，返回 answer 和 contexts"""
    result = await chat_service.agentic_chat(
        user_id=1, query=question, max_attempts=2
    )
    answer = result.get("answer", "")
    chunks = result.get("retrieved_chunks", [])
    contexts = [c.get("content", "") for c in chunks if c.get("content")]
    agentic = result.get("agentic_info", {})
    return {
        "answer": answer,
        "contexts": contexts,
        "grade": agentic.get("evaluation_grade", "unknown"),
        "confidence": agentic.get("confidence", 0),
        "attempts": agentic.get("attempt_count", 0),
    }


async def run_all(chat_service, testset):
    """运行所有测试题"""
    results = []
    for i, tc in enumerate(testset, 1):
        q = tc["question"]
        print(f"[{i}/{len(testset)}] {q}")
        try:
            r = await call_rag(chat_service, q)
            r["question"] = q
            r["expected_answer"] = tc.get("expected_answer", "")
            r["reference"] = tc.get("reference", "")
            r["error"] = None
            print(f"  -> chunks={len(r['contexts'])}, grade={r['grade']}, conf={r['confidence']:.2f}")
        except Exception as e:
            r = {
                "question": q,
                "answer": "",
                "contexts": [],
                "grade": "error",
                "confidence": 0,
                "attempts": 0,
                "expected_answer": tc.get("expected_answer", ""),
                "reference": tc.get("reference", ""),
                "error": str(e),
            }
            print(f"  -> ERROR: {e}")
        results.append(r)
    return results


def run_ragas_eval(results):
    """用 RAGAS 评估"""
    import tests.ragas_patch

    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    # 使用 langchain ChatOpenAI（不会发送 response_format: json_object，兼容豆包）
    from langchain_openai import ChatOpenAI as LangchainChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

    settings = get_settings()

    # LLM：用 langchain ChatOpenAI 包装，避免 json_object 问题
    lc_llm = LangchainChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.CHAT_API_KEY,
        base_url=settings.CHAT_API_BASE,
        temperature=0,
        max_tokens=8192,
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # Embeddings：用本地 bge-m3，避免豆包 API 兼容问题
    lc_embeddings = LCHuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    samples = []
    for r in results:
        if r.get("error") or not r.get("answer"):
            continue
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r.get("expected_answer", ""),
        )
        samples.append(sample)

    print(f"\nRAGAS 评估 {len(samples)} 个有效样本...")

    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # 临时禁用 LangSmith 追踪避免干扰
    old_tracing = os.environ.get("LANGSMITH_TRACING")
    os.environ["LANGSMITH_TRACING"] = "false"

    try:
        eval_result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm)
    finally:
        if old_tracing is not None:
            os.environ["LANGSMITH_TRACING"] = old_tracing
        else:
            os.environ.pop("LANGSMITH_TRACING", None)

    return eval_result


def main():
    print("=" * 60)
    print("劳动法(1994) RAGAS 评估")
    print("=" * 60)

    testset = get_testset()
    print(f"测试集: {len(testset)} 道题\n")

    chat_service, db = init_services()

    # 1. 运行 RAG
    print("\n--- 运行 RAG 检索与生成 ---")
    results = asyncio.run(run_all(chat_service, testset))

    # 2. 简单统计
    total = len(results)
    no_answer = sum(1 for r in results if "未找到" in r.get("answer", "") or not r.get("answer"))
    correct = sum(1 for r in results if r.get("grade") == "correct")
    has_ctx = sum(1 for r in results if len(r.get("contexts", [])) > 0)
    avg_conf = sum(r.get("confidence", 0) for r in results) / total if total else 0

    print(f"\n--- 基础统计 ---")
    print(f"总数: {total}, 有检索结果: {has_ctx}, CRAG correct: {correct}, 未找到答案: {no_answer}, 平均置信度: {avg_conf:.2f}")

    # 3. RAGAS 评估
    try:
        eval_result = run_ragas_eval(results)
        print(f"\n--- RAGAS 评估结果 ---")
        # EvaluationResult 可能是 dict 或有 _scores_dict 属性
        import math
        if hasattr(eval_result, 'items'):
            ragas_scores = {k: round(v, 4) for k, v in eval_result.items()}
        elif hasattr(eval_result, '_scores_dict'):
            ragas_scores = {}
            for k, v in eval_result._scores_dict.items():
                if v:
                    # v 是一个列表，取所有有效值（非 None、非 NaN）的平均
                    valid_scores = []
                    for s in v:
                        if s is None:
                            continue
                        try:
                            f = float(s)
                            if not math.isnan(f):
                                valid_scores.append(f)
                        except (ValueError, TypeError):
                            continue
                    if valid_scores:
                        ragas_scores[k] = round(sum(valid_scores) / len(valid_scores), 4)
        else:
            ragas_scores = {}
        for k, v in ragas_scores.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nRAGAS 评估失败: {e}")
        ragas_scores = {}

    # 4. 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "basic_stats": {
            "total": total,
            "has_context": has_ctx,
            "correct": correct,
            "no_answer": no_answer,
            "avg_confidence": round(avg_conf, 2),
        },
        "ragas_scores": ragas_scores,
        "results": results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/app/tests/ragas_eval_result_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")

    db.close()


if __name__ == "__main__":
    main()
