"""L3 端到端层测试 - 完整 RAG 流程评估"""
import sys
import os
import json
import asyncio
import math
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(Path(__file__).parent.parent.parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.rag.retriever import HybridRetriever, SparseRetriever
from app.db.session import SessionLocal
from app.services.chat_service import ChatService
from tests.testsets.labor_law_1994 import get_testset
from tests.metrics.custom_metrics import (
    article_reference_rate, no_answer_rate, hallucination_indicator,
    hit_rate, avg_confidence,
)
from tests.config.thresholds import THRESHOLDS
from tests.regression.baseline_manager import save_baseline, load_latest_baseline
from tests.regression.regression_detector import detect_regression, check_thresholds


async def call_rag(chat_service, question):
    """调用完整 RAG 流程"""
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
            r["category"] = tc.get("category", "")
            r["keywords"] = tc.get("keywords", [])
            r["error"] = None
            print(f"  -> ctx={len(r['contexts'])}, grade={r['grade']}, conf={r['confidence']:.2f}")
        except Exception as e:
            r = {
                "question": q, "answer": "", "contexts": [],
                "grade": "error", "confidence": 0, "attempts": 0,
                "expected_answer": tc.get("expected_answer", ""),
                "category": tc.get("category", ""),
                "keywords": tc.get("keywords", []),
                "error": str(e),
            }
            print(f"  -> ERROR: {e}")
        results.append(r)
    return results


def run_ragas_eval(results, testset):
    """用 RAGAS 评估端到端指标"""
    import tests.ragas_patch
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from langchain_openai import ChatOpenAI as LangchainChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

    settings = get_settings()

    lc_llm = LangchainChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.CHAT_API_KEY,
        base_url=settings.CHAT_API_BASE,
        temperature=0,
        max_tokens=8192,
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

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

    if not samples:
        return {}

    print(f"\nRAGAS 端到端评估 ({len(samples)} 个有效样本)...")

    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    old_tracing = os.environ.get("LANGSMITH_TRACING")
    os.environ["LANGSMITH_TRACING"] = "false"
    try:
        eval_result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm)
    finally:
        if old_tracing is not None:
            os.environ["LANGSMITH_TRACING"] = old_tracing
        else:
            os.environ.pop("LANGSMITH_TRACING", None)

    ragas_scores = {}
    if hasattr(eval_result, 'items'):
        ragas_scores = {k: round(v, 4) for k, v in eval_result.items()}
    elif hasattr(eval_result, '_scores_dict'):
        for k, v in eval_result._scores_dict.items():
            if v:
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

    return ragas_scores


def main():
    print("=" * 60)
    print("L3 端到端层测试")
    print("=" * 60)

    testset = get_testset()
    settings = get_settings()
    print(f"测试集: {len(testset)} 题")
    print(f"Model: {settings.CHAT_MODEL}")
    print(f"Reranker: {settings.RERANKER_ENABLED}")

    # 初始化服务
    vector_store = get_vector_store()
    sparse_retriever = SparseRetriever()
    retriever = HybridRetriever(
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        rrf_k=60,
        use_reranker=settings.RERANKER_ENABLED,
        reranker_model=settings.RERANKER_MODEL,
        top_n=settings.RERANKER_TOP_N,
    )
    db = SessionLocal()
    chat_service = ChatService(db, retriever)

    # 运行 RAG
    print("\n--- 运行 RAG ---")
    results = asyncio.run(run_all(chat_service, testset))
    db.close()

    # 自定义指标
    answers = [r["answer"] for r in results]
    contexts_list = [r["contexts"] for r in results]
    has_ctx_flags = [len(r["contexts"]) > 0 for r in results]
    confidences = [r["confidence"] for r in results]
    grades = [r["grade"] for r in results]

    custom_scores = {
        "hit_rate": round(hit_rate(has_ctx_flags), 4),
        "no_answer_rate": round(no_answer_rate(answers), 4),
        "article_reference_rate": round(article_reference_rate(answers), 4),
        "hallucination_indicator": round(hallucination_indicator(answers, contexts_list), 4),
        "avg_confidence": round(avg_confidence(confidences), 4),
    }

    # 逐题统计 grade
    grade_counts = {}
    for g in grades:
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print(f"\nGrade 分布: {grade_counts}")

    # RAGAS 评估
    ragas_scores = {}
    try:
        ragas_scores = run_ragas_eval(results, testset)
    except Exception as e:
        print(f"RAGAS 评估失败: {e}")

    # 合并所有分数
    all_scores = {**ragas_scores, **custom_scores}

    print(f"\n--- 端到端层结果 ---")
    for k, v in all_scores.items():
        print(f"  {k}: {v}")

    # 阈值检查
    threshold_result = check_thresholds(all_scores, "e2e")
    print(f"\n--- 阈值检查 ---")
    for r in threshold_result["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['metric']}: {r['value']} (基线: {r['baseline']})")
    print(f"  全部通过: {threshold_result['all_passed']}")

    # 回归检测
    old_baseline = load_latest_baseline()
    regression_info = None
    if old_baseline:
        regression_info = detect_regression(all_scores, old_baseline["scores"], "e2e")
        if regression_info["has_regression"]:
            print(f"\n⚠️ 回归检测:")
            for r in regression_info["regressions"]:
                print(f"  {r['metric']}: {r['old']} -> {r['new']} (Δ={r['delta']}, {r['level']})")
        if regression_info["has_critical"]:
            print(f"\n❌ CRITICAL 回归!")
            for r in regression_info["criticals"]:
                print(f"  {r['metric']}: {r['old']} -> {r['new']} (Δ={r['delta']})")
        if not regression_info["has_regression"]:
            print(f"\n✅ 回归检测通过")
    else:
        print(f"\n⚠️ 无历史基线，跳过回归检测")

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "layer": "e2e",
        "scores": all_scores,
        "threshold_check": threshold_result,
        "regression": regression_info,
        "grade_distribution": grade_counts,
        "per_question": results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent.parent / f"e2e_result_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {filepath}")

    # 是否保存为新基线（需要手动 --save-baseline）
    if "--save-baseline" in sys.argv:
        save_baseline(all_scores)
        print("已保存为新基线")

    # 返回是否通过
    if regression_info and regression_info["has_critical"]:
        return False
    if not threshold_result["all_passed"]:
        return False
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)