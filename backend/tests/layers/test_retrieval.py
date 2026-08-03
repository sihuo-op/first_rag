"""检索层测试 - L1: 测试检索器召回和精度

只测检索，不测生成。关注：
- 检索命中率（hit_rate）
- 上下文精确度（context_precision）
- 上下文召回率（context_recall）
"""
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.chdir(Path(__file__).parent.parent.parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from app.core.config import get_settings
from app.core.dependencies import get_vector_store
from app.rag.retriever import HybridRetriever, SparseRetriever
from tests.testset_labor_law_1994 import get_testset
from tests.metrics.custom_metrics import hit_rate, avg_confidence
from tests.regression.baseline_manager import save_baseline, load_latest_baseline
from tests.regression.regression_detector import detect_regression, check_thresholds


def init_retriever():
    settings = get_settings()
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
    return retriever


def run_retrieval_for_question(retriever, question, top_k=10):
    """对单个问题运行检索，返回结果"""
    start = time.time()
    results, debug_info = retriever.retrieve(question, top_k=top_k)
    elapsed = time.time() - start

    contexts = [r.get("content", "") for r in results if r.get("content")]
    scores = [r.get("score", 0) for r in results]

    return {
        "contexts": contexts,
        "scores": scores,
        "n_results": len(results),
        "elapsed": round(elapsed, 3),
        "rerank_used": debug_info.get("rerank_used", False),
    }


def run_retrieval_layer():
    """运行检索层全量测试"""
    print("=" * 60)
    print("L1 检索层测试")
    print("=" * 60)

    testset = get_testset()
    settings = get_settings()
    print(f"测试集: {len(testset)} 题")
    print(f"Reranker: {settings.RERANKER_ENABLED}")

    retriever = init_retriever()

    # 逐题检索
    results = []
    has_context_flags = []
    for i, tc in enumerate(testset, 1):
        q = tc["question"]
        print(f"[{i}/{len(testset)}] {q}")

        r = run_retrieval_for_question(retriever, q)
        r["question"] = q
        r["category"] = tc.get("category", "")
        r["article"] = tc.get("article", "")

        has_context_flags.append(r["n_results"] > 0)
        results.append(r)
        print(f"  -> 结果数={r['n_results']}, 耗时={r['elapsed']}s, rerank={r['rerank_used']}")

    # 计算指标
    hr = hit_rate(has_context_flags)

    # RAGAS Context 指标（需要 LLM 评估，此处用简化版本）
    # 先用检索层自己的指标
    scores = {
        "hit_rate": round(hr, 4),
        "avg_results": round(sum(r["n_results"] for r in results) / len(results), 2),
        "avg_elapsed": round(sum(r["elapsed"] for r in results) / len(results), 3),
        "rerank_usage_rate": round(sum(1 for r in results if r["rerank_used"]) / len(results), 4),
    }

    # 如果有 RAGAS context 指标，加上
    try:
        ragas_scores = _run_ragas_context_metrics(results, testset)
        scores.update(ragas_scores)
    except Exception as e:
        print(f"RAGAS Context 指标评估失败: {e}")

    print(f"\n--- L1 检索层结果 ---")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    # 阈值检查
    threshold_result = check_thresholds(scores, "retrieval")
    print(f"\n--- 阈值检查 ---")
    for r in threshold_result["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['metric']}: {r['value']} (基线: {r['baseline']})")

    # 回归检测
    old_baseline = load_latest_baseline()
    if old_baseline:
        regression = detect_regression(scores, old_baseline["scores"], "retrieval")
        if regression["has_regression"]:
            print(f"\n⚠️ 回归检测:")
            for r in regression["regressions"]:
                print(f"  {r['metric']}: {r['old']} -> {r['new']} (Δ={r['delta']}, {r['level']})")
        else:
            print(f"\n✅ 回归检测通过")

    # 保存结果
    output = {
        "layer": "retrieval",
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "threshold_check": threshold_result,
        "per_question": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent.parent / f"retrieval_result_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {filepath}")

    return scores


def _run_ragas_context_metrics(results, testset):
    """用 RAGAS 评估 Context 指标"""
    import tests.ragas_patch
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import ContextPrecision, ContextRecall
    from langchain_openai import ChatOpenAI as LangchainChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    lc_llm = LangchainChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.CHAT_API_KEY,
        base_url=settings.CHAT_API_BASE,
        temperature=0,
        max_tokens=8192,
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

    samples = []
    for r, tc in zip(results, testset):
        if not r["contexts"]:
            continue
        sample = SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            reference=tc.get("expected_answer", ""),
        )
        samples.append(sample)

    if not samples:
        return {}

    print(f"\n  RAGAS Context 评估 ({len(samples)} 个有效样本)...")
    dataset = EvaluationDataset(samples=samples)
    metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    old_tracing = os.environ.get("LANGSMITH_TRACING")
    os.environ["LANGSMITH_TRACING"] = "false"
    try:
        eval_result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm)
    finally:
        if old_tracing:
            os.environ["LANGSMITH_TRACING"] = old_tracing
        else:
            os.environ.pop("LANGSMITH_TRACING", None)

    import math
    ragas_scores = {}
    if hasattr(eval_result, 'items'):
        ragas_scores = {k: round(v, 4) for k, v in eval_result.items()}
    elif hasattr(eval_result, '_scores_dict'):
        for k, v in eval_result._scores_dict.items():
            if v:
                valid = [float(s) for s in v if s is not None and not math.isnan(float(s))]
                if valid:
                    ragas_scores[k] = round(sum(valid) / len(valid), 4)
    return ragas_scores


if __name__ == "__main__":
    scores = run_retrieval_layer()
