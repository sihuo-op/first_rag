"""生成层测试 - L2: 给定检索结果，测试生成质量

关注：
- Faithfulness（忠实度）
- AnswerRelevancy（答案相关性）
- 条款引用率
- 幻觉指标
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
from app.agent.tools import get_generation_llm
from tests.testset_labor_law_1994 import get_testset
from tests.metrics.custom_metrics import article_reference_rate, hallucination_indicator
from tests.regression.baseline_manager import save_baseline, load_latest_baseline
from tests.regression.regression_detector import detect_regression, check_thresholds


def generate_answer_with_context(llm, question: str, contexts: list) -> str:
    """给定上下文，让 LLM 生成答案"""
    context_text = "\n\n".join(
        f"文档片段{i+1}：{c}" for i, c in enumerate(contexts)
    ) if contexts else "(无相关文档)"

    prompt = f"""你是一个专业的劳动法知识助手。请严格根据以下提供的参考文档回答用户问题。

参考文档：
{context_text}

用户问题：{question}

回答要求：
1. 严格基于参考文档内容回答，不得添加文档中未提及的任何信息
2. 直接引用文档中的原文条款，引用时标注来源（如"第X条"或"文档片段N"）
3. 如果文档内容足以回答问题，简洁准确地给出答案，不需要额外解读或分析
4. 如果文档内容仅部分相关，只回答文档能支撑的部分，对文档未覆盖的内容明确说明"参考文档未涉及"
5. 只有当所有文档内容都与问题完全无关时，才说明"未找到相关信息"
6. 禁止对法律条款进行延伸解读、推理或补充说明，只陈述文档中已有的内容

请给出回答："""

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def run_generation_layer():
    """运行生成层全量测试"""
    print("=" * 60)
    print("L2 生成层测试")
    print("=" * 60)

    testset = get_testset()
    settings = get_settings()
    print(f"测试集: {len(testset)} 题")

    # 初始化检索器（用于获取上下文）
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

    # 初始化生成 LLM
    llm = get_generation_llm()

    # 逐题测试
    results = []
    answers = []
    contexts_list = []

    for i, tc in enumerate(testset, 1):
        q = tc["question"]
        print(f"[{i}/{len(testset)}] {q}")

        # 先检索
        try:
            retrieval_results, _ = retriever.retrieve(q, top_k=10)
            contexts = [r.get("content", "") for r in retrieval_results if r.get("content")]
        except Exception as e:
            print(f"  检索失败: {e}")
            contexts = []

        # 再生成
        try:
            start = time.time()
            answer = generate_answer_with_context(llm, q, contexts)
            elapsed = time.time() - start
        except Exception as e:
            print(f"  生成失败: {e}")
            answer = ""
            elapsed = 0

        answers.append(answer)
        contexts_list.append(contexts)

        r = {
            "question": q,
            "answer": answer,
            "n_contexts": len(contexts),
            "elapsed": round(elapsed, 3),
            "category": tc.get("category", ""),
        }
        results.append(r)
        ans_preview = answer[:60] if answer else "(空)"
        print(f"  -> ctx={len(contexts)}, 耗时={elapsed:.1f}s, 答案={ans_preview}...")

    # 计算自定义指标
    arr = article_reference_rate(answers)
    hi = hallucination_indicator(answers, contexts_list)

    scores = {
        "article_reference_rate": round(arr, 4),
        "hallucination_indicator": round(hi, 4),
        "avg_elapsed": round(sum(r["elapsed"] for r in results) / len(results), 3),
    }

    # RAGAS 评估
    try:
        ragas_scores = _run_ragas_generation_metrics(results, testset)
        scores.update(ragas_scores)
    except Exception as e:
        print(f"RAGAS 评估失败: {e}")

    print(f"\n--- L2 生成层结果 ---")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    # 阈值检查
    threshold_result = check_thresholds(scores, "generation")
    print(f"\n--- 阈值检查 ---")
    for r in threshold_result["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['metric']}: {r['value']} (基线: {r['baseline']})")

    # 回归检测
    old_baseline = load_latest_baseline()
    if old_baseline:
        regression = detect_regression(scores, old_baseline["scores"], "generation")
        if regression["has_regression"]:
            print(f"\n⚠️ 回归检测:")
            for r in regression["regressions"]:
                print(f"  {r['metric']}: {r['old']} -> {r['new']} (Δ={r['delta']}, {r['level']})")
        else:
            print(f"\n✅ 回归检测通过")

    # 保存结果
    output = {
        "layer": "generation",
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "threshold_check": threshold_result,
        "per_question": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent.parent / f"generation_result_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {filepath}")

    return scores


def _run_ragas_generation_metrics(results, testset):
    """用 RAGAS 评估生成层指标"""
    import tests.ragas_patch
    import math
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy
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
    for r, tc in zip(results, testset):
        if not r.get("answer") or not r.get("n_contexts", 0):
            continue
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=[],  # 生成层不评估 context 指标
            reference=tc.get("expected_answer", ""),
        )
        samples.append(sample)

    if not samples:
        return {}

    print(f"\n  RAGAS 生成层评估 ({len(samples)} 个有效样本)...")
    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
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
    scores = run_generation_layer()
