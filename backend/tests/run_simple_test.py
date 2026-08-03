"""简化版 RAG 测试脚本 - 不依赖 ragas"""
import asyncio
import sys
import os
import json

sys.path.insert(0, '/app')
os.chdir('/app')

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


async def test_questions(chat_service, questions):
    results = []
    for i, q in enumerate(questions, 1):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{i}/{len(questions)}] 问题: {q}")
        try:
            result = await chat_service.agentic_chat(user_id=1, query=q, max_attempts=2)
            answer = result.get('answer', '')
            chunks = result.get('retrieved_chunks', [])
            agentic = result.get('agentic_info', {})
            grade = agentic.get('evaluation_grade', 'unknown')
            confidence = agentic.get('confidence', 0)
            attempts = agentic.get('attempt_count', 0)

            print(f"答案: {answer[:300]}")
            print(f"检索片段数: {len(chunks)}")
            print(f"CRAG: grade={grade}, confidence={confidence:.2f}, attempts={attempts}")

            results.append({
                "question": q,
                "answer": answer,
                "num_chunks": len(chunks),
                "grade": grade,
                "confidence": confidence,
                "attempts": attempts,
                "error": None
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question": q,
                "answer": "",
                "num_chunks": 0,
                "grade": "error",
                "confidence": 0,
                "attempts": 0,
                "error": str(e)
            })
    return results


def main():
    print("=" * 60)
    print("劳动法(1994) RAG 简化测试")
    print("=" * 60)

    # 获取测试集
    testset = get_testset()
    questions = [tc["question"] for tc in testset]
    print(f"\n测试集: {len(questions)} 道题")

    # 初始化
    chat_service, db = init_services()

    # 运行测试
    results = asyncio.run(test_questions(chat_service, questions))

    # 统计
    total = len(results)
    correct = sum(1 for r in results if r["grade"] == "correct")
    incorrect = sum(1 for r in results if r["grade"] == "incorrect")
    ambiguous = sum(1 for r in results if r["grade"] == "ambiguous")
    errors = sum(1 for r in results if r["grade"] == "error")
    no_answer = sum(1 for r in results if "未找到" in r["answer"] or r["answer"] == "")

    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"总数: {total}")
    print(f"CRAG correct: {correct}")
    print(f"CRAG incorrect: {incorrect}")
    print(f"CRAG ambiguous: {ambiguous}")
    print(f"错误: {errors}")
    print(f"未找到答案: {no_answer}")

    if total > 0:
        avg_conf = sum(r["confidence"] for r in results) / total
        print(f"平均置信度: {avg_conf:.2f}")

    # 保存
    from datetime import datetime
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "ambiguous": ambiguous,
            "errors": errors,
            "no_answer": no_answer,
            "avg_confidence": round(sum(r["confidence"] for r in results) / total, 2) if total else 0
        },
        "results": results
    }
    output_file = f"/app/tests/rag_test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")

    db.close()


if __name__ == "__main__":
    main()
