"""
测试边界情况 - 验证 AI 不会在无相关内容时瞎编
"""
import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

def login():
    response = httpx.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    return response.json()["access_token"]

def test_query(token: str, query: str, expect_answer: bool, category: str):
    """Send query and check if answer is hallucinating"""
    headers = {"Authorization": f"Bearer {token}"}

    response = httpx.post(
        f"{BASE_URL}/api/v1/chat",
        headers=headers,
        json={"query": query, "use_rag": True},
        timeout=120.0
    )

    data = response.json()
    answer = data.get("answer", "")
    chunks = data.get("retrieved_chunks", [])

    # 判断是否"瞎编"：
    # - 如果期望有答案但回答"未找到" → 漏报
    # - 如果期望无答案但给出了具体内容 → 可能是瞎编
    has_real_content = len(chunks) > 0 and chunks[0].get("rerank_score", 0) > 0.5

    no_answer_keywords = ["未找到", "暂无", "无法回答", "没有相关", "未查询到", "暂未"]

    is_refusing = any(kw in answer for kw in no_answer_keywords)

    if expect_answer and is_refusing:
        status = "[MISS]"  # 漏报：应该回答但拒绝
    elif not expect_answer and not is_refusing and len(answer) > 50:
        status = "[HALLU?]"  # 可能瞎编：不应该有答案但给出了长回答
    else:
        status = "[OK]"

    return {
        "query": query[:30] + "..." if len(query) > 30 else query,
        "category": category,
        "expect_answer": expect_answer,
        "chunks": len(chunks),
        "top_rerank": round(chunks[0].get("rerank_score", 0), 3) if chunks else 0,
        "is_refusing": is_refusing,
        "status": status,
        "answer_preview": answer[:80] + "..." if len(answer) > 80 else answer
    }

def main():
    print("=" * 70)
    print("Boundary Case Test - Checking for Hallucination")
    print("=" * 70)

    token = login()
    print(f"Logged in.\n")

    # 测试用例
    test_cases = [
        # 1. 相关问题（应该能回答）
        ("老板让我签空白合同，我可以拒绝吗？", True, "relevant"),
        ("劳动合同无效的情形有哪些？", True, "relevant"),
        ("什么情况下可以解除劳动合同？", True, "relevant"),

        # 2. 完全无关问题（应该拒绝回答，不应该瞎编）
        ("今天北京的天气怎么样？", False, "irrelevant"),
        ("中国A股大盘今天涨跌如何？", False, "irrelevant"),
        ("怎么做红烧肉好吃？", False, "irrelevant"),
        ("Python怎么安装？", False, "irrelevant"),
        ("NBA总冠军是谁？", False, "irrelevant"),

        # 3. 边缘问题（知识库可能有部分相关，但不确定）
        ("公司拖欠工资怎么办？", True, "edge"),
        ("怀孕期间可以辞退员工吗？", True, "edge"),
        ("试用期可以不交社保吗？", True, "edge"),

        # 4. 完全无关的法律问题（测试是否法律领域就瞎编）
        ("交通事故赔偿标准是什么？", False, "irrelevant-law"),
        ("离婚财产怎么分割？", False, "irrelevant-law"),
        ("知识产权侵权怎么维权？", False, "irrelevant-law"),
    ]

    results = []
    for query, expect_answer, category in test_cases:
        result = test_query(token, query, expect_answer, category)
        results.append(result)

        print(f"[{result['category']:15s}] {result['status']:8s} "
              f"chunks={result['chunks']} rerank={result['top_rerank']:.3f}")
        print(f"  Q: {result['query']}")
        print(f"  A: {result['answer_preview']}")
        print()

    # 统计
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    hallu_count = sum(1 for r in results if r["status"] == "[HALLU?]")
    miss_count = sum(1 for r in results if r["status"] == "[MISS]")
    ok_count = sum(1 for r in results if r["status"] == "[OK]")

    print(f"\nHallucination suspects: {hallu_count}")
    print(f"Missed answers: {miss_count}")
    print(f"Correct: {ok_count}")

    # 详细展示可疑的瞎编
    if hallu_count > 0:
        print("\n[Potential Hallucinations]")
        for r in results:
            if r["status"] == "[HALLU?]":
                print(f"  - {r['query']}")
                print(f"    Answer: {r['answer_preview']}")

if __name__ == "__main__":
    main()
