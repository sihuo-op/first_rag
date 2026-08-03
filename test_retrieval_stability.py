"""
Test retrieval stability - run same query 50 times and compare results
"""
import httpx
import json
import time
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

def login():
    """Login and get token"""
    response = httpx.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={
            "username": "admin",
            "password": "admin123"
        }
    )
    data = response.json()
    return data["access_token"]

def test_query(token: str, query: str, test_id: int):
    """Send query and return retrieval results"""
    headers = {"Authorization": f"Bearer {token}"}

    response = httpx.post(
        f"{BASE_URL}/api/v1/chat",
        headers=headers,
        json={
            "query": query,
            "use_rag": True
        },
        timeout=120.0
    )

    data = response.json()

    # Extract key info
    chunks = data.get("retrieved_chunks", [])
    debug = data.get("debug_info", {})

    result = {
        "test_id": test_id,
        "answer": data.get("answer", "")[:100],
        "chunk_count": len(chunks),
        "answer_correct": "可以拒绝" in data.get("answer", "") or "拒绝" in data.get("answer", ""),
        "rerank_scores": [],
        "chunk_contents": []
    }

    for i, chunk in enumerate(chunks[:3]):
        content_preview = chunk.get("content", "")[:50] if chunk.get("content") else ""
        result["rerank_scores"].append(round(chunk.get("rerank_score", 0), 4))
        result["chunk_contents"].append(content_preview)

    return result

def main():
    print("=" * 60)
    print("Retrieval Stability Test - 50 runs")
    print("=" * 60)

    # Login
    print("\n[1] Logging in...")
    token = login()
    print(f"    Token: {token[:20]}...")

    query = "老板让我签空白合同，我可以拒绝吗？"

    print(f"\n[2] Test query: {query}")
    print(f"[3] Starting 50 consecutive tests...\n")

    results = []

    for i in range(50):
        try:
            result = test_query(token, query, i + 1)
            results.append(result)

            status = "[OK]" if result["answer_correct"] else "[X] "
            print(f"[{i+1:2d}] {status} chunks={result['chunk_count']} "
                  f"rerank_top3={result['rerank_scores']} "
                  f"answer={'correct' if result['answer_correct'] else 'wrong'}")
        except Exception as e:
            print(f"[{i+1:2d}] [X] ERROR: {e}")

        time.sleep(0.5)

    # Statistics
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    correct_count = sum(1 for r in results if r["answer_correct"])
    fail_count = len(results) - correct_count

    print(f"\nTotal tests: {len(results)}")
    print(f"Correct answers: {correct_count} ({correct_count/len(results)*100:.1f}%)")
    print(f"Wrong answers: {fail_count} ({fail_count/len(results)*100:.1f}%)")

    # Analyze rerank scores
    all_rerank_scores = []
    for r in results:
        all_rerank_scores.extend(r["rerank_scores"])

    if all_rerank_scores:
        print(f"\nRerank Score Statistics:")
        print(f"  Max: {max(all_rerank_scores):.4f}")
        print(f"  Min: {min(all_rerank_scores):.4f}")
        print(f"  Avg: {sum(all_rerank_scores)/len(all_rerank_scores):.4f}")

        # Check score variance
        first_scores = results[0]["rerank_scores"]
        diff_count = sum(1 for r in results if r["rerank_scores"] != first_scores)
        print(f"  Different from first run: {diff_count} times")

    # Chunk count distribution
    chunk_counts = [r["chunk_count"] for r in results]
    print(f"\nChunk Count:")
    print(f"  Min: {min(chunk_counts)}")
    print(f"  Max: {max(chunk_counts)}")
    if min(chunk_counts) != max(chunk_counts):
        print(f"  WARNING: Chunk count varies!")

    # Group by result
    correct_scores = []
    fail_scores = []
    for r in results:
        if r["rerank_scores"]:
            if r["answer_correct"]:
                correct_scores.append(r["rerank_scores"][0])
            else:
                fail_scores.append(r["rerank_scores"][0])

    if correct_scores and fail_scores:
        print(f"\nTop1 Rerank Score by Result:")
        print(f"  Avg when correct: {sum(correct_scores)/len(correct_scores):.4f}")
        print(f"  Avg when wrong: {sum(fail_scores)/len(fail_scores):.4f}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
