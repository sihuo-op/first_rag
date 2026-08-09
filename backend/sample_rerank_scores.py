"""采样 rerank_score 分布，用于确定 score-based evaluate 阈值。

直接调用 HybridRetriever，绕过 HTTP/auth，跑一批多样化查询，
收集每个查询 top-N 的 rerank_score，输出分布统计。
"""
import csv
import os
import sys
import statistics
from pathlib import Path

# 复用 backend 的环境
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

# jieba 临时目录
JIEBA_CACHE_DIR = BACKEND_DIR / "data" / "jieba_cache"
JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMP"] = str(JIEBA_CACHE_DIR)
os.environ["TMPDIR"] = str(JIEBA_CACHE_DIR)


# 测试查询：覆盖高/中/低相关度场景
TEST_QUERIES = [
    # 高相关（劳动法明确条款）
    "年休假有几天",
    "法定年休假多少天",
    "加班工资怎么计算",
    "试用期最长多久",
    "劳动合同终止的情形",
    "经济补偿金计算公式",
    "病假工资怎么发",
    "婚假有几天",
    "产假多少天",
    "社会保险包括哪些",
    # 中等相关（更宽泛）
    "公司不签合同怎么办",
    "员工辞职需要提前多久通知",
    "调岗降薪合法吗",
    "工伤认定标准",
    "拖欠工资怎么维权",
    # 低相关/边界（可能命中差或无命中）
    "今天天气怎么样",
    "劳动法",
    "解除",
    "工资",
    "合同",
    # 改写后会更好的场景
    "他来了之后工资怎么算",
    "那个假期算不算",
    "上周说的那个条款适用吗",
]


def load_bm25_index(retriever):
    """复用 main.py 的 BM25 加载逻辑"""
    from app.db.session import SessionLocal
    from app.entities.database import DocumentChunk, DocumentStatus, Document
    db = SessionLocal()
    try:
        chunks = db.query(DocumentChunk).join(Document).filter(
            Document.status == DocumentStatus.COMPLETED
        ).all()
        if not chunks:
            print(f"[WARN] No chunks found in DB")
            return 0
        contents = [c.content for c in chunks]
        metadata_list = [
            {"document_id": c.document_id, "chunk_type": c.chunk_type.value, "db_chunk_id": c.id}
            for c in chunks
        ]
        retriever.add_to_sparse_index(contents, metadata_list)
        return len(chunks)
    finally:
        db.close()


def main():
    from app.core.dependencies import get_retriever

    print("Initializing retriever...")
    retriever = get_retriever()
    chunk_count = load_bm25_index(retriever)
    print(f"BM25 index loaded: {chunk_count} chunks")

    output_file = BACKEND_DIR / "rerank_score_samples.csv"
    rows = []

    print(f"\nRunning {len(TEST_QUERIES)} queries...")
    for i, query in enumerate(TEST_QUERIES, 1):
        try:
            results, debug_info = retriever.retrieve(query, top_k=10)
            if not results:
                print(f"[{i:>2}/{len(TEST_QUERIES)}] {query!r:<40} -> 0 results")
                rows.append({"query": query, "result_count": 0, "top1_score": "",
                             "top2_score": "", "top3_score": "", "avg_top5": "",
                             "min_score": "", "max_score": ""})
                continue

            scores = [r.get("rerank_score") or r.get("rrf_score") or 0 for r in results]
            scores_sorted = sorted(scores, reverse=True)
            top1 = scores_sorted[0] if len(scores_sorted) > 0 else 0
            top2 = scores_sorted[1] if len(scores_sorted) > 1 else 0
            top3 = scores_sorted[2] if len(scores_sorted) > 2 else 0
            top5 = scores_sorted[:5]
            avg_top5 = sum(top5) / len(top5) if top5 else 0
            min_s = min(scores) if scores else 0
            max_s = max(scores) if scores else 0

            print(f"[{i:>2}/{len(TEST_QUERIES)}] {query!r:<40} -> n={len(results):>2} "
                  f"top1={top1:>7.3f} top3_avg={(top1+top2+top3)/3:>7.3f} min={min_s:>7.3f}")

            rows.append({
                "query": query,
                "result_count": len(results),
                "top1_score": round(top1, 4),
                "top2_score": round(top2, 4),
                "top3_score": round(top3, 4),
                "avg_top5": round(avg_top5, 4),
                "min_score": round(min_s, 4),
                "max_score": round(max_s, 4),
            })
        except Exception as e:
            print(f"[{i:>2}/{len(TEST_QUERIES)}] {query!r:<40} -> ERROR: {e}")
            rows.append({"query": query, "result_count": -1, "top1_score": "",
                         "top2_score": "", "top3_score": "", "avg_top5": "",
                         "min_score": "", "max_score": ""})

    # 写 CSV
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "result_count", "top1_score",
                                               "top2_score", "top3_score", "avg_top5",
                                               "min_score", "max_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSamples written to: {output_file}")

    # 分布统计
    top1_scores = [r["top1_score"] for r in rows if r["top1_score"] != ""]
    if not top1_scores:
        print("No valid samples")
        return

    print("\n" + "=" * 60)
    print("Top-1 rerank_score distribution:")
    print("=" * 60)
    print(f"  count:  {len(top1_scores)}")
    print(f"  min:    {min(top1_scores):.3f}")
    print(f"  p25:    {statistics.quantiles(top1_scores, n=4)[0]:.3f}")
    print(f"  median: {statistics.median(top1_scores):.3f}")
    print(f"  p75:    {statistics.quantiles(top1_scores, n=4)[2]:.3f}")
    print(f"  max:    {max(top1_scores):.3f}")
    print(f"  mean:   {statistics.mean(top1_scores):.3f}")
    print(f"  stdev:  {statistics.stdev(top1_scores):.3f}")

    # 分桶
    buckets = {
        "< 0": 0,
        "0 ~ 0.3": 0,
        "0.3 ~ 1": 0,
        "1 ~ 3": 0,
        "3 ~ 5": 0,
        "5 ~ 8": 0,
        ">= 8": 0,
    }
    for s in top1_scores:
        if s < 0: buckets["< 0"] += 1
        elif s < 0.3: buckets["0 ~ 0.3"] += 1
        elif s < 1: buckets["0.3 ~ 1"] += 1
        elif s < 3: buckets["1 ~ 3"] += 1
        elif s < 5: buckets["3 ~ 5"] += 1
        elif s < 8: buckets["5 ~ 8"] += 1
        else: buckets[">= 8"] += 1

    print("\nTop-1 score buckets:")
    for label, count in buckets.items():
        bar = "#" * count
        print(f"  {label:>10}: {count:>2} {bar}")

    # 分位数：找候选阈值
    sorted_scores = sorted(top1_scores)
    n = len(sorted_scores)
    print("\nThreshold candidates:")
    print(f"  low (p10):  {sorted_scores[max(0, n//10)]:.3f}")
    print(f"  low (p20):  {sorted_scores[max(0, n//5)]:.3f}")
    print(f"  high (p80): {sorted_scores[min(n-1, n*4//5)]:.3f}")
    print(f"  high (p90): {sorted_scores[min(n-1, n*9//10)]:.3f}")


if __name__ == "__main__":
    main()
