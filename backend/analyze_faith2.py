"""分析 Faithfulness 为 0 的具体原因"""
import json

with open('/app/tests/ragas_eval_result_20260607_145606.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("=== 基础统计 ===")
print(json.dumps(d['basic_stats'], indent=2, ensure_ascii=False))
print()
print("=== RAGAS 分数 ===")
print(json.dumps(d['ragas_scores'], indent=2, ensure_ascii=False))
print()

no_answer = 0
has_ctx = 0
no_ctx = 0
interpret_count = 0

for i, r in enumerate(d['results'], 1):
    q = r.get('question', '')
    ans = r.get('answer', '')
    ctxs = r.get('contexts', [])
    conf = r.get('confidence', 0)
    nctx = len(ctxs)

    is_no_answer = '未找到' in ans or not ans or '出错' in ans
    if is_no_answer:
        no_answer += 1
    if nctx > 0:
        has_ctx += 1
    else:
        no_ctx += 1

    interpret_words = ['解读', '理解', '意味着', '建议', '注意', '提醒', '分析', '推断', '可以推断', '说明']
    has_interpret = any(w in ans for w in interpret_words)
    if has_interpret:
        interpret_count += 1

    # 打印每题的答案前100字
    print(f"[{i}] {q[:30]}... | ctx={nctx} conf={conf:.2f} | {'❌无答案' if is_no_answer else '✅'} | {'⚠️有解读' if has_interpret else ''}")
    print(f"    答案: {ans[:150]}...")
    print()

print(f"\n汇总: 无答案={no_answer}, 有ctx={has_ctx}, 无ctx={no_ctx}, 有解读={interpret_count}")
