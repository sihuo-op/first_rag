import json
with open('/app/tests/ragas_eval_result_20260607_125740.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print('=== 基础统计 ===')
print(json.dumps(d['basic_stats'], indent=2, ensure_ascii=False))
print()
print('=== RAGAS 分数 ===')
print(json.dumps(d['ragas_scores'], indent=2, ensure_ascii=False))
print()
print('=== 逐题摘要 ===')
for i, r in enumerate(d['results'], 1):
    q = r.get('question', '')[:25]
    ans = r.get('answer', '')[:60]
    nctx = len(r.get('contexts', []))
    conf = r.get('confidence', 0)
    print(f'[{i}] {q}... | ctx={nctx} conf={conf:.2f} | {ans}...')
