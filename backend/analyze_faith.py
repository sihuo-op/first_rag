"""分析 Faithfulness 低分原因"""
import json

with open('/app/tests/ragas_eval_result_20260607_134209.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("=== 逐题分析 ===\n")
for i, r in enumerate(d['results'], 1):
    q = r.get('question', '')
    ans = r.get('answer', '')
    ctxs = r.get('contexts', [])
    conf = r.get('confidence', 0)
    nctx = len(ctxs)
    
    # 分析答案中哪些内容可能超出上下文
    # 1. 检查答案长度
    ans_len = len(ans)
    
    # 2. 检查答案中的"解读"性内容
    interpret_words = ['解读', '理解', '意味着', '可以看出', '建议', '注意', '提醒', '综上', '分析', '推断']
    has_interpret = any(w in ans for w in interpret_words)
    
    # 3. 检查答案是否引用了具体条文
    import re
    article_refs = re.findall(r'第[一二三四五六七八九十百\d]+条', ans)
    
    # 4. 检查答案中"未找到"或空答案
    no_answer = '未找到' in ans or not ans
    
    # 5. 检查上下文总长度
    ctx_total = sum(len(c) for c in ctxs) if ctxs else 0
    
    # 6. 计算答案与上下文的重叠度（简单关键词匹配）
    if ctxs and ans:
        ctx_text = ' '.join(ctxs)
        ans_chars = set(ans)
        ctx_chars = set(ctx_text)
        overlap = len(ans_chars & ctx_chars) / max(len(ans_chars), 1)
    else:
        overlap = 0
    
    status = "❌ 无答案" if no_answer else ("⚠️ 有解读" if has_interpret else "✅")
    
    print(f"[{i}] {q[:30]}...")
    print(f"    状态: {status} | ctx={nctx} | ctx长度={ctx_total} | 答案长度={ans_len} | 条文引用={len(article_refs)}个 | 字符重叠={overlap:.2f}")
    if has_interpret:
        # 找出解读性句子
        for line in ans.split('\n'):
            for w in interpret_words:
                if w in line:
                    print(f"    >> 解读: {line.strip()[:80]}")
                    break
    if no_answer:
        print(f"    >> 答案: {ans[:100]}")
    print()
