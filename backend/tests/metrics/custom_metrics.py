"""自定义业务指标"""
import re


def article_reference_rate(answers: list) -> float:
    """条款引用率：答案中引用了具体法律条文的比例"""
    if not answers:
        return 0.0
    count = sum(1 for ans in answers
                if ans and re.search(r'第[一二三四五六七八九十百千万\d]+条', ans))
    return count / len(answers)


def no_answer_rate(answers: list) -> float:
    """空答案率：返回"未找到"或空答案的比例"""
    if not answers:
        return 0.0
    count = sum(1 for ans in answers
                if not ans or '未找到' in ans or '出错' in ans)
    return count / len(answers)


def hallucination_indicator(answers: list, contexts_list: list) -> float:
    """幻觉指标：无上下文却有非空答案的比例（简易版）"""
    if not answers:
        return 0.0
    hallu_count = 0
    for ans, ctxs in zip(answers, contexts_list):
        if (not ctxs or len(ctxs) == 0) and ans and '未找到' not in ans:
            hallu_count += 1
    return hallu_count / len(answers)


def hit_rate(has_context_flags: list) -> float:
    """检索命中率"""
    if not has_context_flags:
        return 0.0
    return sum(1 for f in has_context_flags if f) / len(has_context_flags)


def avg_confidence(confidences: list) -> float:
    """平均置信度（排除0值）"""
    valid = [c for c in confidences if c > 0]
    return sum(valid) / len(valid) if valid else 0.0


def mrr_score(relevance_ranks: list) -> float:
    """MRR (Mean Reciprocal Rank)：第一个相关结果的排名倒数均值"""
    if not relevance_ranks:
        return 0.0
    reciprocals = []
    for rank in relevance_ranks:
        if rank and rank > 0:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals)
