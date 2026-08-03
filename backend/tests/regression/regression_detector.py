"""回归检测 - 对比新旧分数，判断是否退化"""
from tests.config.thresholds import THRESHOLDS


def detect_regression(new_scores: dict, old_scores: dict, layer: str = "e2e") -> dict:
    """检测指标是否退化

    Returns:
        {
            "has_regression": bool,   # 是否有退化
            "has_critical": bool,     # 是否有严重退化（应阻断CI）
            "regressions": list,      # 退化列表
            "criticals": list,        # 严重退化列表
            "improvements": list,     # 改善列表
        }
    """
    layer_thresholds = THRESHOLDS.get(layer, {})
    regressions = []
    criticals = []
    improvements = []

    for metric, threshold_config in layer_thresholds.items():
        new_val = new_scores.get(metric)
        old_val = old_scores.get(metric)

        if new_val is None or old_val is None:
            continue

        delta = new_val - old_val

        # 对于 no_answer_rate / hallucination_rate，上升是退化的
        is_inverse = metric in ("no_answer_rate", "hallucination_rate")
        effective_delta = -delta if is_inverse else delta

        regression_delta = threshold_config.get("regression_delta", -0.05)
        critical_delta = threshold_config.get("critical_delta", -0.10)

        entry = {
            "metric": metric,
            "old": round(old_val, 4),
            "new": round(new_val, 4),
            "delta": round(delta, 4),
        }

        if effective_delta >= abs(regression_delta) * 0.5 and delta > 0:
            improvements.append(entry)
        elif effective_delta <= critical_delta:
            criticals.append({**entry, "level": "critical"})
            regressions.append({**entry, "level": "critical"})
        elif effective_delta <= regression_delta:
            regressions.append({**entry, "level": "regression"})

    return {
        "has_regression": len(regressions) > 0,
        "has_critical": len(criticals) > 0,
        "regressions": regressions,
        "criticals": criticals,
        "improvements": improvements,
    }


def check_thresholds(scores: dict, layer: str = "e2e") -> dict:
    """检查分数是否达到基线阈值"""
    layer_thresholds = THRESHOLDS.get(layer, {})
    results = []

    for metric, threshold_config in layer_thresholds.items():
        value = scores.get(metric)
        if value is None:
            continue

        baseline = threshold_config.get("baseline", 0)
        is_inverse = metric in ("no_answer_rate", "hallucination_rate")

        passed = value <= baseline if is_inverse else value >= baseline
        results.append({
            "metric": metric,
            "value": round(value, 4),
            "baseline": baseline,
            "passed": passed,
        })

    return {
        "all_passed": all(r["passed"] for r in results),
        "results": results,
    }
