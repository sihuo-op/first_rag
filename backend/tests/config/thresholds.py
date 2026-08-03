"""基线阈值配置 - 所有评估指标的通过标准"""

THRESHOLDS = {
    "retrieval": {
        "context_precision": {
            "baseline": 0.60,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "context_recall": {
            "baseline": 0.45,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "hit_rate": {
            "baseline": 0.85,
            "regression_delta": -0.10,
            "critical_delta": -0.15,
        },
        "mrr": {
            "baseline": 0.50,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
    },
    "generation": {
        "faithfulness": {
            "baseline": 0.60,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "answer_relevancy": {
            "baseline": 0.65,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "article_reference_rate": {
            "baseline": 0.50,
            "regression_delta": -0.10,
            "critical_delta": -0.15,
        },
    },
    "e2e": {
        "faithfulness": {
            "baseline": 0.60,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "answer_relevancy": {
            "baseline": 0.65,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "context_precision": {
            "baseline": 0.60,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "context_recall": {
            "baseline": 0.45,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "hit_rate": {
            "baseline": 0.85,
            "regression_delta": -0.10,
            "critical_delta": -0.15,
        },
        "no_answer_rate": {
            "baseline": 0.15,
            "regression_delta": 0.05,
            "critical_delta": 0.10,
        },
        "article_reference_rate": {
            "baseline": 0.50,
            "regression_delta": -0.10,
            "critical_delta": -0.15,
        },
        "avg_confidence": {
            "baseline": 0.70,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
    },
}
