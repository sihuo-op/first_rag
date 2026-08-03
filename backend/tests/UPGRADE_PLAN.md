# RAG 企业级测试改造方案

## 一、CRAG Unknown Bug 修复

**根因**：`RAGGraph.run()` 返回值中没有 `grade` 字段，`chat_service.py` 的 `agentic_info` 中也没有 `evaluation_grade`，导致测试脚本用 `agentic.get("evaluation_grade", "unknown")` 时永远返回 "unknown"。

**修复链路**（3 个文件）：

| 文件 | 修复内容 |
|------|---------|
| `app/agent/graph.py` | `RAGGraph.run()` 返回字典中添加 `grade` 和 `evaluation_reason` |
| `app/agent/simple_commander.py` | `SimpleCommander.run()` 返回字典中添加 `grade` |
| `app/services/chat_service.py` | `agentic_info` 中添加 `evaluation_grade` |

---

## 二、分层测试架构

```
┌─────────────────────────────────────────────────┐
│            端到端层 (E2E)                        │
│  完整流程：问题→检索→评估→改写→生成              │
│  指标：Faithfulness, AnswerRelevancy, 整体耗时   │
├─────────────────────────────────────────────────┤
│            生成层 (Generation)                   │
│  给定检索结果，测试生成质量                       │
│  指标：Faithfulness, Hallucination, 条款引用率   │
├─────────────────────────────────────────────────┤
│            检索层 (Retrieval)                    │
│  测试检索器召回和精度                             │
│  指标：HitRate, MRR, ContextPrecision/Recall     │
└─────────────────────────────────────────────────┘
```

---

## 三、新增文件结构

```
backend/tests/
├── conftest.py                  # 共享 fixtures（服务初始化、测试集）
├── config/
│   ├── thresholds.py            # 基线阈值配置
│   └── test_profiles.py         # A/B 测试配置档案
├── testsets/
│   └── labor_law_1994.py        # 20 题（保持不变）
├── layers/
│   ├── test_retrieval.py        # 检索层测试
│   ├── test_generation.py       # 生成层测试
│   └── test_e2e.py              # 端到端测试
├── metrics/
│   ├── ragas_metrics.py         # RAGAS 指标封装
│   ├── deepeval_metrics.py      # DeepEval 幻觉检测
│   └── custom_metrics.py        # 自定义指标（条款引用率等）
├── runners/
│   ├── base_runner.py           # 测试运行器基类
│   ├── retrieval_runner.py      # 检索层运行器
│   ├── generation_runner.py     # 生成层运行器
│   └── e2e_runner.py            # 端到端运行器
├── regression/
│   ├── baseline_manager.py      # 基线管理（保存/加载历史分数）
│   └── regression_detector.py   # 回归检测（对比新旧分数）
├── reporting/
│   └── report_generator.py      # Markdown 报告生成
├── ci/
│   ├── run_eval.py              # CI 入口（一键跑全层）
│   └── ab_compare.py            # A/B 对比工具
└── （现有文件保持兼容）
```

---

## 四、核心模块设计

### 4.1 thresholds.py -- 基线阈值

```python
THRESHOLDS = {
    "retrieval": {
        "context_precision": {
            "baseline": 0.60,
            "regression_delta": -0.05,  # 下降超过0.05 → 告警
            "critical_delta": -0.10,    # 下降超过0.10 → 阻断CI
        },
        "context_recall": {
            "baseline": 0.45,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
        "hit_rate": {
            "baseline": 0.85,           # 至少85%的题有检索结果
            "regression_delta": -0.10,
            "critical_delta": -0.15,
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
        "hallucination_rate": {
            "baseline": 0.15,           # 幻觉率不超过15%
            "regression_delta": 0.05,   # 幻觉率上升0.05 → 告警
            "critical_delta": 0.10,     # 幻觉率上升0.10 → 阻断
        },
        "avg_confidence": {
            "baseline": 0.70,
            "regression_delta": -0.05,
            "critical_delta": -0.10,
        },
    },
}
```

### 4.2 test_profiles.py -- A/B 测试配置

```python
# 不同配置档案，用于 A/B 对比
TEST_PROFILES = {
    "default": {
        "description": "当前默认配置",
        "reranker_enabled": True,
        "reranker_top_n": 3,
        "rrf_k": 60,
        "top_k": 10,
        "max_attempts": 2,
        "prompt_style": "strict",  # 严格引用
    },
    "no_reranker": {
        "description": "关闭 reranker 对比",
        "reranker_enabled": False,
        "reranker_top_n": 0,
        "rrf_k": 60,
        "top_k": 10,
        "max_attempts": 2,
        "prompt_style": "strict",
    },
    "loose_prompt": {
        "description": "宽松提示词（允许解读）",
        "reranker_enabled": True,
        "reranker_top_n": 3,
        "rrf_k": 60,
        "top_k": 10,
        "max_attempts": 2,
        "prompt_style": "loose",
    },
    "more_attempts": {
        "description": "更多改写迭代",
        "reranker_enabled": True,
        "reranker_top_n": 3,
        "rrf_k": 60,
        "top_k": 10,
        "max_attempts": 3,
        "prompt_style": "strict",
    },
}
```

### 4.3 custom_metrics.py -- 自定义指标

```python
"""业务相关自定义指标"""
import re

def article_reference_rate(answers: list) -> float:
    """条款引用率：答案中引用了具体法律条文的比例"""
    count = sum(1 for ans in answers
                if ans and re.search(r'第[一二三四五六七八九十百\d]+条', ans))
    return count / len(answers) if answers else 0

def no_answer_rate(answers: list) -> float:
    """空答案率：返回"未找到"或空答案的比例"""
    count = sum(1 for ans in answers
                if not ans or '未找到' in ans or '出错' in ans)
    return count / len(answers) if answers else 0

def hallucination_indicator(answers: list, contexts: list) -> float:
    """幻觉指标：答案内容超出检索上下文的比例（简易版）"""
    hallu_count = 0
    for ans, ctxs in zip(answers, contexts):
        if not ctxs and ans and '未找到' not in ans:
            hallu_count += 1  # 无上下文却有答案 = 疑似幻觉
    return hallu_count / len(answers) if answers else 0

def hit_rate(has_context_flags: list) -> float:
    """检索命中率：至少检索到相关文档的题目比例"""
    return sum(has_context_flags) / len(has_context_flags)

def avg_confidence(confidences: list) -> float:
    """平均置信度"""
    valid = [c for c in confidences if c > 0]
    return sum(valid) / len(valid) if valid else 0
```

### 4.4 baseline_manager.py -- 基线管理

```python
"""基线分数管理 - 保存/加载历史评估结果"""
import json
from pathlib import Path
from datetime import datetime

BASELINE_DIR = Path(__file__).parent.parent / "baselines"

def save_baseline(results: dict, profile_name: str = "default"):
    """保存当前评估结果为基线"""
    BASELINE_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = BASELINE_DIR / f"{profile_name}_baseline_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "profile": profile_name,
            "scores": results,
        }, f, ensure_ascii=False, indent=2)
    # 同时保存为 latest（方便对比）
    latest_path = BASELINE_DIR / f"{profile_name}_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "profile": profile_name, "scores": results}, f, ensure_ascii=False, indent=2)
    return filepath

def load_latest_baseline(profile_name: str = "default") -> dict:
    """加载最近一次基线"""
    latest_path = BASELINE_DIR / f"{profile_name}_latest.json"
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def list_baselines(profile_name: str = "default") -> list:
    """列出所有历史基线"""
    if not BASELINE_DIR.exists():
        return []
    files = sorted(BASELINE_DIR.glob(f"{profile_name}_baseline_*.json"))
    return [f.name for f in files]
```

### 4.5 regression_detector.py -- 回归检测

```python
"""回归检测 - 对比新旧分数，判断是否退化"""
from tests.config.thresholds import THRESHOLDS

def detect_regression(new_scores: dict, old_scores: dict, layer: str = "e2e") -> dict:
    """检测指标是否退化"""
    layer_thresholds = THRESHOLDS.get(layer, {})
    regressions = []
    criticals = []

    for metric, threshold_config in layer_thresholds.items():
        new_val = new_scores.get(metric)
        old_val = old_scores.get(metric)

        if new_val is None or old_val is None:
            continue

        delta = new_val - old_val
        # 对于幻觉率等指标，上升是退化的
        is_inverse = metric in ("hallucination_rate", "no_answer_rate")
        if is_inverse:
            delta = -delta  # 翻转方向

        regression_delta = threshold_config.get("regression_delta", -0.05)
        critical_delta = threshold_config.get("critical_delta", -0.10)

        if delta <= regression_delta:  # delta 现在统一为"正向=好"
            regressions.append({
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "delta": new_val - old_val,
                "level": "regression",
            })
        if delta <= critical_delta:
            criticals.append({
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "delta": new_val - old_val,
                "level": "critical",
            })

    return {
        "has_regression": len(regressions) > 0,
        "has_critical": len(criticals) > 0,
        "regressions": regressions,
        "criticals": criticals,
    }
```

### 4.6 DeepEval 幻觉检测

```python
"""DeepEval 幻觉检测指标"""
# 安装：pip install deepeval

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

def evaluate_hallucination(question: str, answer: str, contexts: list) -> dict:
    """评估单题幻觉"""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=contexts if contexts else ["(无上下文)"],
    )
    metric = HallucinationMetric(
        threshold=0.5,
        model="doubao-seed-2-0-code-preview-260215",  # 需配置豆包兼容
    )
    metric.measure(test_case)
    return {
        "score": metric.score,
        "reason": metric.reason,
        "is_hallucination": metric.score < metric.threshold,
    }
```

### 4.7 report_generator.py -- Markdown 报告

```python
"""Markdown 浥告生成"""

def generate_report(
    layer: str,
    scores: dict,
    regression_info: dict = None,
    per_question_data: list = None,
    profile_name: str = "default",
) -> str:
    """生成 Markdown 格式的评估报告"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"# RAG 评估报告 - {layer.upper()}层\n\n"
    report += f"> 时间: {ts} | 配置: {profile_name}\n\n"

    # 指标汇总表
    report += "## 指标汇总\n\n"
    report += "| 指标 | 当前分数 | 基线 | 状态 |\n"
    report += "|------|---------|------|------|\n"
    for metric, score in scores.items():
        baseline = THRESHOLDS.get(layer, {}).get(metric, {}).get("baseline", "N/A")
        if regression_info:
            status = "✅" if not any(r["metric"] == metric for r in regression_info["regressions"]) else "⚠️退化"
        else:
            status = "✅" if score >= baseline else "❌未达标"
        report += f"| {metric} | {score:.4f} | {baseline} | {status} |\n\n"

    # 回归检测
    if regression_info and regression_info["has_regression"]:
        report += "## ⚠️ 回归检测\n\n"
        for r in regression_info["regressions"]:
            report += f"- **{r['metric']}**: {r['old']:.4f} → {r['new']:.4f} (Δ={r['delta']:.4f}, {r['level']})\n"
        report += "\n"

    # 逐题明细
    if per_question_data:
        report += "## 逐题明细\n\n"
        for q in per_question_data:
            report += f"- [{q['category']}] {q['question'][:30]}... | ctx={q['n_ctx']} conf={q['confidence']:.2f} grade={q['grade']}\n"

    return report
```

### 4.8 ci/run_eval.py -- CI 入口

```python
"""CI 入口脚本 - 一键跑全层评估"""

def run_full_eval(profile_name="default", save_baseline=False):
    """运行全层评估"""
    print("=" * 60)
    print("RAG 全层评估")
    print("=" * 60)

    # 1. 检索层
    print("\n--- L1 检索层 ---")
    retrieval_scores = run_retrieval_layer(profile_name)

    # 2. 生成层
    print("\n--- L2 生成层 ---")
    generation_scores = run_generation_layer(profile_name)

    # 3. 端到端
    print("\n--- L3 端到端 ---")
    e2e_scores = run_e2e_layer(profile_name)

    # 4. 回归检测
    old_baseline = load_latest_baseline(profile_name)
    if old_baseline:
        regression = detect_regression(e2e_scores, old_baseline["scores"], "e2e")
        if regression["has_critical"]:
            print("❌ CRITICAL 回归检测失败，阻断 CI！")
            print(regression["criticals"])
            return False
        if regression["has_regression"]:
            print("⚠️ REGRESSION 检测告警，但不阻断")
            print(regression["regressions"])

    # 5. 保存基线
    if save_baseline:
        save_baseline(e2e_scores, profile_name)

    # 6. 生成报告
    report = generate_report("e2e", e2e_scores, regression, profile_name)
    report_path = Path(__file__).parent.parent / "reports" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="default")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()
    success = run_full_eval(args.profile, args.save_baseline)
    sys.exit(0 if success else 1)
```

### 4.9 ci/ab_compare.py -- A/B 对比

```python
"""A/B 配置对比工具"""

def ab_compare(profile_a="default", profile_b="no_reranker"):
    """对比两个配置档案的评估结果"""
    scores_a = run_e2e_layer(profile_a)
    scores_b = run_e2e_layer(profile_b)

    print(f"\n{'='*60}")
    print(f"A/B 对比: {profile_a} vs {profile_b}")
    print(f"{'='*60}\n")

    print("| 指标 | A ({profile_a}) | B ({profile_b}) | Δ | 优胜 |")
    print("|------|----------------|----------------|-----|------|")
    for metric in scores_a:
        a = scores_a[metric]
        b = scores_b.get(metric, 0)
        delta = b - a
        winner = "B ✅" if delta > 0.02 else ("A ✅" if delta < -0.02 else "持平")
        print(f"| {metric} | {a:.4f} | {b:.4f} | {delta:+.4f} | {winner} |")
```

---

## 五、运行方式

```bash
# CI 全层评估（默认配置）
docker exec first_rag-backend-1 python -m tests.ci.run_eval

# CI 全层评估 + 保存基线
docker exec first_rag-backend-1 python -m tests.ci.run_eval --save-baseline

# A/B 对比（默认 vs 无reranker）
docker exec first_rag-backend-1 python -m tests.ci.ab_compare --a default --b no_reranker

# 单层测试
docker exec first_rag-backend-1 python -m tests.layers.test_retrieval
docker exec first_rag-backend-1 python -m tests.layers.test_generation
docker exec first_rag-backend-1 python -m tests.layers.test_e2e
```

---

## 六、实施优先级

| 优先级 | 任务 | 预估工作量 |
|--------|------|-----------|
| **P0** | CRAG unknown bug 修复 | 30 分钟 |
| **P1** | 基线阈值 + 回归检测 | 1 小时 |
| **P1** | 分层测试架构 | 2 小时 |
| **P2** | 自定义指标（条款引用率、幻觉率等） | 1 小时 |
| **P2** | Markdown 报告生成 | 30 分钟 |
| **P3** | A/B 对比工具 | 1 小时 |
| **P3** | DeepEval 幻觉检测集成 | 1 小时 |