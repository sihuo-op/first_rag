# RAG Evaluation Report

- Timestamp: `2026-06-28T20:42:57`
- Dataset: `D:\tln\code\first_rag\backend\evals\golden_labor_law.jsonl`
- Overall passed: **False**

## Summary

### e2e

| Metric | Value |
|---|---:|
| answer_rule_score | 1.0 |
| context_hit_rate | 1.0 |
| memory_rewrite_hit_rate | 1.0 |
| multi_task_success_rate | 1.0 |
| case_count | 1 |
| error_rate | 1.0 |
| avg_latency_seconds | 124.861 |
| p95_latency_seconds | 124.861 |

| Threshold | Value | Target | Passed |
|---|---:|---:|:---:|
| answer_rule_score | 1.0 | 0.55 | True |
| memory_rewrite_hit_rate | 1.0 | 0.5 | True |
| multi_task_success_rate | 1.0 | 0.5 | True |
| error_rate | 1.0 | 0.1 | False |
| p95_latency_seconds | 124.861 | 120 | False |

## Failed / Low-score Cases

| Mode | Case | Category | Reason |
|---|---|---|---|
| e2e | negative_001 | negative_or_irrelevant | case has runtime errors |

## Slowest Cases

| Mode | Case | Latency seconds |
|---|---|---:|
| e2e | negative_001 | 124.861 |
