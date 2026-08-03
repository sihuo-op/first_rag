# RAG Evaluation Suite

企业级 RAG 评测入口，覆盖检索层、RAG 工具层和 API 端到端链路。

## 常用命令

```bash
# 校验数据集
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/evals/run_eval.py --mode validate

# 检索层评测
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/evals/run_eval.py --mode retrieval --top-k 10 --no-fail

# RAG 工具层评测
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/evals/run_eval.py --mode rag --no-fail

# API 端到端评测，需要先启动后端
PYTHONPATH=D:/tln/code/first_rag/backend D:/tln/code/first_rag/.venv/Scripts/python.exe D:/tln/code/first_rag/backend/evals/run_eval.py --mode e2e --base-url http://127.0.0.1:8000 --no-fail
```

## 输出

报告输出到：

```text
backend/evals/reports/latest.json
backend/evals/reports/latest.md
```

`latest.json` 保存完整结构化结果；`latest.md` 保存摘要、阈值检查、失败 case 和慢查询。

## Golden Dataset

`golden_labor_law.jsonl` 每行一个 case，核心字段：

- `id`: 唯一编号
- `category`: single_hop / multi_hop / multi_task / follow_up_memory / cross_session_memory / no_answer / negative_or_irrelevant
- `query`: 单轮问题
- `conversation`: 多轮记忆评测，可选
- `expected.must_retrieve.content_keywords`: 检索应覆盖的关键词
- `expected.answer_must_include`: 答案应包含的关键词
- `expected.answer_must_not_include`: 答案不应包含的高风险表述
- `expected.expected_intent_count`: 多任务拆分期望数量
- `expected.standalone_query_keywords`: 记忆改写后应包含的关键词
