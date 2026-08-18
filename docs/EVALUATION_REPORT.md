# 评估报告：Agentic RAG 法律问答系统

> 对象：劳动法领域问答系统（Agentic RAG + 混合检索 + 知识图谱第三路检索 + 长期记忆）
> 评估维度：**检索质量（RAGAS）** 与 **端到端延迟（Jaeger/OpenTelemetry + 计时）**
> 测试集：`backend/tests/testsets/labor_law_full.py`，50 题（40 正样例：30 RAG + 10 direct_llm；10 负样例）

---

## 一、检索质量评估（RAGAS）

RAGAS 使用 LLM 作为评判器（Eval LLM），对生成回答的 4 个维度打分。评估脚本见 `backend/tests/run_ragas_full_eval.py`。

### 1.1 总体分数（n=50，mean / median）

| 指标 | 基线①(08-09) | 修复后②(08-09) | 最新③(08-11) | 说明 |
|------|:---:|:---:|:---:|------|
| faithfulness | 0.550 / 0.707 | **0.711 / 0.889** | 0.676 / 0.852 | 回答是否忠实于检索上下文（幻觉指标） |
| answer_relevancy | 0.778 / 0.956 | 0.789 / 0.908 | 0.784 / 0.957 | 回答是否切题 |
| context_precision | 0.561 / 0.833 | **0.652 / 0.867** | 0.609 / 0.917 | 检索上下文是否相关 |
| context_recall | 0.625 / 1.000 | **0.700 / 1.000** | 0.632 / 1.000 | 检索上下文是否覆盖答案要点 |

① 基线 = `ragas_full_eval_20260809_194731.json`（路由 Bug 修复前）
② 修复后 = `ragas_full_eval_20260809_202148_rescored.json`（关键词路由扩展 + 父子块联动修复）
③ 最新 = `ragas_full_eval_20260811_194407.json`（small-only 检索 + 按需父级提升 + ONNX int8 rerank 之后）

> ⚠️ RAGAS 分数存在 run-to-run 波动（LLM 判分固有的方差），均值更稳健，应看趋势而非单点。

### 1.2 RAG 核心路径分解（n=31，最关键的检索问答场景）

| 指标 | 基线 | 修复后 | 变化 |
|------|:---:|:---:|:---:|
| faithfulness median | 0.800 | **1.000** | **+0.200** |
| faithfulness mean | 0.733 | **0.921** | **+0.188** |
| context_precision mean | 0.787 | **0.931** | **+0.144** |
| context_recall mean | 0.815 | **1.000** | **+0.185** |

→ 核心改进：**幻觉显著减少**（faithfulness）、**检索覆盖完整**（recall 100%）。

### 1.3 正确性 / 路由统计

| 统计 | 基线 | 修复后 |
|------|:---:|:---:|
| 有上下文作答数 has_context | 31 | **38** |
| 正确数 correct | 31 | **35** |
| 平均置信度 avg_confidence | 0.59 | **0.70** |

→ 父子块联动修复后，更多问题拿到有效上下文（31→38）；关键词路由从 24 词扩展到 ~50 词，4 个之前被误路由的劳动法问题（工时/产假/未成年人/招用）恢复走 RAG 路径。

---

## 二、端到端延迟评估

### 2.1 首字节延迟（最核心的体验指标）

| 阶段 | 延迟 | 根因 / 修复 |
|------|:---:|------|
| 优化前 | **26.5s 静默空白**（总 35.7s） | 流式生成走了 `CHAT_MODEL` 且未关思考模式 → 推理模型先耗 ~26s 在不可见的 reasoning 上 |
| 优化后 | **<1s**（首字节随内容/思考流式到达） | 流式改走 `GENERATION_LLM_MODEL` 并按需关闭思考；开启思考时 `reasoning_content` 实时推送前端可见 |

> 数据来源：Jaeger trace `3b8ae902…`（空白段 5.33s→31.88s）与修复后复测。

### 2.2 端到端总耗时（单请求，后台任务排空后实测）

| 问题 | 耗时 | 备注 |
|------|:---:|------|
| Q4 解除劳动合同情形 | 8.9s | 1 轮检索 |
| Q3 经济补偿金 | 9.2s | Agent 触发 2 轮检索（首轮置信度不足重检索） |
| Q1 试用期 | ~10.9s | 与后台记忆抽取并发时测，含资源竞争 |

> 注意：连环快速请求会与后台记忆抽取任务争抢 embedding 模型（实测 embedding 由 0.2s 抬到 1.6-2.8s）；真实用户单请求约 9-11s。

### 2.3 典型分环节耗时（单轮检索请求）

| 环节 | 耗时 | 说明 |
|------|:---:|------|
| 记忆改写 memory.rewrite | ~1.0s | 结合会话记忆改写问题（LLM） |
| Agent 检索 agent.run_parallel | 1.9~4.2s | 混合检索 + 相关性评估；低置信度会重检索一轮 |
| └ rerank（bge-reranker-base int8） | 1.4~1.6s | 5 候选 |
| 记忆抽取 memory.extract | ~1.0s | 后台，异步 |
| 答案生成 generation | 2~3s | 流式 |
| 合计 | **8.9s** | |

---

## 三、优化项逐项量化（before → after）

| # | 优化项 | 改动位置 | 效果 |
|---|--------|---------|------|
| 1 | 流式生成关思考 + 使用 generation 专用模型 | `chat_service.py` | 首字节 **26.5s → <1s** |
| 2 | Rerank 序列限长 512 → 256（`RERANKER_MAX_LENGTH`） | `retriever.py` | rerank **4.2s → 1.4s**（5 候选，-66%） |
| 3 | 评估 LLM 关思考 | `providers.py` | 相关性评估 **4.1s → 1.3~2.5s** |
| 4 | 启动预热（jieba/embedding/rerank/rewrite LLM） | `main.py` | 首请求冷启动 **4.4s → 0.27s** |
| 5 | 修复 SQLite 相对路径导致 BM25 稀疏检索失效 | 部署 | sparse 路由 **0 → 4**（BM25 加载 236 chunks） |
| 6 | 空检索提前 break | `graph.py` | 免去一轮无效重复检索 |
| 7 | 子任务拆分标记收紧 | `main_agent.py` | 减少单任务问题被误拆 |

---

## 四、结论

1. **质量**：修复路由与父子块联动后，RAG 核心路径 faithfulness（幻觉指标）median 由 0.80 升至 1.00、context_recall 达 100%；has_context 31→38、correct 31→35。
2. **延迟**：首字节 26.5s→<1s（量级改进）；rerank 与评估两个次热点分别 -66% 与 -40% 以上；稳态单请求约 9-11s。
3. **权衡**：开启思考模式可展示推理过程（前端实时滚动显示），但会把总耗时从 ~9s 抬到 ~24s（qwen3.7-max 思考生成成本），默认关闭。

---

## 五、复现方法

- 评估：`cd backend && python tests/run_ragas_full_eval.py`（依赖 `.env` 的 CHAT/Eval LLM 配置）
- 追踪：启动 `docker compose -f docker-compose.db.yml up`（含 Jaeger）→ 打开 http://localhost:16686 按服务 `first-rag-backend` 查看 span 耗时
- 压测：直接对 `/api/v1/chat?stream=true` 计时（注意后台记忆抽取的资源竞争）

## 六、相关文件

- 测试集：`backend/tests/testsets/labor_law_full.py`
- 评估脚本：`backend/tests/run_ragas_full_eval.py`、`backend/tests/rerun_ragas_on_saved.py`
- 原始结果：`backend/tests/ragas_full_eval_*.json`
- 本报告历史版本：`backend/tests/RAGAS_BASELINE_20260809.md`、`RAGAS_COMPARISON_20260809.md`
