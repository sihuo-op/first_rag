import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.agent.rag_qa_tool import RAGQATool
from app.core.observability import get_tracer
from app.llm.providers import get_generation_llm, invoke_llm_threadsafe
from app.rag.graph import RAGGraph

tracer = get_tracer("agent")


class MainAgent:
    """
    主 Agent 编排器 - 直接路由到知识库工具，不使用 ReAct 循环

    核心逻辑：
    1. 判断问题是否涉及劳动法相关知识
    2. 如果是，直接调用 knowledge_qa 工具一次
    3. 返回工具结果，不做额外决策

    优点：
    - 避免 LLM 思考耗时（节省约 15 秒）
    - 保证只调用一次工具，不会死循环
    - 执行流程更清晰、可预测
    """

    # 劳动法相关关键词
    LABOR_LAW_KEYWORDS = [
        # 基础术语
        "劳动", "合同", "用人单位", "劳动者", "雇主", "雇佣",
        # 工时休假
        "工时", "工作时间", "延长工作时间", "休息日", "休假",
        "年休假", "法定节假日", "夜班", "加班",
        # 合同管理
        "劳动合同", "试用期", "解除合同", "辞退", "离职", "辞职",
        "劳务派遣", "竞业限制", "集体合同",
        # 工资福利
        "工资", "加班费", "最低工资", "经济补偿", "补偿", "工龄",
        # 特殊保护
        "女职工", "孕期", "产期", "哺乳期", "产假", "生育",
        "未成年人", "童工", "招用", "工伤", "职业病", "医疗期", "患病",
        # 社保福利
        "社保", "社会保险", "公积金", "五险一金", "退休", "失业",
        # 争议处理
        "劳动仲裁", "仲裁", "调解", "诉讼",
        # 法规名称
        "劳动合同法", "劳动法",
    ]

    def __init__(
        self,
        llm,
        retriever,
        rewrite_llm=None,
        evaluation_llm=None,
        max_attempts: int = 2,
        top_k: int = 4
    ):
        """
        初始化主 Agent 编排器

        Args:
            llm: 主 LLM 模型
            retriever: 检索器
            rewrite_llm: 改写查询用的 LLM（可选）
            evaluation_llm: 评估检索结果用的 LLM（可选，默认使用主 LLM）
            max_attempts: RAGGraph 内部最大迭代次数
            top_k: 每次检索返回的文档数
        """
        self.llm = llm
        self.retriever = retriever
        self.rewrite_llm = rewrite_llm or llm
        self.evaluation_llm = evaluation_llm or llm  # 默认使用主 LLM 做评估
        self.max_attempts = max_attempts
        self.top_k = top_k

    def _is_labor_law_question(self, question: str) -> bool:
        """
        判断问题是否涉及劳动法相关知识

        Args:
            question: 用户问题

        Returns:
            bool: 是否为劳动法相关问题
        """
        question_lower = question.lower()
        return any(keyword.lower() in question_lower for keyword in self.LABOR_LAW_KEYWORDS)

    def run(self, question: str) -> Dict[str, Any]:
        """
        运行主 Agent 编排器

        Args:
            question: 用户问题

        Returns:
            Dict: 包含答案、调试信息等的结果
        """
        start_time = time.time()

        rag_result = None

        # 判断问题类型
        if self._is_labor_law_question(question):
            # 劳动法相关问题，直接调用知识库工具
            rag_graph = RAGGraph(
                retriever=self.retriever,
                generation_llm=self.llm,
                rewrite_llm=self.rewrite_llm,
                evaluation_llm=self.evaluation_llm,  # 启用 LLM 语义评估
                max_attempts=self.max_attempts,
                top_k=self.top_k
            )
            rag_result = rag_graph.run(question)
            answer = rag_result.get("answer", "知识库中未找到相关信息")
            retrieved = True
            tool_calls = [{"tool": "knowledge_qa", "args": {"question": question}}]
        else:
            # 非劳动法问题，直接回答
            answer = self._direct_answer(question)
            retrieved = False
            tool_calls = []

        elapsed_time = time.time() - start_time

        # 打印执行汇总
        self._print_summary(elapsed_time, tool_calls, rag_result)

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieved": retrieved,
            "elapsed_time": round(elapsed_time, 3),
            "messages": [],
            # 从 RAGGraph 获取详细信息
            "documents": rag_result.get("documents", []) if rag_result else [],
            "query_history": rag_result.get("query_history", []) if rag_result else [],
            "attempt_count": rag_result.get("attempt_count", 0) if rag_result else 0,
            "max_attempts": rag_result.get("max_attempts", self.max_attempts) if rag_result else self.max_attempts,
            "confidence": rag_result.get("confidence", 0.0) if rag_result else 0.0,
            "grade": rag_result.get("grade", "unknown") if rag_result else "unknown",
            "evaluation_reason": rag_result.get("evaluation_reason", "") if rag_result else "",
            "execution_log": rag_result.get("execution_log", []) if rag_result else [],
            "errors": rag_result.get("errors", []) if rag_result else [],
            "retrieval_steps": rag_result.get("retrieval_steps", []) if rag_result else [],
            "chunks_by_type": rag_result.get("chunks_by_type", {"small": 0, "medium": 0, "large": 0}) if rag_result else {"small": 0, "medium": 0, "large": 0},
            "rerank_used": rag_result.get("rerank_used", False) if rag_result else False,
            "detail": rag_result.get("detail", {}) if rag_result else {},
            "iterations": rag_result.get("iterations", []) if rag_result else [],
            "debug_info": rag_result.get("debug_info", {}) if rag_result else {},
            "step_timings": rag_result.get("step_timings", []) if rag_result else [],
            "generation_time": rag_result.get("generation_time", 0) if rag_result else 0,
        }

    async def run_parallel(
        self,
        question: str,
        max_subtasks: int = 3,
        max_concurrency: int = 3,
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """运行主编排器：拆分复杂问题，并行调用 RAG QA 工具。"""
        with tracer.start_as_current_span("agent.run_parallel") as span:
            span.set_attribute("agent.question", question)
            return await self._run_parallel_impl(question, max_subtasks, max_concurrency, generate_answer)

    async def _run_parallel_impl(
        self,
        question: str,
        max_subtasks: int,
        max_concurrency: int,
        generate_answer: bool,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if not self._is_labor_law_question(question):
            answer = self._direct_answer(question)
            elapsed_time = time.time() - start_time
            return {
                "answer": answer,
                "tool_calls": [],
                "retrieved": False,
                "elapsed_time": round(elapsed_time, 3),
                "messages": [],
                "documents": [],
                "query_history": [],
                "attempt_count": 0,
                "max_attempts": self.max_attempts,
                "confidence": 0.0,
                "grade": "unknown",
                "evaluation_reason": "",
                "execution_log": [],
                "errors": [],
                "retrieval_steps": [],
                "chunks_by_type": {"small": 0, "medium": 0, "large": 0},
                "rerank_used": False,
                "detail": {},
                "iterations": [],
                "debug_info": {},
                "step_timings": [],
                "generation_time": 0,
                "decomposed_tasks": [],
                "sub_tasks": [],
                "mode": "parallel_rag_tools",
            }

        decomposed_tasks = self._decompose_question(question, max_subtasks) if self._should_decompose_question(question) else [{"id": 1, "question": question}]
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                tool = RAGQATool(
                    retriever=self.retriever,
                    max_attempts=self.max_attempts,
                    top_k=self.top_k,
                )
                try:
                    return await asyncio.to_thread(tool.execute, task["question"], task["id"], generate_answer)
                except Exception as e:
                    return {
                        "task_id": task["id"],
                        "question": task["question"],
                        "answer": f"子任务执行失败: {e!s}",
                        "tool": "rag_qa",
                        "args": {"question": task["question"]},
                        "documents": [],
                        "query_history": [task["question"]],
                        "execution_log": [],
                        "retrieval_steps": [],
                        "iterations": [],
                        "detail": {},
                        "confidence": 0.0,
                        "grade": "error",
                        "evaluation_reason": str(e),
                        "elapsed_time": 0,
                        "errors": [str(e)],
                        "chunks_by_type": {"small": 0, "medium": 0, "large": 0},
                        "rerank_used": False,
                        "step_timings": [],
                        "generation_time": 0,
                    }

        sub_tasks = await asyncio.gather(*[run_one(task) for task in decomposed_tasks])
        sub_tasks = sorted(sub_tasks, key=lambda item: item.get("task_id", 0))

        if not generate_answer:
            answer = ""
        elif len(sub_tasks) == 1:
            answer = sub_tasks[0].get("answer", "")
        else:
            answer = self._merge_tool_results(question, sub_tasks)

        elapsed_time = time.time() - start_time
        tool_calls = [
            {
                "tool": item.get("tool", "rag_qa"),
                "args": item.get("args", {"question": item.get("question", "")}),
                "task_id": item.get("task_id"),
                "status": "error" if item.get("errors") else "success",
            }
            for item in sub_tasks
        ]

        documents = []
        candidate_documents = []
        query_history = []
        execution_log = []
        retrieval_steps = []
        iterations = []
        step_timings = []
        errors = []
        detail = {"sub_tasks": []}
        chunks_by_type = {"small": 0, "medium": 0, "large": 0}
        rerank_used = False
        confidence_values = []

        for item in sub_tasks:
            documents.extend(item.get("documents", []))
            candidate_documents.extend(item.get("candidate_documents", item.get("documents", [])))
            query_history.extend(item.get("query_history", []))
            execution_log.extend(item.get("execution_log", []))
            retrieval_steps.extend(item.get("retrieval_steps", []))
            iterations.extend(item.get("iterations", []))
            step_timings.extend(item.get("step_timings", []))
            errors.extend(item.get("errors", []))
            detail["sub_tasks"].append({"task_id": item.get("task_id"), "detail": item.get("detail", {})})
            item_chunks = item.get("chunks_by_type", {})
            for key in chunks_by_type:
                chunks_by_type[key] += item_chunks.get(key, 0)
            rerank_used = rerank_used or item.get("rerank_used", False)
            confidence_values.append(item.get("confidence", 0.0))

        confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0

        self._print_parallel_summary(elapsed_time, sub_tasks)

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieved": bool(sub_tasks),
            "elapsed_time": round(elapsed_time, 3),
            "messages": [],
            "documents": documents,
            "candidate_documents": candidate_documents,
            "query_history": query_history,
            "attempt_count": max((len(item.get("query_history", [])) for item in sub_tasks), default=0),
            "max_attempts": self.max_attempts,
            "confidence": confidence,
            "grade": "multi" if len(sub_tasks) > 1 else sub_tasks[0].get("grade", "unknown"),
            "evaluation_reason": "多子任务并行执行" if len(sub_tasks) > 1 else sub_tasks[0].get("evaluation_reason", ""),
            "execution_log": execution_log,
            "errors": errors,
            "retrieval_steps": retrieval_steps,
            "chunks_by_type": chunks_by_type,
            "rerank_used": rerank_used,
            "detail": detail,
            "iterations": iterations,
            "debug_info": {},
            "step_timings": step_timings,
            "generation_time": sum(item.get("generation_time", 0) for item in sub_tasks),
            "decomposed_tasks": decomposed_tasks,
            "sub_tasks": sub_tasks,
            "mode": "parallel_rag_tools",
        }

    def _decompose_question(self, question: str, max_subtasks: int) -> List[Dict[str, Any]]:
        with tracer.start_as_current_span("agent.decompose") as span:
            span.set_attribute("agent.question", question)
            return self._decompose_question_impl(question, max_subtasks)

    def _decompose_question_impl(self, question: str, max_subtasks: int) -> List[Dict[str, Any]]:
        prompt = f"""你是一个任务拆分器。请判断用户问题是否包含多个独立的劳动法咨询任务。

要求：
1. 如果是单一问题，只返回 1 个任务
2. 如果包含多个独立问题，拆成最多 {max_subtasks} 个可独立检索的子问题
3. 每个子问题必须保留完整语义，不能只返回关键词
4. 只返回 JSON 数组，不要解释

返回格式：
[
  {{"id": 1, "question": "子问题1"}},
  {{"id": 2, "question": "子问题2"}}
]

用户问题：{question}"""
        try:
            response = invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()
            json_match = re.search(r'\[[\s\S]*\]', content)
            raw_tasks = json.loads(json_match.group() if json_match else content)
            tasks = []
            for idx, item in enumerate(raw_tasks[:max_subtasks], start=1):
                sub_question = str(item.get("question", "")).strip() if isinstance(item, dict) else str(item).strip()
                if sub_question:
                    tasks.append({"id": idx, "question": sub_question})
            return tasks or [{"id": 1, "question": question}]
        except Exception as e:
            print(f"[MainAgent] 任务拆分失败，回退单任务: {e}")
            return [{"id": 1, "question": question}]

    def _merge_tool_results(self, original_question: str, sub_tasks: List[Dict[str, Any]]) -> str:
        with tracer.start_as_current_span("agent.merge") as span:
            span.set_attribute("agent.subtask_count", len(sub_tasks))
            return self._merge_tool_results_impl(original_question, sub_tasks)

    def _merge_tool_results_impl(self, original_question: str, sub_tasks: List[Dict[str, Any]]) -> str:
        result_text = "\n\n".join(
            f"子问题{item.get('task_id')}: {item.get('question')}\n回答: {item.get('answer', '')}"
            for item in sub_tasks
        )
        prompt = f"""你是专业的劳动法知识助手。请基于多个子问题的 RAG 回答，汇总回答用户的原始问题。

原始问题：{original_question}

子问题回答：
{result_text}

要求：
1. 按原始问题的顺序组织答案
2. 只使用子问题回答中已有的信息
3. 如果某个子问题没有找到依据，要明确说明
4. 不要添加未被参考文档支撑的新法律解释

汇总答案："""
        try:
            merge_llm = get_generation_llm()
            response = invoke_llm_threadsafe(merge_llm, [HumanMessage(content=prompt)])
            return response.content.strip() if hasattr(response, "content") else str(response).strip()
        except Exception as e:
            print(f"[MainAgent] 汇总答案失败，使用拼接兜底: {e}")
            return self._fallback_merge_answers(sub_tasks)

    def _fallback_merge_answers(self, sub_tasks: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"{idx}. {item.get('question')}\n{item.get('answer', '')}"
            for idx, item in enumerate(sub_tasks, start=1)
        )

    def _print_parallel_summary(self, elapsed_time: float, sub_tasks: List[Dict[str, Any]]):
        print(f"\n{'='*60}")
        print("Parallel RAG Tools 执行完成")
        print(f"{'='*60}")
        print(f"  总耗时: {round(elapsed_time, 3)}s")
        print(f"  子任务数: {len(sub_tasks)}")
        for item in sub_tasks:
            print(f"  - 任务 {item.get('task_id')}: {round(item.get('elapsed_time', 0), 3)}s")
        print(f"{'='*60}\n")

    def _should_decompose_question(self, question: str) -> bool:
        # 只保留强多任务信号："和/及/、"等连词在单任务复合问（如"工资、加班费怎么算"）中
        # 同样高频，误触发会白付一次 LLM decompose 调用（~0.8s）+ 多路检索
        multi_task_markers = [
            "并且", "同时", "另外", "还有", "以及", "分别", "一方面", "另一方面",
            "第一", "第二", "1.", "2.", "；", ";"
        ]
        question_mark_count = question.count("？") + question.count("?")
        return question_mark_count > 1 or any(marker in question for marker in multi_task_markers)

    def _direct_answer(self, question: str) -> str:
        with tracer.start_as_current_span("agent.direct_answer") as span:
            span.set_attribute("agent.question", question)
            return self._direct_answer_impl(question)

    def _direct_answer_impl(self, question: str) -> str:
        """
        直接回答非知识库问题

        Args:
            question: 用户问题

        Returns:
            str: 直接回答内容
        """
        # 识别常见的非知识库问题类型
        greetings = ["你好", "您好", "嗨", "hello", "hi"]
        thanks = ["谢谢", "感谢", "辛苦了"]
        about = ["你是谁", "你叫什么", "自我介绍", "什么是"]

        question_lower = question.lower()

        if any(g in question_lower for g in greetings):
            return "您好！我是专业的劳动法知识助手，请问有什么可以帮助您的？"
        elif any(t in question_lower for t in thanks):
            return "不客气！如果您还有其他问题，随时可以问我。"
        elif any(a in question_lower for a in about):
            return "我是一个基于检索增强生成技术的劳动法知识问答助手，可以帮助您解答关于劳动合同、工资、加班、社保等方面的问题。"
        else:
            # 对于其他非劳动法问题，使用 LLM 直接回答
            try:
                response = invoke_llm_threadsafe(self.llm, question)
                return response.content if hasattr(response, "content") else str(response)
            except Exception:
                return "这个问题我不太清楚，请问您有关于劳动法方面的问题吗？"

    def _print_summary(self, elapsed_time: float, tool_calls: List[Dict[str, Any]], rag_result: Optional[Dict[str, Any]] = None):
        """
        打印执行汇总信息

        Args:
            elapsed_time: 总耗时
            tool_calls: 工具调用列表
            rag_result: RAGGraph 执行结果
        """
        rag_result = rag_result or {}
        step_timings = rag_result.get("step_timings", [])
        generation_time = rag_result.get("generation_time", 0)
        elapsed_time_rag = rag_result.get("elapsed_time", 0)
        tool_call_count = len(tool_calls)

        print(f"\n{'='*60}")
        print("MainAgent 执行完成")
        print(f"{'='*60}")
        print(f"  总耗时: {round(elapsed_time, 3)}s")
        print(f"  工具调用次数: {tool_call_count} 次")
        print(f"  RAGGraph 耗时: {round(elapsed_time_rag, 3)}s")
        print(f"  其他耗时: {round(elapsed_time - elapsed_time_rag, 3)}s")
        print(f"{'='*60}")

        if step_timings:
            for idx, timing in enumerate(step_timings):
                print(f"  迭代 {idx+1}:")
                for step, duration in timing.get("steps", {}).items():
                    print(f"    - {step}: {duration}s")
                print(f"    总计: {timing.get('total_time', 0)}s")

        print(f"  答案生成: {generation_time}s")
        print(f"{'='*60}\n")
