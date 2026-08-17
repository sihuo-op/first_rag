"""
LangGraph ReAct Agent - Agentic RAG

核心思想：LLM 作为 Agent 核心，自主决定是否调用检索工具
流程：用户问题 → Agent 判断 → (需要检索) → 调用工具 → 生成答案
                           → (不需要) → 直接回答
"""

import time
from typing import Optional, List, Dict, Any, TypedDict
from langchain_core.language_models import BaseChatModel

from app.core.observability import get_tracer
from app.rag.steps import RetrieveTool, RewriteTool, GenerateTool, EvaluateTool

tracer = get_tracer("rag.graph")


# ============ 状态定义 ============

class AgentState(TypedDict):
    """
    Agent 状态定义
    """
    question: str
    answer: str
    documents: List[Dict]
    query_history: List[str]
    attempt_count: int
    max_attempts: int
    confidence: float
    execution_log: List[Dict]
    errors: List[str]


def create_initial_state(question: str, max_attempts: int = 4) -> AgentState:
    """
    创建初始状态
    """
    return {
        "question": question,
        "answer": "",
        "documents": [],
        "query_history": [question],
        "attempt_count": 0,
        "max_attempts": max_attempts,
        "confidence": 0.0,
        "execution_log": [],
        "errors": [],
        "_last_retrieved_query_idx": 0,  # expand 模式追踪已检索的查询位置
    }


# ============ RAG Graph 主类 ============

class RAGGraph:
    """
    Agentic RAG Graph - 基于状态机的迭代检索增强生成
    
    核心流程：
    1. 接收问题 → 2. 改写查询 → 3. 检索文档 → 4. 评估结果 → 
    5. (置信度不足) → 重新改写 → 检索 → ... → 6. 生成最终答案
    """

    def __init__(
        self,
        retriever,
        generation_llm: BaseChatModel,
        rewrite_llm: BaseChatModel,
        evaluation_llm: Optional[BaseChatModel] = None,
        max_attempts: int = 2,  # 减少迭代次数，从 4 次改为 2 次
        top_k: int = 8
    ):
        """
        初始化 RAG Graph
        
        Args:
            retriever: HybridRetriever 实例
            generation_llm: 生成答案用的 LLM
            rewrite_llm: 改写查询用的 LLM（低温度）
            evaluation_llm: 评估检索结果用的 LLM（可选，否则用规则评估）
            max_attempts: 最大迭代次数
            top_k: 每次检索返回的文档数
        """
        self.retriever = retriever
        self.generation_llm = generation_llm
        self.rewrite_llm = rewrite_llm
        self.evaluation_llm = evaluation_llm
        self.max_attempts = max_attempts
        self.top_k = top_k

        # 初始化工具
        self.retrieve_tool = RetrieveTool(retriever=retriever, top_k=top_k)
        self.rewrite_tool = RewriteTool(llm=rewrite_llm)
        self.generate_tool = GenerateTool(llm=generation_llm)
        self.evaluate_tool = EvaluateTool(llm=evaluation_llm)

    def run(self, question: str, generate_answer: bool = True) -> Dict[str, Any]:
        """运行完整的 Agentic RAG 流程"""
        with tracer.start_as_current_span("rag.run") as span:
            span.set_attribute("rag.question", question)
            span.set_attribute("rag.max_attempts", self.max_attempts)
            span.set_attribute("rag.top_k", self.top_k)
            return self._run_impl(question, generate_answer)

    def _run_impl(self, question: str, generate_answer: bool) -> Dict[str, Any]:
        """运行完整的 Agentic RAG 流程

        Returns:
            包含回答、检索文档、执行日志等信息的字典
        """
        start_time = time.time()
        
        # 初始化状态
        state = create_initial_state(question, self.max_attempts)
        state["retrieval_debug_info"] = []  # 保存每次检索的详细调试信息
        state["step_timings"] = []  # 保存每个阶段的用时
        state["candidate_documents"] = []
        state["best_documents"] = []
        state["best_confidence"] = -1.0

        while state["attempt_count"] < state["max_attempts"]:
            state["attempt_count"] += 1
            iteration_start = time.time()
            iteration_timing = {"attempt": state["attempt_count"], "steps": {}}
            state["_current_iteration_debug"] = {"attempt": state["attempt_count"]}

            try:
                # Step 1: 检索文档
                # 如果 query_history 有多个未检索的查询（expand 生成的），分别检索并合并
                current_query = state["query_history"][-1]
                all_queries_to_search = [current_query]
                
                # 如果是 expand 改写，query_history 中可能有多个新查询
                # 检查是否有未检索过的查询（从上次检索后新增的）
                last_retrieved_idx = state.get("_last_retrieved_query_idx", 0)
                new_queries = state["query_history"][last_retrieved_idx:]
                if len(new_queries) > 1:
                    all_queries_to_search = new_queries
                    print(f"[DEBUG] expand 模式：使用 {len(new_queries)} 个查询分别检索")
                
                # 用所有查询分别检索，合并结果
                all_documents = []
                seen_contents = set()  # 去重
                total_retrieve_time = 0
                
                for q in all_queries_to_search:
                    retrieve_start = time.time()
                    retrieve_result = self.retrieve_tool.execute({"current_query": q})
                    retrieve_time = time.time() - retrieve_start
                    total_retrieve_time += retrieve_time
                    
                    if retrieve_result.success:
                        for doc in retrieve_result.data.get("documents", []):
                            content_key = doc.get("content", "")[:100]
                            if content_key not in seen_contents:
                                seen_contents.add(content_key)
                                all_documents.append(doc)
                
                iteration_timing["steps"]["retrieve"] = round(total_retrieve_time, 3)
                if all_documents:
                    state["documents"] = all_documents
                    for doc in all_documents:
                        candidate_key = doc.get("content", "")[:100]
                        if candidate_key and all(candidate_key != item.get("content", "")[:100] for item in state["candidate_documents"]):
                            state["candidate_documents"].append(doc)
                state["_last_retrieved_query_idx"] = len(state["query_history"])
                
                # 保存检索调试信息
                if all_queries_to_search != [current_query]:
                    # expand 模式，构造合并后的调试信息
                    state["retrieval_debug_info"].append({
                        "attempt": state["attempt_count"],
                        "query": f"expand({len(all_queries_to_search)} queries)",
                        "queries": all_queries_to_search,
                        "retrieve_time": round(total_retrieve_time, 3),
                        "documents_found": len(all_documents),
                    })
                elif retrieve_result.success and retrieve_result.debug_info:
                    state["retrieval_debug_info"].append({
                        "attempt": state["attempt_count"],
                        "query": current_query,
                        "retrieve_time": round(total_retrieve_time, 3),
                        **retrieve_result.debug_info
                    })
                
                state["execution_log"].append({
                    "step": "retrieve",
                    "attempt": state["attempt_count"],
                    "query": current_query,
                    "queries_used": all_queries_to_search if len(all_queries_to_search) > 1 else None,
                    "documents_found": len(all_documents),
                    "time_s": round(total_retrieve_time, 3),
                    "reason": f"检索到 {len(all_documents)} 条相关文档" + (f"（使用 {len(all_queries_to_search)} 个查询）" if len(all_queries_to_search) > 1 else "")
                })
                if not all_documents:
                    state["errors"].append("检索失败：未获取到任何文档")
                    # 首轮即空说明语料无支持：本路径不会改写查询，重试同一查询结果确定相同，
                    # 纯浪费一次完整检索周期，直接跳出用当前（空）结果收尾
                    state["execution_log"].append({
                        "step": "retrieve",
                        "attempt": state["attempt_count"],
                        "query": current_query,
                        "documents_found": 0,
                        "time_s": round(total_retrieve_time, 3),
                        "reason": "检索未获取到任何文档，语料无支持，跳过无效重试"
                    })
                    iteration_timing["total_time"] = round(time.time() - iteration_start, 3)
                    state["step_timings"].append(iteration_timing)
                    break

                # Step 2: 评估检索结果（CRAG 风格）
                eval_start = time.time()
                eval_result = self.evaluate_tool.execute({
                    "original_question": question,
                    "documents": state["documents"]
                })
                eval_time = time.time() - eval_start
                iteration_timing["steps"]["evaluate"] = round(eval_time, 3)
                state["confidence"] = eval_result.data.get("confidence", 0.0)
                is_sufficient = eval_result.data.get("is_sufficient", False)
                grade = eval_result.data.get("grade", "ambiguous")  # CRAG 三级评估
                if grade != self.evaluate_tool.INCORRECT and state["documents"] and state["confidence"] >= state.get("best_confidence", -1.0):
                    state["best_documents"] = list(state["documents"])
                    state["best_confidence"] = state["confidence"]

                state["execution_log"].append({
                    "step": "evaluate",
                    "attempt": state["attempt_count"],
                    "confidence": state["confidence"],
                    "is_sufficient": is_sufficient,
                    "grade": grade,
                    "time_s": round(eval_time, 3),
                    "reason": eval_result.data.get("reason", "")
                })

                # 保存评估结果到当前迭代的调试信息
                if state.get("_current_iteration_debug"):
                    state["_current_iteration_debug"]["evaluation"] = {
                        "confidence": state["confidence"],
                        "is_sufficient": is_sufficient,
                        "grade": grade,
                        "reason": eval_result.data.get("reason", ""),
                        "suggestion": eval_result.data.get("suggestion", ""),
                        "time_s": round(eval_time, 3)
                    }

                # Step 3: 根据 CRAG 评估等级决定下一步策略
                should_stop_iteration = False
                print(f"\n[DEBUG] 评估等级: {grade}, 置信度: {state['confidence']}, 尝试次数: {state['attempt_count']}/{state['max_attempts']}")

                if grade == "correct" or state["attempt_count"] >= state["max_attempts"]:
                    # Correct：检索结果与问题高度相关，直接生成答案
                    # 或达到最大尝试次数，用当前结果生成
                    print(f"[DEBUG] 退出循环：{'评估通过(correct)' if grade == 'correct' else '达到最大尝试次数'}")
                    should_stop_iteration = True
                elif grade == "incorrect":
                    # Incorrect：检索结果与问题完全无关，必须改写查询重新检索
                    print(f"[DEBUG] 评估为 incorrect，必须改写查询重新检索...")
                    rewrite_start = time.time()
                    rewrite_result = self.rewrite_tool.execute({
                        "original_question": question,
                        "current_query": current_query,
                        "documents": state["documents"],
                        "query_history": state["query_history"],
                        "eval_grade": grade,
                        "eval_reason": eval_result.data.get("reason", "")
                    })
                    rewrite_time = time.time() - rewrite_start
                    iteration_timing["steps"]["rewrite"] = round(rewrite_time, 3)
                    
                    print(f"[DEBUG] 改写结果: success={rewrite_result.success}, message={rewrite_result.message}")

                    if rewrite_result.success:
                        queries = rewrite_result.data.get("queries", [])
                        rewrite_type = rewrite_result.data.get("rewrite_type", "improve")
                        
                        if rewrite_type == "expand" and len(queries) > 1:
                            # expand 模式：把所有同义词查询都加入 query_history
                            added_queries = []
                            for q in queries:
                                if q != current_query and q not in state["query_history"]:
                                    state["query_history"].append(q)
                                    added_queries.append(q)
                            
                            if added_queries:
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "original_query": current_query,
                                    "rewritten_queries": added_queries,
                                    "rewrite_type": rewrite_type,
                                    "grade": grade,
                                    "time_s": round(rewrite_time, 3),
                                    "reason": f"评估为 incorrect，expand 生成 {len(added_queries)} 个同义词查询"
                                })
                                print(f"[DEBUG] expand 生成 {len(added_queries)} 个同义词查询: {added_queries}")
                            else:
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "time_s": round(rewrite_time, 3),
                                    "reason": "expand 未生成有效的新查询，使用当前结果"
                                })
                                print(f"[DEBUG] expand 未生成有效的新查询，退出循环")
                                break
                        else:
                            # improve/decompose 模式：只取第一个查询
                            new_query = queries[0] if queries else current_query
                            print(f"[DEBUG] 新查询: {new_query}, 原查询: {current_query}")
                            
                            if new_query != current_query and new_query not in state["query_history"]:
                                state["query_history"].append(new_query)
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "original_query": current_query,
                                    "rewritten_query": new_query,
                                    "rewrite_type": rewrite_type,
                                    "grade": grade,
                                    "time_s": round(rewrite_time, 3),
                                    "reason": f"评估为 incorrect，改写查询重新检索"
                                })
                                print(f"[DEBUG] 查询改写成功，继续下一次迭代")
                            else:
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "time_s": round(rewrite_time, 3),
                                    "reason": "无法生成有效的新查询，使用当前结果"
                                })
                                print(f"[DEBUG] 无法生成有效的新查询，退出循环")
                                break
                    else:
                        state["errors"].append(f"查询改写失败: {rewrite_result.message}")
                        print(f"[DEBUG] 查询改写失败，退出循环")
                        break
                else:
                    # Ambiguous：检索结果部分相关，改写查询补充检索
                    print(f"[DEBUG] 评估为 ambiguous，尝试改写查询补充检索...")
                    rewrite_start = time.time()
                    rewrite_result = self.rewrite_tool.execute({
                        "original_question": question,
                        "current_query": current_query,
                        "documents": state["documents"],
                        "query_history": state["query_history"],
                        "eval_grade": grade,
                        "eval_reason": eval_result.data.get("reason", "")
                    })
                    rewrite_time = time.time() - rewrite_start
                    iteration_timing["steps"]["rewrite"] = round(rewrite_time, 3)
                    
                    print(f"[DEBUG] 改写结果: success={rewrite_result.success}, message={rewrite_result.message}")

                    if rewrite_result.success:
                        queries = rewrite_result.data.get("queries", [])
                        rewrite_type = rewrite_result.data.get("rewrite_type", "improve")
                        
                        if rewrite_type == "expand" and len(queries) > 1:
                            # expand 模式：把所有同义词查询都加入 query_history
                            added_queries = []
                            for q in queries:
                                if q != current_query and q not in state["query_history"]:
                                    state["query_history"].append(q)
                                    added_queries.append(q)
                            
                            if added_queries:
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "original_query": current_query,
                                    "rewritten_queries": added_queries,
                                    "rewrite_type": rewrite_type,
                                    "grade": grade,
                                    "time_s": round(rewrite_time, 3),
                                    "reason": f"评估为 ambiguous，expand 生成 {len(added_queries)} 个同义词查询"
                                })
                                print(f"[DEBUG] expand 生成 {len(added_queries)} 个同义词查询: {added_queries}")
                            else:
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "time_s": round(rewrite_time, 3),
                                    "reason": "expand 未生成有效的新查询，使用当前部分相关结果"
                                })
                                print(f"[DEBUG] expand 未生成有效的新查询，使用当前结果生成答案")
                                break
                        else:
                            # improve/decompose 模式：只取第一个查询
                            new_query = queries[0] if queries else current_query
                            print(f"[DEBUG] 新查询: {new_query}, 原查询: {current_query}")
                            
                            if new_query != current_query and new_query not in state["query_history"]:
                                state["query_history"].append(new_query)
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "original_query": current_query,
                                    "rewritten_query": new_query,
                                    "rewrite_type": rewrite_type,
                                    "grade": grade,
                                    "time_s": round(rewrite_time, 3),
                                    "reason": f"评估为 ambiguous，改写查询补充检索"
                                })
                                print(f"[DEBUG] 查询改写成功，继续下一次迭代")
                            else:
                                # 无法生成新的查询，用当前部分相关的结果生成答案
                                state["execution_log"].append({
                                    "step": "rewrite",
                                    "attempt": state["attempt_count"],
                                    "time_s": round(rewrite_time, 3),
                                    "reason": "无法生成有效的新查询，使用当前部分相关结果"
                                })
                                print(f"[DEBUG] 无法生成有效的新查询，使用当前结果生成答案")
                                break
                    else:
                        # 改写失败，用当前部分相关的结果生成答案
                        state["execution_log"].append({
                            "step": "rewrite",
                            "attempt": state["attempt_count"],
                            "time_s": round(rewrite_time, 3),
                            "reason": f"查询改写失败，使用当前部分相关结果: {rewrite_result.message}"
                        })
                        print(f"[DEBUG] 查询改写失败，使用当前结果生成答案")
                        break

            except Exception as e:
                state["errors"].append(f"迭代 {state['attempt_count']} 出错: {str(e)}")
                state["execution_log"].append({
                    "step": "error",
                    "attempt": state["attempt_count"],
                    "reason": str(e)
                })
                break

            # 保存当前迭代的用时信息
            iteration_timing["total_time"] = round(time.time() - iteration_start, 3)
            state["step_timings"].append(iteration_timing)
            if should_stop_iteration:
                break

        if state.get("best_documents"):
            state["documents"] = state["best_documents"]
        else:
            state["documents"] = []

        # Step 4: 生成最终答案
        if generate_answer:
            try:
                generate_start = time.time()
                generate_result = self.generate_tool.execute({
                    "original_question": question,
                    "documents": state["documents"]
                })
                generate_time = time.time() - generate_start
                state["generation_time"] = round(generate_time, 3)
                if generate_result.success:
                    state["answer"] = generate_result.data.get("answer", "")
                    llm_prompt = generate_result.data.get("prompt", "")
                    llm_response = generate_result.data.get("answer", "")
                    state["execution_log"].append({
                        "step": "generate",
                        "attempt": state["attempt_count"],
                        "time_s": round(generate_time, 3),
                        "llm_prompt": llm_prompt,
                        "llm_response": llm_response,
                        "reason": "生成最终答案"
                    })
                else:
                    state["answer"] = f"生成答案时出错: {generate_result.message}"
                    state["errors"].append(f"答案生成失败: {generate_result.message}")
            except Exception as e:
                state["answer"] = f"生成答案时发生错误: {str(e)}"
                state["errors"].append(f"答案生成异常: {str(e)}")
        else:
            state["answer"] = ""
            state["generation_time"] = 0
        
        elapsed_time = time.time() - start_time
        
        # 合并所有检索步骤的调试信息
        all_retrieval_steps = []
        chunks_by_type = {"small": 0, "medium": 0, "large": 0}
        rerank_used = False
        detail_info = {"dense_by_type": {}, "sparse_results": [], "merged_results": [], "deduped_results": [], "reranked_results": []}

        print(f"\n[DEBUG] retrieval_debug_info 包含 {len(state.get('retrieval_debug_info', []))} 次迭代")

        for attempt_info in state.get("retrieval_debug_info", []):
            print(f"[DEBUG] 处理第 {attempt_info.get('attempt')} 次迭代的调试信息")
            if "steps" in attempt_info:
                all_retrieval_steps.extend(attempt_info["steps"])
            if "chunks_by_type" in attempt_info:
                for key in chunks_by_type:
                    chunks_by_type[key] += attempt_info["chunks_by_type"].get(key, 0)
            if "rerank_used" in attempt_info:
                rerank_used = rerank_used or attempt_info["rerank_used"]
            if "detail" in attempt_info:
                    # 累积 dense_by_type 而不是覆盖
                    for chunk_type, items in attempt_info["detail"].get("dense_by_type", {}).items():
                        if chunk_type not in detail_info["dense_by_type"]:
                            detail_info["dense_by_type"][chunk_type] = []
                        detail_info["dense_by_type"][chunk_type].extend(items)
                    detail_info["sparse_results"].extend(attempt_info["detail"].get("sparse_results", []))
                    detail_info["merged_results"].extend(attempt_info["detail"].get("merged_results", []))
                    detail_info["deduped_results"].extend(attempt_info["detail"].get("deduped_results", []))
                    detail_info["reranked_results"].extend(attempt_info["detail"].get("reranked_results", []))

        print(f"[DEBUG] 最终 detail_info.merged_results 包含 {len(detail_info['merged_results'])} 条记录")
        
        # 打印阶段用时汇总
        print(f"\n{'='*50}")
        print(f"RAGGraph 执行完成，总耗时: {round(elapsed_time, 3)}s")
        print(f"{'='*50}")
        for timing in state.get("step_timings", []):
            print(f"  第 {timing['attempt']} 次迭代:")
            for step, duration in timing.get("steps", {}).items():
                print(f"    - {step}: {duration}s")
            print(f"    总计: {timing.get('total_time', 0)}s")
        print(f"  答案生成: {state.get('generation_time', 0)}s")
        print(f"{'='*50}\n")

        # 构建每次迭代的完整调试信息
        iterations = []
        for idx, debug_info in enumerate(state.get("retrieval_debug_info", [])):
            timing_info = state.get("step_timings", [])[idx] if idx < len(state.get("step_timings", [])) else {}
            steps_with_timing = debug_info.get("steps", [])
            
            iteration_entry = {
                "attempt": debug_info.get("attempt"),
                "query": debug_info.get("query"),
                "steps": steps_with_timing,
                "timing": timing_info,
                "detail": {
                    "dense_by_type": debug_info.get("detail", {}).get("dense_by_type", {}),
                    "sparse_results": debug_info.get("detail", {}).get("sparse_results", []),
                    "merged_results": debug_info.get("detail", {}).get("merged_results", []),
                    "deduped_results": debug_info.get("detail", {}).get("deduped_results", []),
                    "reranked_results": debug_info.get("detail", {}).get("reranked_results", [])
                }
            }
            iterations.append(iteration_entry)
        
        # 将 execution_log 按迭代分组
        grouped_execution_log = []
        for attempt in range(1, state["attempt_count"] + 1):
            attempt_log = {}
            for log_entry in state["execution_log"]:
                if log_entry.get("attempt") == attempt:
                    step = log_entry.get("step")
                    if step == "retrieve":
                        attempt_log["retrieve"] = log_entry
                        attempt_log["retrieve_time"] = log_entry.get("time_s", 0)
                    elif step == "evaluate":
                        attempt_log["evaluation"] = {
                            "confidence": log_entry.get("confidence"),
                            "is_sufficient": log_entry.get("is_sufficient"),
                            "reason": log_entry.get("reason"),
                            "suggestion": log_entry.get("suggestion", ""),
                            "time_s": log_entry.get("time_s", 0)  # 添加评估用时
                        }
                    elif step == "rewrite":
                        attempt_log["rewrite_type"] = log_entry.get("rewritten_query", "")
                        attempt_log["rewrite_reason"] = log_entry.get("reason")
                        attempt_log["rewrite_time"] = log_entry.get("time_s", 0)  # 添加改写用时
                    elif step == "generate":
                        # 生成步骤放在最后一次迭代
                        attempt_log["llm_prompt"] = log_entry.get("llm_prompt", "")
                        attempt_log["llm_response"] = log_entry.get("llm_response", "")
                        attempt_log["generation_time"] = log_entry.get("time_s", 0)  # 添加生成用时
            if attempt_log:
                grouped_execution_log.append(attempt_log)
        
        # 如果生成步骤不在任何迭代中，添加到最后
        generate_log = next((log for log in state["execution_log"] if log.get("step") == "generate"), None)
        if generate_log and (not grouped_execution_log or "llm_prompt" not in grouped_execution_log[-1]):
            if grouped_execution_log:
                grouped_execution_log[-1]["llm_prompt"] = generate_log.get("llm_prompt", "")
                grouped_execution_log[-1]["llm_response"] = generate_log.get("llm_response", "")
            else:
                grouped_execution_log.append({
                    "llm_prompt": generate_log.get("llm_prompt", ""),
                    "llm_response": generate_log.get("llm_response", "")
                })

        # 提取最终的 CRAG 评估等级和原因
        final_grade = "ambiguous"
        final_eval_reason = ""
        for log_entry in reversed(state["execution_log"]):
            if log_entry.get("step") == "evaluate":
                final_grade = log_entry.get("grade", "ambiguous")
                final_eval_reason = log_entry.get("reason", "")
                break

        return {
            "answer": state["answer"],
            "documents": state["documents"],
            "candidate_documents": state.get("candidate_documents", []),
            "query_history": state["query_history"],
            "attempt_count": state["attempt_count"],
            "max_attempts": state["max_attempts"],
            "confidence": state["confidence"],
            "grade": final_grade,
            "evaluation_reason": final_eval_reason,
            "execution_log": grouped_execution_log,  # 使用分组后的日志
            "errors": state["errors"],
            "elapsed_time": round(elapsed_time, 3),
            # 传统 RAG 格式的调试信息（保留兼容性）
            "retrieval_steps": all_retrieval_steps,
            "chunks_by_type": chunks_by_type,
            "rerank_used": rerank_used,
            "detail": detail_info,
            "llm_messages_count": state["attempt_count"] * 3,
            # 新增：每次迭代的完整调试信息
            "iterations": iterations,
        }


# ============ 便捷函数 ============

def build_rag_graph(
    retriever,
    generation_llm: BaseChatModel,
    rewrite_llm: BaseChatModel,
    evaluation_llm: Optional[BaseChatModel] = None,
    max_attempts: int = 2,
    top_k: int = 8
):
    """构建 RAG Graph（状态机模式）"""
    return RAGGraph(
        retriever=retriever,
        generation_llm=generation_llm,
        rewrite_llm=rewrite_llm,
        evaluation_llm=evaluation_llm,
        max_attempts=max_attempts,
        top_k=top_k
    )

