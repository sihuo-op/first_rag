"""
Agent 可调用的单问题 RAG QA 工具。
"""

from typing import Any, ClassVar, Dict

from app.llm.providers import get_evaluation_llm, get_generation_llm, get_rewrite_llm
from app.rag.graph import RAGGraph

# ============ RAG QA 工具 ============

class RAGQATool:
    """单问题 RAG 问答工具"""
    name: ClassVar[str] = "rag_qa"

    def __init__(self, retriever, max_attempts: int = 2, top_k: int = 4):
        self.retriever = retriever
        self.max_attempts = max_attempts
        self.top_k = top_k

    def execute(self, question: str, task_id: int = 1, generate_answer: bool = True) -> Dict[str, Any]:

        generation_llm = get_generation_llm()
        rewrite_llm = get_rewrite_llm()
        evaluation_llm = get_evaluation_llm()
        rag_graph = RAGGraph(
            retriever=self.retriever,
            generation_llm=generation_llm,
            rewrite_llm=rewrite_llm,
            evaluation_llm=evaluation_llm,
            max_attempts=self.max_attempts,
            top_k=self.top_k,
        )
        rag_result = rag_graph.run(question, generate_answer=generate_answer)

        return {
            "task_id": task_id,
            "question": question,
            "answer": rag_result.get("answer", ""),
            "tool": self.name,
            "args": {"question": question},
            "documents": rag_result.get("documents", []),
            "candidate_documents": rag_result.get("candidate_documents", []),
            "query_history": rag_result.get("query_history", []),
            "execution_log": rag_result.get("execution_log", []),
            "retrieval_steps": rag_result.get("retrieval_steps", []),
            "iterations": rag_result.get("iterations", []),
            "detail": rag_result.get("detail", {}),
            "confidence": rag_result.get("confidence", 0.0),
            "grade": rag_result.get("grade", "unknown"),
            "evaluation_reason": rag_result.get("evaluation_reason", ""),
            "elapsed_time": rag_result.get("elapsed_time", 0),
            "errors": rag_result.get("errors", []),
            "chunks_by_type": rag_result.get("chunks_by_type", {"small": 0, "medium": 0, "large": 0}),
            "rerank_used": rag_result.get("rerank_used", False),
            "step_timings": rag_result.get("step_timings", []),
            "generation_time": rag_result.get("generation_time", 0),
        }
