"""Evaluation runners for retrieval, RAG tool, and end-to-end API layers."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from evals.dataset import EvalCase
from evals.metrics import (
    answer_rule_score,
    corpus_keyword_hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rewrite_hit_rate,
    summarize_case_scores,
)


def _case_query(case: EvalCase) -> str:
    if case.conversation:
        return str(case.conversation[-1].get("content") or case.query)
    return case.query


def _compact_debug(debug_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": debug_info.get("steps") or debug_info.get("retrieval_steps") or [],
        "dense_results": debug_info.get("dense_results"),
        "sparse_results": debug_info.get("sparse_results"),
        "rerank_used": debug_info.get("rerank_used", False),
        "chunks_by_type": debug_info.get("chunks_by_type", {}),
    }


def run_retrieval_cases(cases: list[EvalCase], top_k: int = 10) -> dict[str, Any]:
    from app.core.dependencies import get_retriever

    retriever = get_retriever()
    results = []
    for case in cases:
        query = _case_query(case)
        start = time.time()
        try:
            retrieved, debug_info = retriever.retrieve(query, top_k=top_k)
            latency = time.time() - start
            metrics = {
                "recall_at_5": recall_at_k(retrieved, case.expected, min(5, top_k)),
                "recall_at_10": recall_at_k(retrieved, case.expected, min(10, top_k)),
                "mrr": mrr(retrieved, case.expected),
                "ndcg_at_10": ndcg_at_k(retrieved, case.expected, min(10, top_k)),
                "keyword_hit_rate": corpus_keyword_hit_rate(retrieved, case.expected),
                "latency_seconds": round(latency, 4),
                "result_count": len(retrieved),
                "has_error": 0.0,
            }
            results.append({
                "id": case.id,
                "category": case.category,
                "query": query,
                "metrics": metrics,
                "retrieved": _compact_results(retrieved),
                "debug": _compact_debug(debug_info),
            })
        except Exception as exc:
            results.append(_error_case(case, query, exc, time.time() - start))

    summary = summarize_case_scores(results, ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "keyword_hit_rate"])
    return {"mode": "retrieval", "summary": summary, "cases": results}


def run_rag_tool_cases(cases: list[EvalCase]) -> dict[str, Any]:
    from app.agent.rag_qa_tool import RAGQATool
    from app.core.dependencies import get_retriever

    tool = RAGQATool(get_retriever())
    results = []
    for case in cases:
        query = _case_query(case)
        start = time.time()
        try:
            output = tool.execute(query)
            latency = time.time() - start
            documents = output.get("documents") or []
            metrics = {
                "answer_rule_score": answer_rule_score(output.get("answer", ""), case.expected),
                "context_hit_rate": corpus_keyword_hit_rate(documents, case.expected),
                "confidence": float(output.get("confidence") or 0.0),
                "latency_seconds": round(float(output.get("elapsed_time") or latency), 4),
                "generation_time": round(float(output.get("generation_time") or 0.0), 4),
                "has_error": 1.0 if output.get("errors") else 0.0,
            }
            results.append({
                "id": case.id,
                "category": case.category,
                "query": query,
                "metrics": metrics,
                "answer": output.get("answer", ""),
                "retrieved": _compact_results(documents),
                "debug": {
                    "query_history": output.get("query_history", []),
                    "grade": output.get("grade"),
                    "evaluation_reason": output.get("evaluation_reason"),
                    "errors": output.get("errors", []),
                    "retrieval_steps": output.get("retrieval_steps", []),
                },
            })
        except Exception as exc:
            results.append(_error_case(case, query, exc, time.time() - start))

    summary = summarize_case_scores(results, ["answer_rule_score", "context_hit_rate", "confidence"])
    return {"mode": "rag", "summary": summary, "cases": results}


async def run_e2e_api_cases(
    cases: list[EvalCase],
    base_url: str,
    username: str = "admin",
    password: str = "admin123",
    timeout: float = 180.0,
) -> dict[str, Any]:
    results = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        token = await _login(client, username, password)
        headers = {"Authorization": f"Bearer {token}"}
        for case in cases:
            start = time.time()
            try:
                output = await _run_case_conversation(client, headers, case)
                latency = time.time() - start
                final_response = output["final_response"]
                debug_info = final_response.get("debug_info") or {}
                agentic_info = final_response.get("agentic_info") or debug_info.get("agentic_info") or {}
                memory_info = debug_info.get("memory_info") or {}
                decomposed_tasks = agentic_info.get("decomposed_tasks") or []
                sub_tasks = agentic_info.get("sub_tasks") or []
                expected_intent_count = int(case.expected.get("expected_intent_count") or 0)
                multi_task_success = _multi_task_success(expected_intent_count, decomposed_tasks, sub_tasks)
                metrics = {
                    "answer_rule_score": answer_rule_score(final_response.get("answer", ""), case.expected),
                    "context_hit_rate": corpus_keyword_hit_rate(final_response.get("retrieved_chunks") or [], case.expected),
                    "memory_rewrite_hit_rate": rewrite_hit_rate(memory_info.get("standalone_query", ""), case.expected),
                    "multi_task_success_rate": multi_task_success,
                    "latency_seconds": round(float(final_response.get("process_time") or latency), 4),
                    "has_error": 1.0 if any(task.get("errors") for task in sub_tasks) else 0.0,
                }
                results.append({
                    "id": case.id,
                    "category": case.category,
                    "query": _case_query(case),
                    "metrics": metrics,
                    "answer": final_response.get("answer", ""),
                    "conversation_id": final_response.get("conversation_id"),
                    "retrieved": _compact_results(final_response.get("retrieved_chunks") or []),
                    "debug": {
                        "memory_info": memory_info,
                        "decomposed_tasks": decomposed_tasks,
                        "sub_task_count": len(sub_tasks),
                        "mode": debug_info.get("mode") or agentic_info.get("mode"),
                    },
                })
            except Exception as exc:
                results.append(_error_case(case, _case_query(case), exc, time.time() - start))

    summary = summarize_case_scores(results, ["answer_rule_score", "context_hit_rate", "memory_rewrite_hit_rate", "multi_task_success_rate"])
    return {"mode": "e2e", "summary": summary, "cases": results}


def run_e2e_api_cases_sync(cases: list[EvalCase], base_url: str, username: str = "admin", password: str = "admin123") -> dict[str, Any]:
    return asyncio.run(run_e2e_api_cases(cases, base_url, username, password))


async def _login(client: httpx.AsyncClient, username: str, password: str) -> str:
    response = await client.post("/api/v1/auth/login", json={"username": username, "password": password})
    response.raise_for_status()
    data = response.json()
    return data["access_token"]


async def _run_case_conversation(client: httpx.AsyncClient, headers: dict[str, str], case: EvalCase) -> dict[str, Any]:
    conversation_id = None
    final_response = None
    turns = case.conversation or [{"role": "user", "content": case.query}]
    for turn in turns:
        if turn.get("role") != "user":
            continue
        response = await client.post(
            "/api/v1/chat",
            headers=headers,
            json={"query": turn.get("content", ""), "conversation_id": conversation_id, "use_rag": True, "stream": False},
        )
        response.raise_for_status()
        final_response = response.json()
        conversation_id = final_response.get("conversation_id")
    if final_response is None:
        raise ValueError("case has no user turn")
    return {"final_response": final_response}


def _multi_task_success(expected_count: int, decomposed_tasks: list[dict[str, Any]], sub_tasks: list[dict[str, Any]]) -> float:
    if expected_count <= 1:
        return 1.0
    observed = len(decomposed_tasks) or len(sub_tasks)
    if observed < min(expected_count, 2):
        return 0.0
    if any(task.get("errors") for task in sub_tasks):
        return 0.0
    return 1.0


def _compact_results(results: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    compact = []
    for result in results[:limit]:
        content = str(result.get("content") or "")
        compact.append({
            "id": result.get("id"),
            "document_id": result.get("document_id") or (result.get("metadata") or {}).get("document_id"),
            "score": result.get("score") or result.get("rerank_score") or result.get("rrf_score"),
            "content_preview": content[:240],
        })
    return compact


def _error_case(case: EvalCase, query: str, exc: Exception, latency: float) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "query": query,
        "metrics": {"has_error": 1.0, "latency_seconds": round(latency, 4)},
        "error": f"{type(exc).__name__}: {exc}",
    }
