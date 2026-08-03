"""Metrics for enterprise RAG evaluation."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


def normalize_text(text: Any) -> str:
    return str(text or "").lower().strip()


def result_content(result: dict[str, Any]) -> str:
    return normalize_text(result.get("content") or result.get("text") or result.get("page_content"))


def result_document_id(result: dict[str, Any]) -> Any:
    if "document_id" in result:
        return result.get("document_id")
    metadata = result.get("metadata") or {}
    return metadata.get("document_id")


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    normalized = normalize_text(text)
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits / len(keywords)


def is_relevant(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    must_retrieve = expected.get("must_retrieve") or {}
    expected_doc_ids = {str(item) for item in must_retrieve.get("document_ids") or []}
    if expected_doc_ids:
        document_id = result_document_id(result)
        if document_id is not None and str(document_id) in expected_doc_ids:
            return True

    keywords = must_retrieve.get("content_keywords") or []
    if keywords:
        return any(normalize_text(keyword) in result_content(result) for keyword in keywords)
    return bool(result_content(result))


def relevance_score(result: dict[str, Any], expected: dict[str, Any]) -> float:
    must_retrieve = expected.get("must_retrieve") or {}
    keywords = must_retrieve.get("content_keywords") or []
    score = keyword_hit_rate(result_content(result), keywords) if keywords else 0.0

    expected_doc_ids = {str(item) for item in must_retrieve.get("document_ids") or []}
    document_id = result_document_id(result)
    if expected_doc_ids and document_id is not None and str(document_id) in expected_doc_ids:
        score = max(score, 1.0)
    return score


def recall_at_k(results: list[dict[str, Any]], expected: dict[str, Any], k: int) -> float:
    if not ((expected.get("must_retrieve") or {}).get("content_keywords") or (expected.get("must_retrieve") or {}).get("document_ids")):
        return 1.0
    return 1.0 if any(is_relevant(result, expected) for result in results[:k]) else 0.0


def mrr(results: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[dict[str, Any]], expected: dict[str, Any], k: int) -> float:
    gains = [relevance_score(result, expected) for result in results[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def corpus_keyword_hit_rate(results: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    keywords = (expected.get("must_retrieve") or {}).get("content_keywords") or []
    combined = "\n".join(result_content(result) for result in results)
    return keyword_hit_rate(combined, keywords)


def answer_rule_score(answer: str, expected: dict[str, Any]) -> float:
    must_include = expected.get("answer_must_include") or []
    must_not_include = expected.get("answer_must_not_include") or []
    normalized_answer = normalize_text(answer)

    include_score = keyword_hit_rate(normalized_answer, must_include)
    if must_not_include:
        forbidden_hits = sum(1 for item in must_not_include if normalize_text(item) in normalized_answer)
        forbidden_score = 1.0 - forbidden_hits / len(must_not_include)
    else:
        forbidden_score = 1.0

    if not must_include:
        return round(forbidden_score, 4)
    return round((include_score * 0.7) + (forbidden_score * 0.3), 4)


def rewrite_hit_rate(standalone_query: str, expected: dict[str, Any]) -> float:
    return keyword_hit_rate(standalone_query, expected.get("standalone_query_keywords") or [])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_case_scores(case_results: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(result.get("metrics", {}).get(key, 0.0)) for result in case_results]
        summary[key] = round(mean(values), 4) if values else 0.0
    errors = [result for result in case_results if result.get("error") or result.get("metrics", {}).get("has_error")]
    latencies = [float(result.get("metrics", {}).get("latency_seconds", 0.0)) for result in case_results]
    summary["case_count"] = len(case_results)
    summary["error_rate"] = round(len(errors) / len(case_results), 4) if case_results else 0.0
    summary["avg_latency_seconds"] = round(mean(latencies), 4) if latencies else 0.0
    summary["p95_latency_seconds"] = round(percentile(latencies, 0.95), 4) if latencies else 0.0
    return summary


def check_thresholds(summary: dict[str, Any], thresholds: dict[str, Any], section: str) -> dict[str, Any]:
    section_thresholds = thresholds.get(section) or {}
    checks = []
    passed = True
    for key, threshold in section_thresholds.items():
        if key.endswith("_min"):
            metric_key = key[:-4]
            value = float(summary.get(metric_key, 0.0))
            ok = value >= float(threshold)
        elif key.endswith("_max"):
            metric_key = key[:-4]
            value = float(summary.get(metric_key, 0.0))
            ok = value <= float(threshold)
        else:
            metric_key = key
            value = float(summary.get(metric_key, 0.0))
            ok = value >= float(threshold)
        passed = passed and ok
        checks.append({"metric": metric_key, "value": value, "threshold": threshold, "passed": ok})
    return {"passed": passed, "checks": checks}
