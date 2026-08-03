"""Optional LLM-as-judge support for RAG evaluations."""

from __future__ import annotations

from typing import Any


def judge_answer(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "enabled": False,
        "score": None,
        "reason": "LLM judge is disabled in the first evaluation implementation.",
    }
