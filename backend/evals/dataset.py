"""Dataset loading and validation for RAG evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = Path(__file__).with_name("golden_labor_law.jsonl")


@dataclass
class EvalCase:
    id: str
    category: str
    query: str
    expected: dict[str, Any]
    conversation: list[dict[str, Any]] | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], line_no: int | None = None) -> "EvalCase":
        location = f"line {line_no}: " if line_no else ""
        for key in ("id", "category", "expected"):
            if key not in raw:
                raise ValueError(f"{location}missing required field: {key}")

        query = str(raw.get("query") or "").strip()
        conversation = raw.get("conversation")
        if not query and not conversation:
            raise ValueError(f"{location}case must provide query or conversation")
        if conversation is not None and not isinstance(conversation, list):
            raise ValueError(f"{location}conversation must be a list")

        expected = raw.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"{location}expected must be an object")

        return cls(
            id=str(raw["id"]),
            category=str(raw["category"]),
            query=query,
            conversation=conversation,
            expected=expected,
            thresholds=raw.get("thresholds") or {},
            metadata=raw.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "conversation": self.conversation,
            "expected": self.expected,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
        }


def load_cases(path: str | Path = DEFAULT_DATASET_PATH) -> list[EvalCase]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases: list[EvalCase] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
            cases.append(EvalCase.from_dict(raw, line_no=line_no))

    ids = [case.id for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate case ids: {', '.join(duplicate_ids)}")
    return cases


def filter_cases(
    cases: list[EvalCase],
    category: str | None = None,
    case_id: str | None = None,
) -> list[EvalCase]:
    selected = cases
    if category:
        selected = [case for case in selected if case.category == category]
    if case_id:
        selected = [case for case in selected if case.id == case_id]
    return selected


def validate_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    cases = load_cases(path)
    categories: dict[str, int] = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    return {
        "path": str(Path(path)),
        "total": len(cases),
        "categories": categories,
        "case_ids": [case.id for case in cases],
    }
