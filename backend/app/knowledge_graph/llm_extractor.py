"""LLM 抽 Concept/Party/关系，per chunk 调用。

调用通过 ``invoke_llm_threadsafe`` 走线程安全包装；LLM 异常时重试 2 次。
解析失败的输出（非 JSON / 缺字段）静默返回空 ``ExtractionResult``，
未知节点/关系类型按白名单过滤。
"""
import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from app.core.observability import get_tracer
from app.llm.providers import invoke_llm_threadsafe

logger = logging.getLogger(__name__)

tracer = get_tracer("kg.extract")

# 与 schema.NodeType / EdgeType 对齐的白名单
VALID_ENTITY_TYPES = {"Concept", "Party"}
VALID_RELATION_TYPES = {
    "CITES",
    "IS_A",
    "CONFLICTS_WITH",
    "APPLIES_TO",
    "CONTAINS",
    "EXPLAINS",
}

# LLM 调用失败时的重试次数（总尝试次数 = 1 + MAX_LLM_RETRIES）
MAX_LLM_RETRIES = 2


@dataclass
class ExtractedEntity:
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class ExtractedRelation:
    type: str
    from_ref: str
    to_ref: str
    confidence: float = 0.5


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


EXTRACTION_PROMPT_TEMPLATE = """你是劳动法知识图谱抽取助手。从下面的法律文本片段抽取实体和关系。

节点类型：
- Concept: 法律概念（如"试用期""经济补偿""加班费"）
- Party: 主体（如"用人单位""劳动者""工会"）

关系类型：
- CITES: 当前 Article 引用了另一条 Article（from="article:<法名>:<条号>", to="article:<法名>:<条号>"）
- IS_A: 概念上下位（from="concept:<name>", to="concept:<name>"）
- EXPLAINS: Article 解释 Concept（from="article:<法名>:<条号>", to="concept:<name>"）
- APPLIES_TO: Article 适用 Party/Region（from="article:<法名>:<条号>", to="party:<name>" 或 "region:<name>"）
- CONTAINS: 父节点包含子节点（如 Law CONTAINS Article）
- CONFLICTS_WITH: 节点之间存在冲突

文本片段（Article {article_no}）：
{chunk_text}

输出严格 JSON（不要 markdown 代码块）：
{{
  "entities": [{{"type": "Concept|Party", "name": "...", "aliases": [...]}}],
  "relations": [{{"type": "...", "from": "...", "to": "...", "confidence": 0.0-1.0}}]
}}
"""


def _call_llm_with_retry(llm, prompt: str, chunk_id: str):
    """调用 LLM，失败时重试 MAX_LLM_RETRIES 次。返回 response 或 None。"""
    messages = [HumanMessage(content=prompt)]
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            return invoke_llm_threadsafe(llm, messages)
        except Exception as exc:  # noqa: BLE001 - LLM 提供方异常类型不固定
            last_exc = exc
            logger.warning(
                "kg.extract.llm call failed chunk_id=%s attempt=%d/%d: %s",
                chunk_id,
                attempt + 1,
                MAX_LLM_RETRIES + 1,
                exc,
            )
            if attempt < MAX_LLM_RETRIES:
                continue
    # 所有重试都失败：记录到 OTel 并返回 None
    with tracer.start_as_current_span("kg.extract.llm.error") as span:
        if last_exc is not None:
            span.record_exception(last_exc)
        span.set_attribute("chunk.id", chunk_id)
        span.set_attribute("error.kind", "llm_call_failed")
    return None


def extract_from_chunk(
    chunk_text: str,
    chunk_id: str,
    article_no: int | None,
    llm,
) -> ExtractionResult:
    """从 chunk 文本抽取实体/关系。

    - LLM 调用失败：重试 2 次，全部失败后返回空结果。
    - 输出非 JSON / 缺字段：返回空结果。
    - 未知 type / 缺 from/to：按白名单过滤。
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        article_no=article_no if article_no is not None else "?",
        chunk_text=chunk_text[:2000],  # 截断防止超长
    )

    with tracer.start_as_current_span("kg.extract.llm") as span:
        span.set_attribute("chunk.id", chunk_id)
        if article_no is not None:
            span.set_attribute("article.no", article_no)

        response = _call_llm_with_retry(llm, prompt, chunk_id)
        if response is None:
            span.set_attribute("result.status", "llm_failed")
            return ExtractionResult()

        content = getattr(response, "content", "") or ""
        content = content.strip()
        span.set_attribute("response.length", len(content))

        # 容错：尝试从可能的 markdown 代码块中提取 JSON
        try:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                span.set_attribute("result.status", "no_json_found")
                return ExtractionResult()
            data = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError, ValueError) as exc:
            span.set_attribute("result.status", "parse_failed")
            span.record_exception(exc)
            return ExtractionResult()

    def _safe_confidence(raw) -> float:
        """Convert confidence to float, falling back to 0.5 on non-numeric values."""
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.5

    entities = [
        ExtractedEntity(
            type=e["type"],
            name=e["name"],
            aliases=list(e.get("aliases") or []),
        )
        for e in data.get("entities", [])
        if isinstance(e, dict)
        and e.get("type") in VALID_ENTITY_TYPES
        and e.get("name")
    ]

    relations = [
        ExtractedRelation(
            type=r["type"],
            from_ref=r["from"],
            to_ref=r["to"],
            confidence=_safe_confidence(r.get("confidence", 0.5)),
        )
        for r in data.get("relations", [])
        if isinstance(r, dict)
        and r.get("type") in VALID_RELATION_TYPES
        and r.get("from")
        and r.get("to")
    ]

    return ExtractionResult(entities=entities, relations=relations)
