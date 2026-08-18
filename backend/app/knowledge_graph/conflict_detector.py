"""冲突检测：新 Article EXPLAINS Concept 时，与已有 Articles 矛盾判定。

检测与 supersede 解耦：这里只标记 ``CONFLICTS_WITH`` 边（status=pending_review），
由人工（Task 11 kg_admin）决定哪条胜出，本模块不执行任何 supersede 动作。
"""
import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage

from app.core.observability import get_tracer
from app.knowledge_graph.schema import ConflictStatus, EdgeType
from app.llm.providers import invoke_llm_threadsafe

logger = logging.getLogger(__name__)

tracer = get_tracer("kg.extract")

# LLM 调用失败时的重试次数（总尝试次数 = 1 + MAX_LLM_RETRIES），与 llm_extractor 一致
MAX_LLM_RETRIES = 2

CONFLICT_PROMPT_TEMPLATE = """你是劳动法冲突检测专家。判断两条法条对同一概念的规定是否矛盾。

概念：{concept_name}
法条A（新入库）：{new_content}
法条B（已存在）：{existing_content}

判断：
- 矛盾：两条对同一概念给出互斥规定（如"最长6个月" vs "最长3个月"）
- 互补：两条从不同角度规定，不互斥
- 不相关：虽然都讲同一概念但内容无交集

输出 JSON: {{"is_conflict": bool, "reason": "...", "confidence": 0.0-1.0}}
"""


def _safe_confidence(raw) -> float:
    """confidence 转 float，非数值时回退 0.0（低置信）。"""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _safe_bool(raw) -> bool:
    """is_conflict 宽松解析：LLM 偶尔回字符串 ``"false"``，朴素 bool() 会误判为 True。"""
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1")
    return bool(raw)


class ConflictDetector:
    def __init__(self, store, llm):
        self.store = store
        self.llm = llm

    def detect_for_article(self, article_id: str) -> int:
        """检测新 Article 与已有 Articles 解释同 Concept 时的矛盾。

        返回本次新检测出的冲突数（int，绝不为 None；Task 9 extractor 用 += 累加）。
        """
        with tracer.start_as_current_span("kg.extract.conflict_detect") as span:
            span.set_attribute("article.id", article_id)
            existing_articles = self._find_related_articles(article_id)
            span.set_attribute("existing.count", len(existing_articles))
            if not existing_articles:
                return 0

            conflict_count = 0
            for existing in existing_articles:
                is_conflict, reason, confidence = self._llm_judge(
                    existing["concept_name"],
                    existing["new_content"],
                    existing["existing_content"],
                )
                if is_conflict:
                    self._write_conflict_edge(
                        from_id=article_id,
                        to_id=existing["existing_id"],
                        reason=reason,
                        confidence=confidence,
                    )
                    conflict_count += 1

            span.set_attribute("conflict.count", conflict_count)
            return conflict_count

    def _find_related_articles(self, article_id: str) -> list[dict]:
        """找与新 Article 共享 Concept 的已有 Articles。

        - 只经 ``(:Article)-[:EXPLAINS]->(:Concept)<-[:EXPLAINS]-(:Article)`` 找候选；
        - 排除已存在 CONFLICTS_WITH 边（无方向）的文章对：避免重复 LLM 判定，
          也避免 merge SET 把人工已审核（confirmed/dismissed）的状态覆盖回 pending_review。
        """
        with self.store.session() as s:
            result = s.run(
                """
                MATCH (new:Article {id: $article_id})-[:EXPLAINS]->(c:Concept)<-[:EXPLAINS]-(existing:Article)
                WHERE existing.id <> $article_id
                  AND NOT (new)-[:CONFLICTS_WITH]-(existing)
                RETURN existing.id AS existing_id, c.name AS concept_name, c.id AS concept_id,
                       new.content AS new_content, existing.content AS existing_content
                """,
                article_id=article_id,
            )
            return [dict(r) for r in result]

    def _llm_judge(self, concept: str, new_content: str, existing_content: str) -> tuple[bool, str, float]:
        """LLM 判定两条内容是否矛盾。返回 (is_conflict, reason, confidence)。

        LLM 调用失败重试 MAX_LLM_RETRIES 次，全部失败按"无冲突"处理（宁缺勿滥）；
        输出非 JSON / confidence 非数值同样按解析失败容错。
        """
        # or "" 兜底：旧数据的 Article 节点可能没有 content 属性（Cypher 返回 null），
        # None[:500] 会 TypeError 并炸掉整条 extractor 管道。
        prompt = CONFLICT_PROMPT_TEMPLATE.format(
            concept_name=concept,
            new_content=(new_content or "")[:500],
            existing_content=(existing_content or "")[:500],
        )
        with tracer.start_as_current_span("kg.extract.conflict_judge") as jspan:
            response = self._invoke_with_retry(prompt)
            if response is None:
                jspan.set_attribute("result.status", "llm_failed")
                return False, "LLM 调用失败", 0.0

            content = (getattr(response, "content", "") or "").strip()
            try:
                json_match = re.search(r"\{[\s\S]*\}", content)
                if not json_match:
                    jspan.set_attribute("result.status", "no_json_found")
                    return False, "JSON 解析失败", 0.0
                data = json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError, ValueError) as exc:
                jspan.set_attribute("result.status", "parse_failed")
                jspan.record_exception(exc)
                return False, str(exc), 0.0

            return (
                _safe_bool(data.get("is_conflict")),
                str(data.get("reason", "")),
                _safe_confidence(data.get("confidence", 0.0)),
            )

    def _invoke_with_retry(self, prompt: str):
        """调用 LLM，失败时重试 MAX_LLM_RETRIES 次。返回 response 或 None。"""
        messages = [HumanMessage(content=prompt)]
        last_exc: Exception | None = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return invoke_llm_threadsafe(self.llm, messages)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "kg.extract.conflict llm call failed attempt=%d/%d: %s",
                    attempt + 1, MAX_LLM_RETRIES + 1, exc,
                )
                if attempt < MAX_LLM_RETRIES:
                    continue
        # 所有重试都失败：记录到 OTel 并返回 None
        with tracer.start_as_current_span("kg.extract.conflict_llm.error") as span:
            if last_exc is not None:
                span.record_exception(last_exc)
            span.set_attribute("error.kind", "llm_call_failed")
        return None

    def _write_conflict_edge(self, from_id: str, to_id: str, reason: str, confidence: float) -> None:
        props = {
            "status": ConflictStatus.PENDING_REVIEW.value,
            "reason": reason,
            "confidence": confidence,
            "detected_at": datetime.now().isoformat(),
        }
        self.store.merge_relation(from_id, to_id, EdgeType.CONFLICTS_WITH.value, props)
