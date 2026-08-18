"""KG 后台 API：审核队列、冲突确认、Article supersede、图查询、统计。

语义要点（spec Section 6）：
- 冲突检测与 supersede 解耦：confirm/dismiss 只改 CONFLICTS_WITH 边状态，
  绝不自动 supersede 任何 Article；supersede 是独立的人工决策。
- 冲突列表返回两条 Article 全文（ArticleNode.content，Task 10 起落库），
  供审核人对照。
- 全部端点要求 admin 身份（与 app/api/admin.py 同一依赖
  get_current_admin_user）；confirm/dismiss 将审核人 users.id 写入
  边的 reviewed_by 属性（对齐 chunk 审核的 reviewed_by 约定）。
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import get_current_admin_user
from app.entities.database import User
from app.knowledge_graph.exceptions import KGError
from app.knowledge_graph.graph_store import get_graph_store
from app.knowledge_graph.schema import ArticleStatus, ConflictStatus

router = APIRouter(prefix="/api/v1/admin/kg", tags=["kg-admin"])


def _get_store():
    try:
        return get_graph_store()
    except KGError as e:
        raise HTTPException(status_code=503, detail=f"KG unavailable: {e}") from e


class ReviewRequest(BaseModel):
    review_note: str = ""


class SupersedeRequest(BaseModel):
    reason: str
    conflict_edge_id: str | None = None  # 可选：记录 supersede 溯源，不影响冲突边


@router.get("/conflicts")
def list_conflicts(
    status: str = ConflictStatus.PENDING_REVIEW.value,
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    valid_statuses = {s.value for s in ConflictStatus}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"invalid status {status!r}, must be one of {sorted(valid_statuses)}",
        )
    with store.session() as s:
        result = s.run(
            """
            MATCH (a:Article)-[r:CONFLICTS_WITH {status: $status}]->(b:Article)
            RETURN id(r) AS edge_id, a.id AS a_id, a.law_id AS a_law_id,
                   a.article_no AS a_no, a.content AS a_content,
                   b.id AS b_id, b.law_id AS b_law_id,
                   b.article_no AS b_no, b.content AS b_content,
                   r.reason AS reason, r.confidence AS confidence,
                   r.detected_at AS detected_at
            """,
            status=status,
        )
        conflicts = [dict(r) for r in result]
    return {"conflicts": conflicts}


@router.post("/conflicts/{edge_id}/confirm")
def confirm_conflict(
    edge_id: int,
    req: ReviewRequest,
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    with store.session() as s:
        summary = s.run(
            """
            MATCH ()-[r:CONFLICTS_WITH]->()
            WHERE id(r) = $edge_id
            SET r.status = $status,
                r.reviewed_by = $reviewed_by,
                r.reviewed_at = $now,
                r.review_note = $note
            """,
            edge_id=edge_id,
            status=ConflictStatus.CONFIRMED.value,
            reviewed_by=current_user.id,
            now=datetime.now().isoformat(),
            note=req.review_note,
        ).consume()
    if summary.counters.properties_set == 0:
        raise HTTPException(status_code=404, detail=f"conflict edge {edge_id} not found")
    return {"status": "confirmed"}


@router.post("/conflicts/{edge_id}/dismiss")
def dismiss_conflict(
    edge_id: int,
    req: ReviewRequest,
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    with store.session() as s:
        summary = s.run(
            """
            MATCH ()-[r:CONFLICTS_WITH]->()
            WHERE id(r) = $edge_id
            SET r.status = $status,
                r.reviewed_by = $reviewed_by,
                r.reviewed_at = $now,
                r.review_note = $note
            """,
            edge_id=edge_id,
            status=ConflictStatus.DISMISSED.value,
            reviewed_by=current_user.id,
            now=datetime.now().isoformat(),
            note=req.review_note,
        ).consume()
    if summary.counters.properties_set == 0:
        raise HTTPException(status_code=404, detail=f"conflict edge {edge_id} not found")
    return {"status": "dismissed"}


@router.post("/articles/{article_id}/supersede")
def supersede_article(
    article_id: str,
    req: SupersedeRequest,
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    with store.session() as s:
        summary = s.run(
            """
            MATCH (a:Article {id: $article_id})
            SET a.status = $status,
                a.supersede_reason = $reason,
                a.supersede_via_conflict_edge = $conflict_edge_id,
                a.superseded_at = $now
            """,
            article_id=article_id,
            status=ArticleStatus.SUPERSEDED.value,
            reason=req.reason,
            conflict_edge_id=req.conflict_edge_id,
            now=datetime.now().isoformat(),
        ).consume()
    if summary.counters.properties_set == 0:
        raise HTTPException(status_code=404, detail=f"article {article_id} not found")
    return {"status": "superseded", "article_id": article_id}


@router.get("/graph")
def get_subgraph(
    concept_id: str,
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    """Concept 周围 1..2 跳子图（可视化/调试用）。"""
    with store.session() as s:
        result = s.run(
            """
            MATCH (c:Concept {id: $concept_id})-[r*1..2]-(n)
            RETURN n, r
            LIMIT 50
            """,
            concept_id=concept_id,
        )
        nodes = []
        edges = []
        seen_nodes = set()
        for record in result:
            node = record["n"]
            props = dict(node)
            node_key = props.get("id") or getattr(node, "element_id", None) or str(len(nodes))
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                nodes.append({"id": node_key, "labels": list(node.labels), "properties": props})
            for rel in record["r"]:
                edges.append({
                    "type": rel.type,
                    "from": dict(rel.start_node).get("id") or rel.start_node.element_id,
                    "to": dict(rel.end_node).get("id") or rel.end_node.element_id,
                })
    return {"nodes": nodes, "edges": edges}


@router.get("/stats")
def get_stats(
    store=Depends(_get_store),
    current_user: User = Depends(get_current_admin_user),
):
    with store.session() as s:
        rec = s.run(
            """
            CALL { MATCH (n) RETURN count(n) AS node_count }
            CALL { MATCH ()-[r]->() RETURN count(r) AS edge_count }
            CALL {
                MATCH ()-[r:CONFLICTS_WITH {status: $pending}]->()
                RETURN count(r) AS pending_count
            }
            RETURN node_count, edge_count, pending_count
            """,
            pending=ConflictStatus.PENDING_REVIEW.value,
        ).single()
    return {
        "node_count": rec["node_count"],
        "edge_count": rec["edge_count"],
        "pending_count": rec["pending_count"],
    }
