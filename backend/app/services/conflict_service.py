"""
冲突检测服务

新文档入库后，对每个新 chunk 检索相似旧 chunk，用 LLM 判定是否语义冲突。
高置信度自动作废，低置信度转人工审核。
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.observability import get_tracer
from app.entities.database import DocumentChunk
from app.llm.providers import get_rewrite_llm, invoke_llm_threadsafe
from app.rag.vector_store import MilvusStore

tracer = get_tracer("conflict")


class ConflictService:
    def __init__(self, db: Session, vector_store: MilvusStore):
        self.db = db
        self.vector_store = vector_store
        self.settings = get_settings()
        self.llm = get_rewrite_llm()

    def judge_conflicts(self, new_content: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        用 LLM 判断 new_content 与每个 candidate 是否冲突。

        Args:
            new_content: 新 chunk 的文本内容
            candidates: 候选旧 chunk 列表，每项含 id 和 content

        Returns:
            判定结果列表，每项含 old_id / conflict / confidence / reason
        """
        if not candidates:
            return []

        prompt = self._build_prompt(new_content, candidates)
        try:
            response = self._invoke_llm(prompt)
            return self._parse_response(response, candidates)
        except Exception as e:
            print(f"[ConflictService] LLM 判冲突失败: {e}")
            return []

    def _invoke_llm(self, prompt: str):
        return invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])

    def _build_prompt(self, new_content: str, candidates: List[Dict]) -> str:
        cand_text = "\n".join(
            f"[{i}] id={c['id']}\n内容: {c['content']}"
            for i, c in enumerate(candidates)
        )
        return f"""判断新内容与下列每条旧内容是否语义冲突（讲同一件事但结论不同/已过期）。

新内容：{new_content}

旧内容列表：
{cand_text}

要求：
1. 对每条旧内容判断是否与新内容冲突
2. conflict=true 表示讲同一事但结论矛盾，新内容应取代旧内容
3. conflict=false 表示讲不同事、互补关系、或无关
4. confidence 范围 0-1

只输出 JSON 数组，不要其他文字：
[{{"old_id": "<id>", "conflict": true/false, "confidence": 0.0, "reason": "<简短原因>"}}]"""

    def _parse_response(self, response, candidates: List[Dict]) -> List[Dict]:
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON 数组（容错：LLM 可能加额外文字）
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group(0))
            # 校验结构
            valid_ids = {c["id"] for c in candidates}
            return [
                {
                    "old_id": item.get("old_id"),
                    "conflict": bool(item.get("conflict", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "reason": item.get("reason", "")
                }
                for item in result
                if item.get("old_id") in valid_ids
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[ConflictService] 解析 LLM 响应失败: {e}")
            return []
