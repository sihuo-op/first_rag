"""
RAGGraph 内部步骤工具。
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import jieba
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as langchain_tool
from opentelemetry import trace
from pydantic import BaseModel, Field

from app.core.observability import get_tracer
from app.llm.providers import invoke_llm_threadsafe

tracer = get_tracer("rag.tools")

# ============ 工具基类 ============

class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool = Field(description="执行是否成功")
    data: Any = Field(default=None, description="返回数据")
    message: str = Field(default="", description="结果描述")
    debug_info: Dict = Field(default_factory=dict, description="调试信息")


class BaseTool(ABC):
    """工具基类"""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init__(self, llm=None):
        self.llm = llm

    def execute(self, state: Dict) -> ToolResult:
        """模板方法：自动包 span、记录 attribute 和 status。子类实现 _execute_impl。"""
        with tracer.start_as_current_span(f"rag.{self.name}") as span:
            span.set_attribute("tool.name", self.name)
            try:
                result = self._execute_impl(state)
                span.set_attribute("tool.success", result.success)
                if not result.success:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, result.message))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    @abstractmethod
    def _execute_impl(self, state: Dict) -> ToolResult:
        """子类实现具体逻辑。异常可自行 try/except 返回 ToolResult，或让基类捕获记录。"""
        pass

    def to_langchain_tool(self):
        """转换为 LangChain Tool"""
        @langchain_tool(name=self.name)
        def _tool_func(state: str) -> str:
            state_dict = json.loads(state) if isinstance(state, str) else state
            result = self.execute(state_dict)
            return json.dumps({"success": result.success, "data": result.data, "message": result.message}, ensure_ascii=False)
        _tool_func.description = self.description
        return _tool_func


# ============ 检索工具 ============

class RetrieveTool(BaseTool):
    """检索工具 - 不需要 LLM"""
    name: ClassVar[str] = "retrieve"
    description: ClassVar[str] = "从文档库中检索与查询相关的内容片段"

    def __init__(self, retriever, top_k: int = 10):
        super().__init__(llm=None)
        self.retriever = retriever
        self.default_top_k = top_k

    def _execute_impl(self, state: Dict) -> ToolResult:
        query = state.get("current_query", state.get("original_question", ""))
        top_k = state.get("top_k", self.default_top_k)

        try:
            chunks, debug_info = self.retriever.retrieve(query=query, top_k=top_k)
            documents = [{
                "id": c.get("id"),
                "content": c.get("content", ""),
                "score": c.get("rrf_score", 0) or c.get("rerank_score", 0),
                "chunk_type": c.get("chunk_type", "unknown"),
                "document_id": c.get("document_id"),
                "dense_score": c.get("dense_score", 0),
                "sparse_score": c.get("sparse_score", 0),
                "rerank_score": c.get("rerank_score", 0),
                "rrf_score": c.get("rrf_score", 0),
            } for c in chunks]

            trace.get_current_span().set_attribute("tool.docs_found", len(documents))
            return ToolResult(
                success=True,
                data={"documents": documents, "count": len(documents), "query": query},
                message=f"检索到 {len(documents)} 个相关文档片段",
                debug_info=debug_info  # 保存详细调试信息
            )
        except Exception as e:
            return ToolResult(success=False, data=None, message=f"检索失败: {str(e)}")


# ============ 改写工具 ============

class RewriteTool(BaseTool):
    """查询改写工具"""
    name: ClassVar[str] = "rewrite"
    description: ClassVar[str] = "改写查询词使其更适合文档检索"

    REWRITE_TYPES = ["improve", "expand", "decompose"]

    def __init__(self, llm=None, rewrite_type: str = "improve"):
        super().__init__(llm=llm)
        self.default_rewrite_type = rewrite_type

    def _execute_impl(self, state: Dict) -> ToolResult:
        question = state.get("original_question", "")
        current_query = state.get("current_query", question)
        rewrite_type = state.get("rewrite_type", self.default_rewrite_type)
        documents = state.get("documents", [])
        eval_grade = state.get("eval_grade", "")  # CRAG 评估等级
        eval_reason = state.get("eval_reason", "")  # CRAG 评估理由

        if self.llm is None:
            return ToolResult(success=False, data=None, message="查询改写需要 LLM")

        # 根据 CRAG 评估等级自动选择改写策略
        if eval_grade and rewrite_type == self.default_rewrite_type:
            rewrite_type = self._select_rewrite_by_grade(eval_grade, current_query)

        if rewrite_type not in self.REWRITE_TYPES:
            rewrite_type = self._auto_select_rewrite_type(state)

        span = trace.get_current_span()
        span.set_attribute("tool.rewrite_type", rewrite_type)
        try:
            prompt = self._build_prompt(current_query, rewrite_type, documents, eval_grade, eval_reason)
            response = invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])
            queries = [q.strip() for q in self._parse_result(response.content.strip(), rewrite_type) if q.strip()]
            if not queries:
                return ToolResult(success=False, data=None, message="改写结果为空")

            span.set_attribute("tool.queries_count", len(queries))
            return ToolResult(
                success=True,
                data={"queries": queries, "rewrite_type": rewrite_type, "original_query": current_query},
                message=f"改写类型: {rewrite_type}, 生成 {len(queries)} 个查询",
            )
        except Exception as e:
            return ToolResult(success=False, data=None, message=f"改写失败: {str(e)}")

    def _select_rewrite_by_grade(self, grade: str, query: str) -> str:
        """根据 CRAG 评估等级选择改写策略

        - incorrect: 检索结果完全无关，需要 expand（扩展同义词/近义词）
        - ambiguous: 检索结果部分相关，需要 improve（优化查询词）
        """
        if grade == "incorrect":
            return "expand"  # 扩展同义词，如"终止"→"终止 解除 合同到期"
        elif grade == "ambiguous":
            return "improve"  # 优化查询词，使其更精准
        return self.default_rewrite_type

    def _auto_select_rewrite_type(self, state: Dict) -> str:
        question = state.get("original_question", "")
        attempt_count = state.get("attempt_count", 0)
        if len(question) > 100 or "和" in question or "以及" in question:
            return "decompose"
        if attempt_count >= 2:
            return "expand"
        return "improve"

    def _build_prompt(self, query: str, rewrite_type: str, documents: List, eval_grade: str = "", eval_reason: str = "") -> str:
        # 根据评估等级添加上下文提示
        eval_context = ""
        if eval_grade == "incorrect":
            eval_context = f"\n注意：之前的检索结果与问题完全无关（原因：{eval_reason}），请尝试使用同义词、近义词或相关法律术语来扩展查询。"
        elif eval_grade == "ambiguous":
            eval_context = f"\n注意：之前的检索结果与问题部分相关但不完整（原因：{eval_reason}），请优化查询词使其更精准。"

        prompts = {
            "improve": f"""请将以下问题改写为更适合文档检索的查询语句。
要求：保留核心意图、使用准确的专业关键词、去除口语化表达、简洁明了。
原问题：{query}{eval_context}
改写后的查询（只返回一行）：""",
            "decompose": f"""请将以下复杂问题分解为多个独立的子问题，每个子问题可以独立检索。
要求：每个子问题解决原问题的一个方面、子问题应该具体可检索。
原问题：{query}{eval_context}
分解后的子问题（每行一个）：""",
            "expand": f"""请为以下查询生成5个同义词替换版本，用于扩大检索范围。

规则：将原查询中的核心关键词替换为其同义词或近义词，保持查询意图不变。
每个版本替换不同的关键词，生成5行不同的查询。

示例：
原查询：劳动合同终止的情形
输出：
劳动合同解除的情形
劳动合同到期 期满的情形
劳动合同结束 合同失效的情形
劳动关系终止的情形
劳动合同不再继续的情形

原查询：{query}{eval_context}
输出："""
        }
        base_prompt = prompts.get(rewrite_type, prompts["improve"])
        if documents:
            doc_context = "\n".join([f"- {d.get('content', '')[:100]}..." for d in documents[:3]])
            base_prompt += f"\n\n已检索到的相关文档片段（供参考，避免重复）:\n{doc_context}"
        return base_prompt

    def _parse_result(self, text: str, rewrite_type: str) -> List[str]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = []
        for line in lines:
            line = re.sub(r'^[\d]+[.、]\s*', '', line)
            if line:
                cleaned.append(line)
        fallback = text.strip()
        return [fallback] if fallback else []


# ============ 生成工具 ============

class GenerateTool(BaseTool):
    """生成答案工具"""
    name: ClassVar[str] = "generate"
    description: ClassVar[str] = "根据检索到的文档内容生成答案"

    def __init__(self, llm=None, max_context_docs: int = 8):
        super().__init__(llm=llm)
        self.max_context_docs = max_context_docs

    def _execute_impl(self, state: Dict) -> ToolResult:
        question = state.get("original_question", "")
        documents = state.get("documents", [])

        if self.llm is None:
            return ToolResult(success=False, data=None, message="生成答案需要 LLM")
        if not documents:
            return ToolResult(success=False, data=None, message="无文档可用于生成答案")

        try:
            answer, prompt = self._generate_answer(question, documents)
            trace.get_current_span().set_attribute("tool.docs_used", min(len(documents), self.max_context_docs))
            return ToolResult(
                success=True,
                data={"answer": answer, "used_documents": min(len(documents), self.max_context_docs), "prompt": prompt},
                message="生成答案完成",
                debug_info={"prompt": prompt, "answer": answer}
            )
        except Exception as e:
            return ToolResult(success=False, data=None, message=f"生成失败: {str(e)}")

    def _generate_answer(self, question: str, documents: List) -> str:
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        context_docs = sorted_docs[:self.max_context_docs]
        context = "\n\n---\n\n".join([
            f"【文档片段 {i+1}】(相关度: {d.get('score', 0):.2f})\n{d.get('content', '')}"
            for i, d in enumerate(context_docs)
        ])
        prompt = f"""你是一个专业的劳动法知识助手。请严格根据以下提供的参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

回答要求：
1. 严格基于参考文档内容回答，不得添加文档中未提及的任何信息
2. 直接引用文档中的原文条款，引用时标注来源（如"第X条"或"文档片段N"）
3. 如果文档内容足以回答问题，简洁准确地给出答案，不需要额外解读或分析
4. 如果文档内容仅部分相关，只回答文档能支撑的部分，对文档未覆盖的内容明确说明"参考文档未涉及"
5. 只有当所有文档内容都与问题完全无关时，才说明"未找到相关信息"
6. 禁止对法律条款进行延伸解读、推理或补充说明，只陈述文档中已有的内容

请给出回答："""
        response = invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])
        return response.content.strip(), prompt


# ============ 评估工具 ============

class EvaluateTool(BaseTool):
    """检索结果评估工具 - CRAG 风格

    评估检索到的文档是否能真正回答用户问题，分为三个等级：
    - Correct：文档与问题高度相关，可直接生成答案
    - Incorrect：文档与问题无关，需要改写查询重新检索
    - Ambiguous：文档部分相关，建议改写查询补充检索

    评估方式：优先使用 LLM 语义评估（判断内容是否匹配），LLM 不可用时回退到规则评估
    """

    name: ClassVar[str] = "evaluate"
    description: ClassVar[str] = "评估检索到的文档片段是否能回答用户问题"

    # CRAG 三级评估结果
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"

    # Score-based fast-path 阈值（基于 bge-reranker-base 在 23 查询样本上的分布）
    # top1 rerank_score >= HIGH -> 直接判 correct，跳过 LLM
    # top1 rerank_score <= LOW  -> 直接判 incorrect，跳过 LLM
    # 中间区间走 LLM 语义评估
    HIGH_SCORE_THRESHOLD = 0.9
    LOW_SCORE_THRESHOLD = 0.2

    def __init__(self, llm=None, confidence_threshold: float = 0.5):
        super().__init__(llm=llm)
        self.confidence_threshold = confidence_threshold

    def _execute_impl(self, state: Dict) -> ToolResult:
        question = state.get("original_question", "")
        documents = state.get("documents", [])

        if not documents:
            return ToolResult(
                success=True,
                data={
                    "confidence": 0.0,
                    "is_sufficient": False,
                    "grade": self.INCORRECT,
                    "reason": "没有检索到任何文档",
                    "suggestion": "尝试改写查询",
                },
                message="评估完成：无文档",
                debug_info={"evaluation": {"confidence": 0.0, "grade": self.INCORRECT, "reason": "没有检索到任何文档", "method": "rule"}}
            )

        # Score-based fast-path：明确高/低分直接判定，跳过 LLM
        fast_path = self._score_fast_path_evaluate(documents)
        if fast_path is not None:
            evaluation = fast_path
        elif self.llm:
            evaluation = self._llm_evaluate_chunks(question, documents)
        else:
            evaluation = self._rule_evaluate_chunks(question, documents)

        is_sufficient = evaluation["grade"] == self.CORRECT

        span = trace.get_current_span()
        span.set_attribute("tool.grade", evaluation["grade"])
        span.set_attribute("tool.confidence", evaluation["confidence"])
        span.set_attribute("tool.evaluation_method", evaluation.get("method", "unknown"))

        return ToolResult(
            success=True,
            data={
                "confidence": evaluation["confidence"],
                "is_sufficient": is_sufficient,
                "grade": evaluation["grade"],
                "reason": evaluation.get("reason", ""),
                "suggestion": evaluation.get("suggestion", ""),
            },
            message=f"评估完成：{evaluation['grade']}，置信度 {evaluation['confidence']:.2f}",
            debug_info={"evaluation": evaluation}
        )

    def _score_fast_path_evaluate(self, documents: List[Dict]) -> Dict:
        """基于 rerank_score 的快速判定：明确高/低分跳过 LLM。

        bge-reranker-base 输出已 sigmoid 归一化到 [0, 1]：
        - top1 >= HIGH_SCORE_THRESHOLD：文档强相关，直接判 correct
        - top1 <= LOW_SCORE_THRESHOLD：文档弱相关或无命中，直接判 incorrect
        - 中间区间返回 None，由调用方走 LLM 评估
        """
        top1_score = 0.0
        for doc in documents:
            s = doc.get("rerank_score")
            if s is None or s == 0:
                s = doc.get("rrf_score", 0)
            if s and s > top1_score:
                top1_score = s

        if top1_score >= self.HIGH_SCORE_THRESHOLD:
            return {
                "confidence": round(min(1.0, top1_score), 2),
                "grade": self.CORRECT,
                "reason": f"top1 rerank_score={top1_score:.3f} >= {self.HIGH_SCORE_THRESHOLD}，强相关",
                "suggestion": "可直接生成答案",
                "method": "score_fast_path_high",
                "top1_score": round(top1_score, 4),
            }
        if top1_score <= self.LOW_SCORE_THRESHOLD:
            return {
                "confidence": round(top1_score, 2),
                "grade": self.INCORRECT,
                "reason": f"top1 rerank_score={top1_score:.3f} <= {self.LOW_SCORE_THRESHOLD}，弱相关或无命中",
                "suggestion": "建议改写查询重新检索",
                "method": "score_fast_path_low",
                "top1_score": round(top1_score, 4),
            }
        return None

    def _llm_evaluate_chunks(self, question: str, documents: List[Dict]) -> Dict:
        """CRAG 风格的 LLM 语义评估

        核心思路：让 LLM 判断检索到的文档内容是否真正能回答用户问题，
        而不是只看分数信号。这是发现"解除≠终止"等语义不匹配的关键。
        """
        doc_summary = "\n".join([
            f"[文档{i+1}] {d.get('content', '')[:300]}"
            for i, d in enumerate(documents[:5])
        ])

        prompt = f"""你是一个检索质量评估专家。请判断以下检索到的文档是否能回答用户的问题。

用户问题: {question}

检索到的文档:
{doc_summary}

请评估这些文档与用户问题的关系，返回 JSON 格式:
{{
  "grade": "correct/incorrect/ambiguous",
  "confidence": 0.0-1.0,
  "reason": "评估理由",
  "suggestion": "改进建议"
}}

评估标准：
- correct: 文档中包含能直接回答用户问题的内容，信息充分
- incorrect: 文档内容与用户问题完全无关，无法从中找到答案
- ambiguous: 文档内容与问题部分相关，但信息不够完整或不够直接

注意：
1. 要判断文档的实质内容是否匹配问题，而不是看关键词是否重叠
2. 例如问题问"劳动合同终止"，但文档只讲"劳动合同解除"，这属于 incorrect（法律上"解除"和"终止"是不同概念）
3. 例如问题问"加班工资计算"，文档提到了加班但没提计算方法，这属于 ambiguous

只返回 JSON:"""

        try:
            response = invoke_llm_threadsafe(self.llm, [HumanMessage(content=prompt)])
            content = response.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                grade = result.get("grade", self.AMBIGUOUS).lower()
                if grade not in (self.CORRECT, self.INCORRECT, self.AMBIGUOUS):
                    grade = self.AMBIGUOUS
                confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))

                # 根据等级调整置信度，确保与等级一致
                if grade == self.CORRECT:
                    confidence = max(confidence, 0.7)
                elif grade == self.INCORRECT:
                    confidence = min(confidence, 0.3)
                else:  # ambiguous
                    confidence = max(0.3, min(0.7, confidence))

                return {
                    "confidence": round(confidence, 2),
                    "grade": grade,
                    "reason": result.get("reason", ""),
                    "suggestion": result.get("suggestion", ""),
                    "method": "llm",
                }
        except Exception as e:
            print(f"[EvaluateTool] LLM 评估失败，回退到规则评估: {e}")

        return self._rule_evaluate_chunks(question, documents)

    def _rule_evaluate_chunks(self, question: str, documents: List[Dict]) -> Dict:
        """基于规则的评估逻辑（LLM 不可用时的兜底方案）

        综合多个信号评估检索质量：
        1. 文档数量（检索到的相关文档越多越好）
        2. Top1 rerank 分数（最高相关性）
        3. 平均 rerank 分数（整体相关性）
        4. RRF 双路命中比例（两路检索都命中的文档更可靠）
        5. 关键词覆盖率（问题关键词在文档中出现的比例）
        """
        if not documents:
            return {"confidence": 0.0, "grade": self.INCORRECT, "reason": "未检索到相关文档", "suggestion": "尝试改写查询词", "method": "rule"}

        # --- 信号 1：文档数量（0~0.15 分）---
        doc_count = len(documents)
        count_score = min(0.15, doc_count * 0.03)

        # --- 信号 2：Top1 rerank 分数（0~0.20 分）---
        top1_score = 0.0
        for doc in documents:
            s = doc.get("rerank_score", 0)
            if s is None or s == 0:
                s = doc.get("rrf_score", 0)
            top1_score = max(top1_score, s)

        has_rerank = any(d.get("rerank_score") is not None and d.get("rerank_score", 0) != 0 for d in documents)
        if has_rerank:
            top1_signal = min(0.20, max(0.0, top1_score))
        else:
            top1_signal = min(0.20, top1_score * 10)

        # --- 信号 3：平均相关性（0~0.15 分）---
        all_scores = []
        for doc in documents:
            s = doc.get("rerank_score", 0)
            if s is None or s == 0:
                s = doc.get("rrf_score", 0)
            all_scores.append(s)

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        if has_rerank:
            avg_signal = min(0.15, max(0.0, avg_score))
        else:
            avg_signal = min(0.15, avg_score * 10)

        # --- 信号 4：RRF 双路命中比例（0~0.10 分）---
        both_hit_count = sum(
            1 for d in documents
            if d.get("dense_score", 0) > 0 and d.get("sparse_score", 0) > 0
        )
        both_hit_ratio = both_hit_count / doc_count if doc_count > 0 else 0
        overlap_signal = both_hit_ratio * 0.10

        # --- 信号 5：关键词覆盖率（0~0.40 分）--- 新增！
        # 提取问题中的关键词，检查在文档中出现的比例
        question_keywords = [w for w in jieba.cut(question) if len(w) > 1]
        if question_keywords:
            keyword_hit_count = 0
            for kw in question_keywords:
                for doc in documents:
                    if kw in doc.get("content", ""):
                        keyword_hit_count += 1
                        break
            keyword_coverage = keyword_hit_count / len(question_keywords)
            keyword_signal = keyword_coverage * 0.40
        else:
            keyword_signal = 0.20  # 无法提取关键词时给中间分

        # --- 综合置信度 ---
        confidence = round(min(1.0, count_score + top1_signal + avg_signal + overlap_signal + keyword_signal), 2)

        # --- 根据 confidence 确定 CRAG 等级 ---
        if confidence >= 0.6:
            grade = self.CORRECT
            reason = f"检索结果与问题相关（文档数:{doc_count}, top1:{top1_score:.3f}, 关键词覆盖率:{keyword_coverage:.0%}）"
            suggestion = "可直接生成答案"
        elif confidence >= 0.3:
            grade = self.AMBIGUOUS
            reason = f"检索结果部分相关（文档数:{doc_count}, top1:{top1_score:.3f}, 关键词覆盖率:{keyword_coverage:.0%}）"
            suggestion = "建议改写查询补充检索"
        else:
            grade = self.INCORRECT
            reason = f"检索结果与问题不相关（文档数:{doc_count}, top1:{top1_score:.3f}, 关键词覆盖率:{keyword_coverage:.0%}）"
            suggestion = "建议改写查询重新检索"

        return {"confidence": confidence, "grade": grade, "reason": reason, "suggestion": suggestion, "method": "rule"}
