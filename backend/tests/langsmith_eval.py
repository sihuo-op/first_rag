"""
LangSmith RAG 评估脚本 - Agentic RAG 版本

适配当前的 Agentic RAG 系统（ReAct Commander 模式）

RAGAS 指标说明:
- faithfulness: 答案对上下文的忠诚度
- answer_relevancy: 答案与问题的相关性
- context_precision: 上下文精确度
- context_recall: 上下文召回率（需要 ground_truth）

使用方法:
1. 在项目根目录的 .env 文件中添加:
   LANGSMITH_API_KEY=your_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=rag-evaluation
2. 在 LangSmith 网页上创建 dataset
3. 修改下方的 DATASET_NAME 为你的数据集名称
4. 运行: python -m tests.langsmith_eval

数据集格式要求:
- inputs: {"question": "用户问题"}
- outputs: {"expected_answer": "参考答案", "reference": "参考上下文/ground_truth"}（可选，缺失时自动用 LLM 生成）
"""

import os
import sys
import asyncio
import math
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 确保工作目录正确
os.chdir(Path(__file__).parent.parent.parent)

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from langsmith.wrappers import wrap_openai
from openai import OpenAI

# RAGAS 相关导入（兼容 ragas 0.4.x）
# ragas 0.4.x 的 ragas.metrics.collections 中的指标类不继承 Metric，
# 导致 evaluate() 的 isinstance 检查失败，因此使用 ragas.metrics 中的旧版类
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

try:
    from ragas import evaluate as ragas_evaluate
    from datasets import Dataset
    try:
        # RAGAS >= 0.4 需要显式配置 LLM 和 embeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        RAGAS_REQUIRES_CONFIG = True
    except ImportError:
        # RAGAS < 0.4 不需要显式配置
        RAGAS_REQUIRES_CONFIG = False
    
    # 修复 RAGAS 0.4.x 的 parse_run_traces bug：当 traces 为空时 IndexError
    try:
        from ragas.callbacks import parse_run_traces as _orig_parse_run_traces
        import ragas.callbacks as _ragas_callbacks_module
        import ragas.dataset_schema as _ragas_ds_module
        def _safe_parse_run_traces(traces, parent_run_id=None):
            root_traces = [
                ct for ct in traces.values()
                if ct.parent_run_id == parent_run_id
            ]
            if not root_traces:
                return []  # 空 traces 时返回空列表而非抛出 IndexError
            return _orig_parse_run_traces(traces, parent_run_id)
        _ragas_callbacks_module.parse_run_traces = _safe_parse_run_traces
        _ragas_ds_module.parse_run_traces = _safe_parse_run_traces  # dataset_schema 也引用了此函数
    except Exception:
        pass  # 如果 patch 失败，不影响主流程
    
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    RAGAS_REQUIRES_CONFIG = False
    print("警告: RAGAS 未安装，跳过 RAGAS 评估")

# ============================================
# 在这里修改为你的 LangSmith 数据集名称
# ============================================
DATASET_NAME = "firstRagTest"  # 改成你在网页上创建的数据集名称
# ============================================


class RAGEvaluator:
    """RAG 评估器"""

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        # 从设置中获取 API 配置并创建 OpenAI 客户端
        self.llm_client = wrap_openai(OpenAI(
            api_key=settings.CHAT_API_KEY,
            base_url=settings.CHAT_API_BASE or None
        ))
        self.evaluator_model = settings.CHAT_MODEL  # 使用配置的 Chat 模型
        
        # RAGAS 配置（使用 langchain_openai 避免 InstructorLLM 的 json_object 问题）
        self.ragas_configured = False
        self.ragas_llm = None
        self.ragas_embeddings = None
        
        if RAGAS_AVAILABLE and RAGAS_REQUIRES_CONFIG:
            try:
                from langchain_openai import ChatOpenAI as LangchainChatOpenAI
                
                # 使用 langchain ChatOpenAI（不会发送 response_format: json_object）
                lc_llm = LangchainChatOpenAI(
                    model=settings.CHAT_MODEL,
                    api_key=settings.CHAT_API_KEY,
                    base_url=settings.CHAT_API_BASE,
                    temperature=0,
                )
                self.ragas_llm = LangchainLLMWrapper(lc_llm)
                
                # 使用 langchain_community HuggingFaceEmbeddings（本地 BAAI/bge-m3，避免豆包 API 兼容问题）
                from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
                lc_embeddings = LCHuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                )
                self.ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
                
                self.ragas_configured = True
                print("RAGAS 配置成功（langchain_openai ChatOpenAI + HuggingFaceEmbeddings bge-m3）")
            except ImportError:
                print("langchain_openai 未安装，尝试安装: pip install langchain-openai")
                self.ragas_configured = False
            except Exception as e:
                print(f"RAGAS 配置失败: {e}")
                import traceback
                traceback.print_exc()
                self.ragas_configured = False

    def _init_retriever(self):
        """懒初始化 retriever（避免每次调用都重新创建 Milvus 连接）"""
        if hasattr(self, '_retriever') and self._retriever is not None:
            return self._retriever
        from app.core.config import get_settings
        from app.core.dependencies import get_vector_store
        from app.rag.retriever import HybridRetriever, SparseRetriever

        settings = get_settings()
        vector_store = get_vector_store()
        sparse_retriever = SparseRetriever()
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            rrf_k=60,
            use_reranker=settings.RERANKER_ENABLED,
            reranker_model=settings.RERANKER_MODEL,
            top_n=settings.RERANKER_TOP_N
        )
        return self._retriever

    def call_rag_direct(self, query: str) -> Dict[str, Any]:
        """直接调用 Agentic RAG ChatService"""
        from app.db.session import SessionLocal
        from app.services.chat_service import ChatService

        retriever = self._init_retriever()

        db = SessionLocal()
        try:
            chat_service = ChatService(db, retriever)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # 使用 Agentic RAG（ReAct Commander 模式）
            result = loop.run_until_complete(
                chat_service.agentic_chat(user_id=1, query=query, max_attempts=2)
            )
            loop.close()
            
            # 提取用于评估的关键数据
            agentic_info = result.get("agentic_info", {})
            evaluation_result = {
                "answer": result.get("answer", ""),
                "retrieved_chunks": result.get("retrieved_chunks", []),
                # Agentic RAG 特有字段（从 agentic_info 中提取）
                "confidence": agentic_info.get("confidence", 0.0),
                "attempt_count": agentic_info.get("attempt_count", 0),
                "process_time": result.get("process_time", 0),
            }
            return evaluation_result
        finally:
            db.close()

    def evaluate_correctness(self, run: Run, example: Example) -> float:
        """用 LLM 评判答案正确性"""
        question = example.inputs.get("question", "")
        expected = example.outputs.get("expected_answer", "")
        actual = run.outputs.get("answer", "")

        if not expected:
            print(f"警告: 没有参考答案，跳过评估")
            return 0.5  # 中性分数

        # 简化 prompt 减少 token 数量和超时风险
        prompt = f"""评估答案正确性(0-10分):
问题: {question}
参考: {expected[:200]}
实际: {actual[:200]}
只输出数字。"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.evaluator_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=30,  # 添加超时设置
                )
                score_text = response.choices[0].message.content.strip()
                print(f"LLM 评分原始输出: {score_text}")
                score = float(score_text)
                return min(max(score / 10.0, 0), 1.0)
            except Exception as e:
                print(f"评分失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("使用基于规则的备选评估")
                    return self._rule_based_correctness(question, expected, actual)
                import time
                time.sleep(2)  # 重试前等待
        return 0.0

    def _rule_based_correctness(self, question: str, expected: str, actual: str) -> float:
        """基于规则的备选评估（当 LLM 超时时使用）"""
        import re
        score = 0.0

        # 1. 检查是否引用了法律条款
        if re.search(r'第[一二三四五六七八九十\d]+条', actual):
            score += 0.3

        # 2. 检查回答长度
        if len(actual) > 50:
            score += 0.2

        # 3. 检查是否包含明确判断
        if any(word in actual for word in ["可以", "不可以", "应当", "必须", "不得"]):
            score += 0.2

        # 4. 检查是否包含参考答案中的关键词
        expected_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', expected)
        if expected_keywords:
            matched = sum(1 for kw in expected_keywords[:10] if kw in actual)
            score += 0.3 * (matched / min(10, len(expected_keywords)))

        return min(score, 1.0)

    def evaluate_retrieval(self, run: Run, example: Example) -> float:
        """评估检索相关性"""
        # 尝试多种可能的字段名
        keywords = (
            example.inputs.get("context_keywords", []) or
            example.inputs.get("keywords", []) or
            example.outputs.get("keywords", [])
        )
        chunks = run.outputs.get("retrieved_chunks", [])

        if not chunks:
            print("警告: 没有检索到 chunks")
            return 0.0

        if not keywords:
            print("警告: 没有 keywords，使用问题中的关键词")
            # 从问题中提取关键词作为备选
            question = example.inputs.get("question", "")
            keywords = [question]  # 使用整个问题作为关键词

        content = " ".join([c.get("content", "") for c in chunks])
        matched = sum(1 for kw in keywords if kw in content)
        score = matched / len(keywords)
        print(f"检索评估: {matched}/{len(keywords)} = {score}")
        return score

    def evaluate_answer_quality(self, run: Run, example: Example) -> Dict[str, Any]:
        """基于规则的简单评估（不依赖 LLM）"""
        question = example.inputs.get("question", "")
        actual = run.outputs.get("answer", "")
        chunks = run.outputs.get("retrieved_chunks", [])

        score = 0.0
        feedback = []

        # 1. 检查是否有回答
        if not actual:
            return {"score": 0.0, "key": "answer_quality"}

        # 2. 检查是否引用了法律条款（有数字）
        import re
        has_article_refs = bool(re.search(r'第[一二三四五六七八九十\d]+条', actual))
        if has_article_refs:
            score += 0.3
            feedback.append("引用了法律条款")

        # 3. 检查回答长度（太短可能不完整）
        if len(actual) > 50:
            score += 0.2
            feedback.append("回答详细")

        # 4. 检查是否直接回答问题
        if any(word in actual for word in ["可以", "不可以", "应当", "必须", "不得"]):
            score += 0.2
            feedback.append("包含明确判断")

        # 5. 检查检索质量
        if chunks and len(chunks) > 0:
            score += 0.3
            feedback.append(f"检索到 {len(chunks)} 个相关片段")

        print(f"答案质量评估: {score} - {'; '.join(feedback)}")
        return {
            "score": min(score, 1.0),
            "key": "answer_quality",
            "comment": "; ".join(feedback)
        }

    def evaluate_agentic_metrics(self, run: Run, example: Example) -> Dict[str, Any]:
        """评估 Agentic RAG 特有指标"""
        confidence = run.outputs.get("confidence", 0.0)
        attempt_count = run.outputs.get("attempt_count", 0)
        process_time = run.outputs.get("process_time", 0)

        # 计算综合得分：置信度权重0.6，迭代效率权重0.4
        # 迭代次数越少越好（1次得满分，2次得0.5分）
        iteration_score = 1.0 if attempt_count <= 1 else max(0.5, 1.0 - (attempt_count - 1) * 0.3)
        final_score = (confidence * 0.6) + (iteration_score * 0.4)

        feedback = []
        if confidence >= 0.7:
            feedback.append(f"高置信度 ({confidence:.2f})")
        elif confidence >= 0.4:
            feedback.append(f"中等置信度 ({confidence:.2f})")
        else:
            feedback.append(f"低置信度 ({confidence:.2f})")
        
        feedback.append(f"迭代次数: {attempt_count}")
        feedback.append(f"处理时间: {process_time:.2f}s")

        print(f"Agentic 指标评估: {final_score:.2f} - {'; '.join(feedback)}")
        return {
            "score": round(final_score, 2),
            "key": "agentic_metrics",
            "comment": "; ".join(feedback)
        }

    def evaluate_ragas_faithfulness(self, run: Run, example: Example) -> Dict[str, Any]:
        """RAGAS faithfulness: 答案对上下文的忠诚度"""
        if not RAGAS_AVAILABLE:
            return {"score": 0.0, "key": "ragas_faithfulness", "comment": "RAGAS 未安装"}
        
        # RAGAS >= 1.0 需要显式配置 LLM 和 embeddings
        if RAGAS_REQUIRES_CONFIG and not self.ragas_configured:
            return {"score": 0.0, "key": "ragas_faithfulness", "comment": "RAGAS 未配置"}

        question = example.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        chunks = run.outputs.get("retrieved_chunks", [])
        contexts = [c.get("content", "") for c in chunks] if chunks else []

        if not answer or not contexts:
            return {"score": 0.0, "key": "ragas_faithfulness", "comment": "缺少 answer 或 contexts"}

        try:
            # RAGAS 需要特定的输入格式
            eval_dataset = Dataset.from_dict({
                "user_input": [question],
                "response": [answer],
                "retrieved_contexts": [contexts]
            })

            # 使用 RAGAS 的 faithfulness 指标（需要传入 llm 实例化）
            metric_instance = Faithfulness(llm=self.ragas_llm) if self.ragas_llm else Faithfulness()
            
            # 临时禁用 LangSmith 追踪，避免 rate limit 导致 RAGAS parse_run_traces 失败
            old_tracing = os.environ.get("LANGSMITH_TRACING")
            os.environ["LANGSMITH_TRACING"] = "false"
            try:
                result = ragas_evaluate(
                    eval_dataset,
                    metrics=[metric_instance]
                )
            finally:
                if old_tracing is not None:
                    os.environ["LANGSMITH_TRACING"] = old_tracing
                else:
                    os.environ.pop("LANGSMITH_TRACING", None)

            # 提取分数（EvaluationResult 通过 _scores_dict 属性访问）
            score = 0.0
            try:
                scores_dict = getattr(result, '_scores_dict', {})
                score_list = scores_dict.get("faithfulness", [])
                if score_list and len(score_list) > 0:
                    score_value = score_list[0]
                    if score_value is not None:
                        score = float(score_value)
            except (IndexError, TypeError, AttributeError, KeyError, ValueError) as ex:
                print(f"提取 faithfulness 分数时出错: {ex}")
                score = 0.0

            print(f"RAGAS faithfulness: {score}")
            return {
                "score": score,
                "key": "ragas_faithfulness"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"RAGAS faithfulness 评估失败: {e}")
            return {"score": 0.0, "key": "ragas_faithfulness", "comment": str(e)}

    def evaluate_ragas_answer_relevancy(self, run: Run, example: Example) -> Dict[str, Any]:
        """RAGAS answer_relevancy: 答案与问题的相关性"""
        if not RAGAS_AVAILABLE:
            return {"score": 0.0, "key": "ragas_answer_relevancy", "comment": "RAGAS 未安装"}
        
        # RAGAS >= 1.0 需要显式配置 LLM 和 embeddings
        if RAGAS_REQUIRES_CONFIG and not self.ragas_configured:
            return {"score": 0.0, "key": "ragas_answer_relevancy", "comment": "RAGAS 未配置"}

        question = example.inputs.get("question", "")
        answer = run.outputs.get("answer", "")

        if not answer:
            return {"score": 0.0, "key": "ragas_answer_relevancy", "comment": "缺少 answer"}

        try:
            eval_dataset = Dataset.from_dict({
                "user_input": [question],
                "response": [answer]
            })

            # 使用 RAGAS 的 answer_relevancy 指标（strictness=1 避免 Doubao API 不支持 n>1 的问题）
            metric_instance = AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings, strictness=1) if self.ragas_llm and self.ragas_embeddings else AnswerRelevancy()
            
            # 临时禁用 LangSmith 追踪，避免 rate limit 导致 RAGAS parse_run_traces 失败
            old_tracing = os.environ.get("LANGSMITH_TRACING")
            os.environ["LANGSMITH_TRACING"] = "false"
            try:
                result = ragas_evaluate(
                    eval_dataset,
                    metrics=[metric_instance]
                )
            finally:
                if old_tracing is not None:
                    os.environ["LANGSMITH_TRACING"] = old_tracing
                else:
                    os.environ.pop("LANGSMITH_TRACING", None)

            # 提取分数（EvaluationResult 通过 _scores_dict 属性访问）
            score = 0.0
            try:
                scores_dict = getattr(result, '_scores_dict', {})
                score_list = scores_dict.get("answer_relevancy", [])
                if score_list and len(score_list) > 0:
                    score_value = score_list[0]
                    if score_value is not None:
                        score = float(score_value)
            except (IndexError, TypeError, AttributeError, KeyError, ValueError) as ex:
                print(f"提取 answer_relevancy 分数时出错: {ex}")
                score = 0.0

            print(f"RAGAS answer_relevancy: {score}")
            return {
                "score": score,
                "key": "ragas_answer_relevancy"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"RAGAS answer_relevancy 评估失败: {e}")
            return {"score": 0.0, "key": "ragas_answer_relevancy", "comment": str(e)}

    def _generate_reference_with_llm(self, question: str, contexts: list) -> str:
        """当数据集缺少 reference/ground_truth 时，用 LLM 根据检索上下文生成参考答案"""
        if not contexts:
            return ""
        context_text = "\n\n".join(contexts[:3])  # 最多用前3个上下文
        prompt = f"""根据以下上下文信息，简要回答问题。只给出答案内容，不要添加额外解释。

上下文：
{context_text}

问题：{question}

参考答案："""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=30,
            )
            reference = response.choices[0].message.content.strip()
            print(f"LLM 生成参考答案: {reference[:100]}...")
            return reference
        except Exception as e:
            print(f"LLM 生成参考答案失败: {e}")
            return ""

    def evaluate_ragas_context_precision(self, run: Run, example: Example) -> Dict[str, Any]:
        """RAGAS context_precision: 上下文精确度"""
        if not RAGAS_AVAILABLE:
            return {"score": 0.0, "key": "ragas_context_precision", "comment": "RAGAS 未安装"}
        
        # RAGAS >= 1.0 需要显式配置 LLM 和 embeddings
        if RAGAS_REQUIRES_CONFIG and not self.ragas_configured:
            return {"score": 0.0, "key": "ragas_context_precision", "comment": "RAGAS 未配置"}

        question = example.inputs.get("question", "")
        chunks = run.outputs.get("retrieved_chunks", [])
        contexts = [c.get("content", "") for c in chunks] if chunks else []
        
        # context_precision 需要 reference 字段（参考上下文）
        reference = example.outputs.get("reference", "") or example.outputs.get("ground_truth", "")

        if not contexts:
            return {"score": 0.0, "key": "ragas_context_precision", "comment": "缺少 contexts"}
        
        # 当缺少 reference 时，用 LLM 根据检索上下文自动生成参考答案
        if not reference:
            print("数据集缺少 reference，使用 LLM 自动生成参考答案...")
            reference = self._generate_reference_with_llm(question, contexts)
            if not reference:
                return {"score": 0.0, "key": "ragas_context_precision", "comment": "缺少 reference 且 LLM 生成失败"}

        try:
            eval_dataset = Dataset.from_dict({
                "user_input": [question],
                "retrieved_contexts": [contexts],
                "reference": [reference]
            })

            # 使用 RAGAS 的 context_precision 指标（需要传入 llm 实例化）
            metric_instance = ContextPrecision(llm=self.ragas_llm) if self.ragas_llm else ContextPrecision()
            
            # 临时禁用 LangSmith 追踪，避免 rate limit 导致 RAGAS parse_run_traces 失败
            old_tracing = os.environ.get("LANGSMITH_TRACING")
            os.environ["LANGSMITH_TRACING"] = "false"
            try:
                result = ragas_evaluate(
                    eval_dataset,
                    metrics=[metric_instance]
                )
            finally:
                if old_tracing is not None:
                    os.environ["LANGSMITH_TRACING"] = old_tracing
                else:
                    os.environ.pop("LANGSMITH_TRACING", None)

            # 提取分数（EvaluationResult 通过 _scores_dict 属性访问）
            score = 0.0
            try:
                scores_dict = getattr(result, '_scores_dict', {})
                score_list = scores_dict.get("context_precision", [])
                if score_list and len(score_list) > 0:
                    score_value = score_list[0]
                    if score_value is not None:
                        score = float(score_value)
            except (IndexError, TypeError, AttributeError, KeyError, ValueError) as ex:
                print(f"提取 context_precision 分数时出错: {ex}")
                score = 0.0

            print(f"RAGAS context_precision: {score}")
            return {
                "score": score,
                "key": "ragas_context_precision"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"RAGAS context_precision 评估失败: {e}")
            return {"score": 0.0, "key": "ragas_context_precision", "comment": str(e)}

    def evaluate_ragas_context_recall(self, run: Run, example: Example) -> Dict[str, Any]:
        """RAGAS context_recall: 上下文召回率（需要 reference）"""
        if not RAGAS_AVAILABLE:
            return {"score": 0.0, "key": "ragas_context_recall", "comment": "RAGAS 未安装"}
        
        # RAGAS >= 1.0 需要显式配置 LLM 和 embeddings
        if RAGAS_REQUIRES_CONFIG and not self.ragas_configured:
            return {"score": 0.0, "key": "ragas_context_recall", "comment": "RAGAS 未配置"}

        question = example.inputs.get("question", "")
        # RAGAS 0.4.x 统一使用 reference 字段（兼容旧版 ground_truth）
        reference = example.outputs.get("reference", "") or example.outputs.get("ground_truth", "")
        chunks = run.outputs.get("retrieved_chunks", [])
        contexts = [c.get("content", "") for c in chunks] if chunks else []

        if not contexts:
            return {"score": 0.0, "key": "ragas_context_recall", "comment": "缺少 contexts"}

        # 当缺少 reference 时，用 LLM 根据检索上下文自动生成参考答案
        if not reference:
            print("数据集缺少 reference，使用 LLM 自动生成参考答案...")
            reference = self._generate_reference_with_llm(question, contexts)
            if not reference:
                return {"score": 0.0, "key": "ragas_context_recall", "comment": "缺少 reference 且 LLM 生成失败"}

        try:
            eval_dataset = Dataset.from_dict({
                "user_input": [question],
                "retrieved_contexts": [contexts],
                "reference": [reference]
            })

            # 使用 RAGAS 的 context_recall 指标（需要传入 llm 实例化）
            metric_instance = ContextRecall(llm=self.ragas_llm) if self.ragas_llm else ContextRecall()
            
            # 临时禁用 LangSmith 追踪，避免 rate limit 导致 RAGAS parse_run_traces 失败
            old_tracing = os.environ.get("LANGSMITH_TRACING")
            os.environ["LANGSMITH_TRACING"] = "false"
            try:
                result = ragas_evaluate(
                    eval_dataset,
                    metrics=[metric_instance]
                )
            finally:
                if old_tracing is not None:
                    os.environ["LANGSMITH_TRACING"] = old_tracing
                else:
                    os.environ.pop("LANGSMITH_TRACING", None)

            # 提取分数（EvaluationResult 通过 _scores_dict 属性访问）
            score = 0.0
            try:
                scores_dict = getattr(result, '_scores_dict', {})
                score_list = scores_dict.get("context_recall", [])
                if score_list and len(score_list) > 0:
                    score_value = score_list[0]
                    if score_value is not None:
                        score = float(score_value)
            except (IndexError, TypeError, AttributeError, KeyError, ValueError) as ex:
                print(f"提取 context_recall 分数时出错: {ex}")
                score = 0.0

            print(f"RAGAS context_recall: {score}")
            return {
                "score": score,
                "key": "ragas_context_recall"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"RAGAS context_recall 评估失败: {e}")
            return {"score": 0.0, "key": "ragas_context_recall", "comment": str(e)}


def target_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """RAG 目标函数"""
    query = inputs.get("question", "")
    print(f"\n处理问题: {query}")

    evaluator = RAGEvaluator()
    result = evaluator.call_rag_direct(query)

    return {
        "answer": result.get("answer", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        # 返回 Agentic 指标数据供评估使用
        "confidence": result.get("confidence", 0.0),
        "attempt_count": result.get("attempt_count", 0),
        "process_time": result.get("process_time", 0),
    }


def run_evaluation():
    """运行评估"""
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("错误: 未设置 LANGSMITH_API_KEY")
        print("请在 .env 文件中添加: LANGSMITH_API_KEY=your_key")
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "rag-evaluation")

    client = Client()

    # 获取已存在的数据集
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"使用数据集: {DATASET_NAME}")
    except Exception as e:
        print(f"错误: 找不到数据集 '{DATASET_NAME}'")
        print(f"请确认数据集名称，或在 LangSmith 网页上创建")
        return

    evaluator = RAGEvaluator()

    # 构建评估器列表（包含 Agentic RAG 特有指标）
    eval_list = [
        evaluator.evaluate_correctness,
        evaluator.evaluate_retrieval,
        evaluator.evaluate_answer_quality,
        evaluator.evaluate_agentic_metrics,  # Agentic RAG 特有指标
    ]

    # 如果 RAGAS 可用，添加 RAGAS 指标
    if RAGAS_AVAILABLE:
        print("RAGAS 可用，添加 RAGAS 评估指标...")
        eval_list.extend([
            evaluator.evaluate_ragas_faithfulness,
            evaluator.evaluate_ragas_answer_relevancy,
            evaluator.evaluate_ragas_context_precision,
            evaluator.evaluate_ragas_context_recall,
        ])
    else:
        print("RAGAS 不可用，仅使用基础评估指标")

    print("\n开始评估...")
    results = evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=eval_list,
        max_concurrency=1,
    )

    print("\n评估完成！")


if __name__ == "__main__":
    run_evaluation()
