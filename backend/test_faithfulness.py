"""单独测试 Faithfulness 指标，看详细过程"""
import sys
sys.path.insert(0, '/app')

import tests.ragas_patch

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from ragas.llms import LangchainLLMWrapper

from app.core.config import get_settings
settings = get_settings()

lc_llm = LangchainChatOpenAI(
    model=settings.CHAT_MODEL,
    api_key=settings.CHAT_API_KEY,
    base_url=settings.CHAT_API_BASE,
    temperature=0,
    max_tokens=8192,
)
ragas_llm = LangchainLLMWrapper(lc_llm)

# 测试一个简单的样本
sample = SingleTurnSample(
    user_input="用人单位应当保证劳动者每周休息几天？",
    response="用人单位应当保证劳动者每周至少休息一日。（第三十八条，文档片段1）",
    retrieved_contexts=["第三十八条 用人单位应当保证劳动者每周至少休息一日。"],
)

dataset = EvaluationDataset(samples=[sample])
metric = Faithfulness(llm=ragas_llm)

import os
old_tracing = os.environ.get("LANGSMITH_TRACING")
os.environ["LANGSMITH_TRACING"] = "false"

try:
    result = evaluate(dataset=dataset, metrics=[metric], llm=ragas_llm)
except Exception as e:
    print(f"Error: {e}")
    result = None
finally:
    if old_tracing is not None:
        os.environ["LANGSMITH_TRACING"] = old_tracing
    else:
        os.environ.pop("LANGSMITH_TRACING", None)

if result:
    print(f"\nResult type: {type(result)}")
    print(f"Result: {result}")
    if hasattr(result, '_scores_dict'):
        print(f"Scores dict: {result._scores_dict}")
    if hasattr(result, 'scores'):
        print(f"Scores: {result.scores}")
    if hasattr(result, '__dict__'):
        for k, v in result.__dict__.items():
            print(f"  {k}: {v}")
