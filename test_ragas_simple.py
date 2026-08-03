import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("CHAT_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("CHAT_API_BASE", "")

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness

test_data = {
    "user_input": ["老板让我签空白合同，我可以拒绝吗？"],
    "response": ["可以拒绝。"],
    "retrieved_contexts": [["《劳动合同法》规定。"]]
}

dataset = Dataset.from_dict(test_data)

try:
    result = evaluate(dataset, metrics=[faithfulness])
    print(f"Result: {result}")
    
    # 使用正确的方式访问分数
    scores_dict = getattr(result, '_scores_dict', {})
    score_list = scores_dict.get("faithfulness", [])
    score_value = score_list[0] if score_list else None
    score = float(score_value) if score_value is not None else 0.0
    
    print(f"Score: {score}")
    print("提取分数成功！")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
