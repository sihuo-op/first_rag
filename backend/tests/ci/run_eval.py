"""CI 入口脚本 - 一键跑全层评估"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from tests.regression.baseline_manager import save_baseline, load_latest_baseline
from tests.regression.regression_detector import detect_regression, check_thresholds


def run_full_eval(save_baseline_flag=False):
    """运行全层评估"""
    start_time = time.time()
    print("=" * 60)
    print("RAG 全层评估 (L1 + L2 + L3)")
    print("=" * 60)

    all_layer_scores = {}
    all_passed = True

    # L1 检索层
    print("\n" + "=" * 60)
    print("L1 检索层")
    print("=" * 60)
    try:
        from tests.layers.test_retrieval import run_retrieval_layer
        l1_scores = run_retrieval_layer()
        all_layer_scores["retrieval"] = l1_scores
    except Exception as e:
        print(f"L1 检索层失败: {e}")
        all_layer_scores["retrieval"] = {}
        all_passed = False

    # L2 生成层
    print("\n" + "=" * 60)
    print("L2 生成层")
    print("=" * 60)
    try:
        from tests.layers.test_generation import run_generation_layer
        l2_scores = run_generation_layer()
        all_layer_scores["generation"] = l2_scores
    except Exception as e:
        print(f"L2 生成层失败: {e}")
        all_layer_scores["generation"] = {}
        all_passed = False

    # L3 端到端层
    print("\n" + "=" * 60)
    print("L3 端到端层")
    print("=" * 60)
    try:
        from tests.layers.test_e2e import main as run_e2e
        l3_passed = run_e2e()
        all_passed = all_passed and l3_passed
    except Exception as e:
        print(f"L3 端到端层失败: {e}")
        all_passed = False

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"全层评估完成，总耗时: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # 保存基线
    if save_baseline_flag and all_layer_scores.get("e2e"):
        save_baseline(all_layer_scores["e2e"])
        print("已保存为新基线")

    return all_passed


if __name__ == "__main__":
    save_baseline_flag = "--save-baseline" in sys.argv
    success = run_full_eval(save_baseline_flag)
    sys.exit(0 if success else 1)
