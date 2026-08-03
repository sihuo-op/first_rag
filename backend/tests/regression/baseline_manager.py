"""基线分数管理 - 保存/加载历史评估结果"""
import json
from pathlib import Path
from datetime import datetime

BASELINE_DIR = Path(__file__).parent.parent / "baselines"


def save_baseline(scores: dict, profile_name: str = "default"):
    """保存当前评估结果为基线"""
    BASELINE_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存带时间戳的历史版本
    filepath = BASELINE_DIR / f"{profile_name}_baseline_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "profile": profile_name,
            "scores": scores,
        }, f, ensure_ascii=False, indent=2)

    # 同时保存为 latest（方便对比）
    latest_path = BASELINE_DIR / f"{profile_name}_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "profile": profile_name,
            "scores": scores,
        }, f, ensure_ascii=False, indent=2)

    return filepath


def load_latest_baseline(profile_name: str = "default") -> dict:
    """加载最近一次基线"""
    latest_path = BASELINE_DIR / f"{profile_name}_latest.json"
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_baselines(profile_name: str = "default") -> list:
    """列出所有历史基线"""
    if not BASELINE_DIR.exists():
        return []
    files = sorted(BASELINE_DIR.glob(f"{profile_name}_baseline_*.json"))
    return [f.name for f in files]
