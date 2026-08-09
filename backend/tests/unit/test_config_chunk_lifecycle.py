"""验证 chunk 生命周期相关配置字段存在且有正确默认值。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import get_settings


def test_conflict_detection_settings_exist_with_defaults():
    settings = get_settings()
    assert hasattr(settings, "CONFLICT_DETECTION_HIGH_CONFIDENCE")
    assert hasattr(settings, "CONFLICT_DETECTION_LOW_CONFIDENCE")
    assert settings.CONFLICT_DETECTION_HIGH_CONFIDENCE == 0.85
    assert settings.CONFLICT_DETECTION_LOW_CONFIDENCE == 0.5


def test_cold_knowledge_settings_exist_with_defaults():
    settings = get_settings()
    assert settings.COLD_KNOWLEDGE_TIMEOUT_DAYS == 90
    assert settings.COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD == 2
    assert settings.COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS == 30
    assert settings.COLD_KNOWLEDGE_LOW_QUALITY_SCORE == 0.3
    assert settings.COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS == 5
    assert settings.COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS == 90
    assert settings.COLD_KNOWLEDGE_SWEEP_CRON == "0 3 * * *"
    assert settings.HARD_DELETE_SWEEP_CRON == "0 4 * * *"
