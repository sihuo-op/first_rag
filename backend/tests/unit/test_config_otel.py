"""验证 Settings 类的 OTEL_* 字段存在且有正确默认值。

依赖项目根目录的 .env 文件存在（与现有 tests/layers/ 测试风格一致）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import get_settings


def test_otel_settings_fields_exist_with_defaults():
    """OTEL_* 字段应存在；当 .env 未设置时使用默认值。"""
    settings = get_settings()

    assert hasattr(settings, "OTEL_ENABLED")
    assert hasattr(settings, "OTEL_SERVICE_NAME")
    assert hasattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT")
    assert hasattr(settings, "OTEL_SAMPLING_RATE")

    # 默认值检查（如果 .env 中未设置 OTEL_*，则使用 Settings 类中的默认值）
    assert settings.OTEL_ENABLED is True
    assert settings.OTEL_SERVICE_NAME == "first-rag-backend"
    assert settings.OTEL_EXPORTER_OTLP_ENDPOINT == "http://localhost:4317"
    assert settings.OTEL_SAMPLING_RATE == 1.0
