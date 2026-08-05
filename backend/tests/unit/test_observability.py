"""验证 observability.py 的基本行为（不依赖 Jaeger 运行）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_setup_otel_disabled_is_noop():
    """OTEL_ENABLED=False 时 setup_otel 不应抛异常，也不应设置 provider。"""
    from app.core import observability
    observability._initialized = False  # 重置模块状态

    settings = SimpleNamespace(
        OTEL_ENABLED=False,
        OTEL_SERVICE_NAME="test",
        OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317",
        OTEL_SAMPLING_RATE=1.0,
    )

    observability.setup_otel(settings)

    assert observability._initialized is False


def test_get_tracer_returns_tracer():
    """get_tracer 应返回非 None 的 tracer 对象。"""
    from app.core import observability
    from opentelemetry import trace

    tracer = observability.get_tracer("test")
    assert tracer is not None
    assert hasattr(tracer, "start_as_current_span")


def test_instrument_app_without_init_is_noop():
    """未初始化时 instrument_app 应直接返回，不抛异常。"""
    from app.core import observability
    observability._initialized = False

    # 传入 None 作为 app，不应抛异常（应直接 return）
    observability.instrument_app(None)
