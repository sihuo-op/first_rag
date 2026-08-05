"""OpenTelemetry 初始化与 instrumentor 注册。

初始化失败不拖垮应用：setup_otel 内部 try/except，失败时只 print 警告。
OTEL_ENABLED=False 时零开销，直接 return。
"""
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

_initialized = False


def setup_otel(settings) -> None:
    """配置 TracerProvider + OTLP gRPC exporter。失败时只 print 警告，不抛异常。"""
    global _initialized
    if _initialized:
        return
    if not settings.OTEL_ENABLED:
        print("[OTel] disabled by OTEL_ENABLED=False")
        return
    try:
        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(settings.OTEL_SAMPLING_RATE),
        )
        exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
        print(
            f"[OTel] tracing enabled: service={settings.OTEL_SERVICE_NAME}, "
            f"endpoint={settings.OTEL_EXPORTER_OTLP_ENDPOINT}, "
            f"sampling_rate={settings.OTEL_SAMPLING_RATE}"
        )
    except Exception as e:
        print(f"[OTel] init failed, tracing disabled: {e}")


def instrument_app(app: FastAPI, engine=None) -> None:
    """注册 FastAPI / httpx / SQLAlchemy 自动埋点。必须在所有 middleware 和 router 注册之后调用。

    Args:
        app: FastAPI 实例
        engine: SQLAlchemy engine（已存在的 engine 必须显式传入，否则 query 级 span 不生效）
    """
    if not _initialized:
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        if engine is not None:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        else:
            SQLAlchemyInstrumentor().instrument()
        print("[OTel] instrumentors registered: fastapi, httpx, sqlalchemy")
    except Exception as e:
        print(f"[OTel] instrument_app failed: {e}")


def get_tracer(name: str):
    """获取 tracer，供手动 span 使用。"""
    return trace.get_tracer(name)
