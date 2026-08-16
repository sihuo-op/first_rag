from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
from pathlib import Path
import os
from dotenv import dotenv_values

# 项目根目录的 .env 文件
ENV_FILE_PATH = Path(__file__).parent.parent.parent.parent / ".env"

# 将 .env 中的变量加载到 os.environ（确保 HF_ENDPOINT 等第三方库可读取）
for k, v in dotenv_values(ENV_FILE_PATH).items():
    if k not in os.environ:
        os.environ[k] = v

local_no_proxy = "localhost,127.0.0.1,::1"
for key in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(key, "")
    values = [item.strip() for item in existing.split(",") if item.strip()]
    for value in local_no_proxy.split(","):
        if value not in values:
            values.append(value)
    os.environ[key] = ",".join(values)


class Settings(BaseSettings):
    """
    应用配置管理

    使用 Pydantic Settings 管理所有配置项，支持从 .env 文件和
    环境变量加载配置。

    配置分类：
    - 应用基础：APP_NAME、APP_ENV、APP_DEBUG
    - JWT 认证：密钥、算法、令牌过期时间
    - 数据库：SQLite 连接 URL
    - Milvus：向量数据库连接配置
    - Embedding：统一配置，通过 EMBEDDING_USE_LOCAL 切换本地/云服务
    - Chat：OpenAI 兼容接口，支持豆包、OpenAI 等（统一配置）
    - RAG：重排序器、混合检索、文档切分参数
    - 文件上传：大小限制、允许的扩展名、存储目录
    """
    APP_NAME: str = "Agentic RAG"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True

    JWT_SECRET_KEY: str = "super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "sqlite:///./data/sqlite/app.db"

    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_PREFIX: str = "rag_"

    # Embedding 配置（使用本地模型）
    # 所有参数必须在 .env 文件中配置，代码中不设默认值
    EMBEDDING_USE_LOCAL: bool                      # true=本地模型
    EMBEDDING_DIMENSION: int                       # 向量维度

    # 本地 Embedding 模型（EMBEDDING_USE_LOCAL=true 时使用）
    SENTENCE_TRANSFORMER_MODEL: str                # 本地模型名称

    # Chat 配置（OpenAI 兼容接口，支持豆包、OpenAI、智谱等）
    # 所有参数必须在 .env 文件中配置
    CHAT_API_KEY: Optional[str] = None             # API密钥（可选，无则返回mock响应）
    CHAT_API_BASE: str                             # API地址
    CHAT_MODEL: str                                # 模型名称

    # 分离的 LLM 配置（用于 RAG 中的生成和改写）
    GENERATION_LLM_MODEL: str = ""                 # 生成用 LLM 模型名称（默认使用 CHAT_MODEL）
    GENERATION_LLM_TEMPERATURE: float = 0.1       # 生成用 LLM 温度（较低值确保输出稳定）
    GENERATION_LLM_MAX_TOKENS: int = 2000          # 生成用 LLM 最大 token 数
    REWRITE_LLM_MODEL: str = ""                   # 改写用 LLM 模型名称（默认使用 CHAT_MODEL）
    REWRITE_LLM_TEMPERATURE: float = 0.3          # 改写用 LLM 温度
    REWRITE_LLM_MAX_TOKENS: int = 500             # 改写用 LLM 最大 token 数
    EVALUATION_LLM_MODEL: str = ""               # 评估用 LLM 模型名称（默认使用 GENERATION_LLM_MODEL）
    EVALUATION_LLM_TEMPERATURE: float = 0.0       # 评估用 LLM 温度
    EVALUATION_LLM_MAX_TOKENS: int = 300          # 评估用 LLM 最大 token 数

    # Reranker 配置（所有参数必须在 .env 文件中配置）
    RERANKER_ENABLED: bool                         # 是否启用重排序
    RERANKER_MODEL: str                            # CrossEncoder 模型名称
    RERANKER_TOP_N: int                            # 重排序后保留数量

    HYBRID_SEARCH_ENABLED: bool = True
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3

    CHUNK_LARGE_SIZE: int = 2000
    CHUNK_MEDIUM_SIZE: int = 500
    CHUNK_SMALL_SIZE: int = 150
    CHUNK_OVERLAP: int = 50

    # Memory 配置
    MEMORY_ENABLED: bool = True
    MEMORY_COLLECTION_NAME: str = "memories"
    MEMORY_RETRIEVAL_TOP_K: int = 5
    MEMORY_RECENT_MESSAGE_LIMIT: int = 10
    MEMORY_SUMMARY_TARGET_TOKENS: int = 1200
    MEMORY_EXTRACTION_MAX_ITEMS: int = 5
    CHAT_MODEL_CONTEXT_WINDOW: int = 128000
    REWRITE_MODEL_CONTEXT_WINDOW: int = 32000
    MEMORY_CONTEXT_RATIO: float = 0.8
    MEMORY_RESERVED_OUTPUT_TOKENS: int = 4000
    MEMORY_RESERVED_RAG_TOKENS: int = 20000
    MEMORY_MAX_COMPRESS_ROUNDS: int = 2

    MAX_UPLOAD_SIZE: int = 52428800
    ALLOWED_EXTENSIONS: str = "pdf,txt,docx,md"

    # Chunk 生命周期配置
    # 冲突检测：高置信度自动作废，低置信度转人工审核
    CONFLICT_DETECTION_HIGH_CONFIDENCE: float = 0.85
    CONFLICT_DETECTION_LOW_CONFIDENCE: float = 0.5

    # 冷知识识别规则
    COLD_KNOWLEDGE_TIMEOUT_DAYS: int = 90              # last_accessed_at 超过此天数归档
    COLD_KNOWLEDGE_LOW_FREQ_THRESHOLD: int = 2         # access_count 低于此值
    COLD_KNOWLEDGE_LOW_FREQ_MIN_DAYS: int = 30         # 上传超过此天数才适用频次规则
    COLD_KNOWLEDGE_LOW_QUALITY_SCORE: float = 0.3      # avg_score 低于此值
    COLD_KNOWLEDGE_LOW_QUALITY_MIN_HITS: int = 5       # hit_count 达到此值才适用质量规则
    COLD_KNOWLEDGE_ARCHIVE_RETENTION_DAYS: int = 90    # 归档后保留天数，过期硬删

    # 调度 cron
    COLD_KNOWLEDGE_SWEEP_CRON: str = "0 3 * * *"       # 冷知识扫描（每天 3 点）
    HARD_DELETE_SWEEP_CRON: str = "0 4 * * *"          # 硬删除扫描（每天 4 点）

    # Knowledge Graph 配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme"
    KG_ENABLED: bool = True
    KG_CONCEPT_SIMILARITY_THRESHOLD: float = 0.7
    KG_MULTI_HOP_DEPTH: int = 2
    KG_EXTRACTION_LLM_MODEL: str = ""
    KG_EXTRACTION_LLM_TEMPERATURE: float = 0.0
    KG_EXTRACTION_LLM_MAX_TOKENS: int = 1000

    UPLOAD_DIR: str = "./data/uploads"
    SQLITE_DIR: str = "./data/sqlite"

    FIRST_ADMIN_USERNAME: str = "admin"
    FIRST_ADMIN_PASSWORD: str = "admin123"
    FIRST_ADMIN_EMAIL: str = "admin@example.com"

    # OpenTelemetry 配置
    OTEL_ENABLED: bool = True
    OTEL_SERVICE_NAME: str = "first-rag-backend"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SAMPLING_RATE: float = 1.0

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    class Config:
        env_file = str(ENV_FILE_PATH)
        case_sensitive = True
        extra = "ignore"  # 忽略 .env 中的额外字段如 HF_ENDPOINT


@lru_cache()
def get_settings() -> Settings:
    return Settings()
