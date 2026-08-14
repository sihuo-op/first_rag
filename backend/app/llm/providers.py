"""
LLM provider 创建与线程安全调用工具。
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel

_llm_locks = {}
_llm_locks_guard = threading.Lock()


def invoke_llm_threadsafe(llm, messages):
    llm_id = id(llm)
    with _llm_locks_guard:
        lock = _llm_locks.get(llm_id)
        if lock is None:
            lock = threading.Lock()
            _llm_locks[llm_id] = lock
    with lock:
        return llm.invoke(messages)


# ============ LLM 配置和提供者 ============

class LLMProviderType(Enum):
    OPENAI = "openai"
    ZHIPU = "zhipu"
    DOUBAO = "doubao"
    LOCAL = "local"


@dataclass
class LLMConfig:
    provider: LLMProviderType = LLMProviderType.OPENAI
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """LLM 提供者基类"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm: Optional[BaseChatModel] = None

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供者（支持兼容接口）"""
    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            from app.core.config import get_settings
            _settings = get_settings()
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                openai_api_key=self.config.api_key or getattr(_settings, "CHAT_API_KEY", ""),
                openai_api_base=self.config.api_base or getattr(_settings, "CHAT_API_BASE", ""),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                request_timeout=self.config.timeout,
            )
        return self._llm


def get_provider(config: LLMConfig) -> BaseLLMProvider:
    """工厂函数"""
    providers = {LLMProviderType.OPENAI: OpenAIProvider, LLMProviderType.ZHIPU: OpenAIProvider, LLMProviderType.DOUBAO: OpenAIProvider}
    return providers.get(config.provider, OpenAIProvider)(config)


def get_generation_llm() -> BaseChatModel:
    """获取生成用 LLM（中等温度）"""
    from app.core.config import get_settings
    _settings = get_settings()
    model_name = _settings.GENERATION_LLM_MODEL or _settings.CHAT_MODEL
    config = LLMConfig(
        provider=LLMProviderType.OPENAI,
        model_name=model_name,
        temperature=_settings.GENERATION_LLM_TEMPERATURE,
        max_tokens=_settings.GENERATION_LLM_MAX_TOKENS,
    )
    return OpenAIProvider(config).get_llm()


def get_rewrite_llm() -> BaseChatModel:
    """获取改写用 LLM（低温度）"""
    from app.core.config import get_settings
    _settings = get_settings()
    model_name = _settings.REWRITE_LLM_MODEL or _settings.CHAT_MODEL
    config = LLMConfig(
        provider=LLMProviderType.OPENAI,
        model_name=model_name,
        temperature=_settings.REWRITE_LLM_TEMPERATURE,
        max_tokens=_settings.REWRITE_LLM_MAX_TOKENS,
    )
    return OpenAIProvider(config).get_llm()


def get_evaluation_llm() -> BaseChatModel:
    """获取评估用 LLM（低温、短输出）。"""
    from app.core.config import get_settings
    _settings = get_settings()
    model_name = _settings.EVALUATION_LLM_MODEL or _settings.GENERATION_LLM_MODEL or _settings.CHAT_MODEL
    config = LLMConfig(
        provider=LLMProviderType.OPENAI,
        model_name=model_name,
        temperature=_settings.EVALUATION_LLM_TEMPERATURE,
        max_tokens=_settings.EVALUATION_LLM_MAX_TOKENS,
    )
    return OpenAIProvider(config).get_llm()


def get_extraction_llm() -> BaseChatModel:
    """抽取用 LLM：温度 0，输出 JSON 格式严格。"""
    from app.core.config import get_settings
    from langchain_openai import ChatOpenAI
    s = get_settings()
    model = s.KG_EXTRACTION_LLM_MODEL or s.CHAT_MODEL
    return ChatOpenAI(
        model=model,
        openai_api_key=s.CHAT_API_KEY,
        openai_api_base=s.CHAT_API_BASE,
        temperature=s.KG_EXTRACTION_LLM_TEMPERATURE,
        max_tokens=s.KG_EXTRACTION_LLM_MAX_TOKENS,
        request_timeout=60,
    )
