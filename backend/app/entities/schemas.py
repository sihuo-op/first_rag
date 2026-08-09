"""
API 请求/响应模型定义 (Pydantic Schemas)

定义了 FastAPI 接口的输入输出数据结构，用于请求验证和响应序列化。
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """用户角色枚举 - 用于 API 层"""
    USER = "user"
    ADMIN = "admin"


class DocumentStatus(str, Enum):
    """文档状态枚举 - 用于 API 层"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryType(str, Enum):
    """长期记忆类型枚举"""
    PROFILE = "profile"
    PREFERENCE = "preference"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    CASE_FACT = "case_fact"


class MessageRole(str, Enum):
    """消息角色枚举 - 用于 API 层"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Token(BaseModel):
    """
    认证令牌响应

    用户登录成功后返回的 JWT 令牌信息。
    """
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """
    令牌数据

    从 JWT 令牌中解析出的用户信息。
    """
    username: Optional[str] = None


class UserBase(BaseModel):
    """
    用户基础模型

    包含用户的基本信息，作为创建和响应模型的父类。
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    """
    用户创建请求模型

    用于用户注册，继承基础模型并添加密码字段。
    """
    password: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    """
    用户登录请求模型

    用于用户认证，包含用户名和密码。
    """
    username: str
    password: str


class UserResponse(UserBase):
    """
    用户响应模型

    返回给客户端的用户信息，不包含敏感数据（如密码）。
    """
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """
    用户更新请求模型

    用于更新用户信息，所有字段均为可选。
    """
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None


class DocumentBase(BaseModel):
    """
    文档基础模型

    包含文档的基本信息。
    """
    title: str
    file_name: str


class DocumentCreate(DocumentBase):
    """
    文档创建请求模型

    用于上传新文档，包含文件路径和类型信息。
    """
    file_path: str
    file_type: str
    file_size: Optional[int] = None


class DocumentResponse(DocumentBase):
    """
    文档响应模型

    返回给客户端的文档信息，包含处理状态。
    """
    id: int
    file_type: str
    file_size: Optional[int] = None
    user_id: int
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentUpdate(BaseModel):
    """
    文档更新请求模型

    用于更新文档标题。
    """
    title: Optional[str] = None


class ChunkResponse(BaseModel):
    """
    文档片段响应模型

    返回文档切分后的片段信息。
    """
    id: int
    document_id: int
    chunk_type: str
    content: str
    position: Optional[int] = None
    token_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """
    对话会话基础模型

    包含会话的基本信息。
    """
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """
    对话会话创建请求模型

    用于创建新的对话会话。
    """
    pass


class ConversationResponse(ConversationBase):
    """
    对话会话响应模型

    返回给客户端的会话信息。
    """
    id: int
    user_id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConversationUpdate(BaseModel):
    """
    对话会话更新请求模型

    用于更新会话标题。
    """
    title: Optional[str] = None


class ConversationSummaryResponse(BaseModel):
    """会话滚动摘要响应模型"""
    id: int
    conversation_id: int
    summary: str
    last_summarized_message_id: Optional[int] = None
    message_count: int = 0
    summary_token_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserMemoryResponse(BaseModel):
    """用户长期记忆响应模型"""
    id: int
    user_id: int
    conversation_id: Optional[int] = None
    memory_type: MemoryType
    content: str
    source_message_ids: Optional[List[int]] = None
    importance: float = 0.5
    status: str
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatMessageBase(BaseModel):
    """
    对话消息基础模型

    包含消息的基本内容。
    """
    content: str


class ChatMessageCreate(ChatMessageBase):
    """
    对话消息创建请求模型

    用于创建新消息。
    """
    pass


class ChatMessageResponse(ChatMessageBase):
    """
    对话消息响应模型

    返回消息详情，包括检索到的片段和调试信息。
    """
    id: int
    conversation_id: int
    role: MessageRole
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[Dict[str, Any]] = None  # RAG 检索调试信息
    process_time: Optional[int] = None  # 处理耗时（毫秒）
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMode(str, Enum):
    """对话模式枚举"""
    TRADITIONAL = "traditional"  # 传统 RAG（单次检索）
    AGENTIC = "agentic"          # Agentic RAG（迭代检索）


class ChatRequest(BaseModel):
    """
    对话请求模型

    RAG 问答的核心接口，支持多种配置选项。

    参数：
    - query: 用户问题
    - conversation_id: 会话 ID（可选，不传则创建新会话）
    - use_rag: 是否使用检索增强（默认 True）
    - mode: 对话模式（默认 agentic）
    - stream: 是否流式返回（默认 False）
    """
    query: str
    conversation_id: Optional[int] = None
    use_rag: bool = True
    mode: ChatMode = ChatMode.AGENTIC
    stream: bool = False


class DebugInfo(BaseModel):
    """
    调试信息模型

    用于追踪 RAG 流程的各个步骤，便于排查问题和优化检索效果。

    字段说明：
    - original_query: 原始用户问题
    - rewritten_query: LLM 改写后的查询（可选）
    - retrieval_steps: 检索步骤记录（向量化、检索、融合、去重、重排序）
    - total_chunks_retrieved: 总检索到的 chunk 数
    - chunks_by_type: 按类型统计 chunk 数（small/medium/large）
    - rerank_used: 是否使用了 CrossEncoder 重排序
    - final_context: 最终发给 LLM 的上下文
    - llm_messages_count: 发给 LLM 的消息数
    - detail: 详细检索结果（用于展开查看各阶段结果）
      - dense_by_type: 密集向量检索按粒度分类结果
      - sparse_results: 稀疏检索（BM25）结果
      - merged_results: 分数融合后结果
      - deduped_results: 去重后结果
      - reranked_results: 重排序后最终结果
    """
    original_query: str                              # 原始用户问题
    rewritten_query: Optional[str] = None            # LLM 改写后的查询
    retrieval_steps: List[Dict[str, Any]] = []       # 检索步骤记录
    total_chunks_retrieved: int = 0                  # 总检索到的 chunk 数
    candidate_chunks_retrieved: int = 0              # 候选检索 chunk 数
    chunks_by_type: Dict[str, int] = {}              # 按类型统计 chunk 数
    rerank_used: bool = False                        # 是否使用了 rerank
    final_context: Optional[str] = None               # 最终发给 LLM 的上下文
    llm_messages_count: int = 0                      # 发给 LLM 的消息数
    detail: Optional[Dict[str, Any]] = None          # 详细检索结果（各阶段）
    step_timings: List[Dict[str, Any]] = []          # 每个阶段的耗时
    generation_time: float = 0.0                    # 答案生成耗时
    agentic_info: Optional[Dict[str, Any]] = None    # MainAgent 并行编排信息
    memory_info: Optional[Dict[str, Any]] = None     # 长期记忆和 query rewrite 信息


class AgenticDebugInfo(BaseModel):
    """
    Agentic RAG 调试信息模型

    记录 Agentic RAG 的迭代检索过程
    """
    mode: str = "agentic"
    attempt_count: int = 0                              # 尝试次数
    query_history: List[str] = []                       # 查询历史（包括改写）
    confidence: float = 0.0                             # 最终置信度
    evaluation_reason: str = ""                         # 评估理由
    execution_log: List[Dict[str, Any]] = []           # 执行步骤日志
    step_timings: List[Dict[str, Any]] = []            # 每个阶段的耗时
    generation_time: float = 0.0                        # 答案生成耗时
    tool_calls: List[Dict[str, Any]] = []                # 工具调用记录
    retrieved: bool = False                              # 是否使用检索
    commander_elapsed_time: float = 0.0                  # MainAgent 总耗时
    evaluation_grade: str = "unknown"                   # 评估等级
    all_iterations: List[Dict[str, Any]] = []            # 所有迭代记录
    decomposed_tasks: List[Dict[str, Any]] = []          # 拆分后的子任务
    sub_tasks: List[Dict[str, Any]] = []                 # 子任务执行结果
    candidate_documents: List[Dict[str, Any]] = []       # 候选检索片段


class ChatResponse(BaseModel):
    """
    对话响应模型

    RAG 问答的返回结果。
    """
    answer: str
    conversation_id: int
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    process_time: Optional[float] = None              # 处理耗时（秒）
    debug_info: Optional[DebugInfo] = None            # 传统 RAG 调试信息
    agentic_info: Optional[AgenticDebugInfo] = None   # Agentic RAG 调试信息


class RetrieveRequest(BaseModel):
    """
    检索请求模型

    用于直接调用检索功能，不经过 LLM 生成。
    """
    query: str
    top_k: int = 10


class RetrieveResponse(BaseModel):
    """
    检索响应模型

    返回检索到的文档片段。
    """
    chunks: List[Dict[str, Any]]
    total: int


class HealthResponse(BaseModel):
    """
    健康检查响应模型

    用于监控系统状态。
    """
    status: str
    milvus_connected: bool
    database_connected: bool


class StatsResponse(BaseModel):
    """
    统计信息响应模型

    返回系统的各项统计数据。
    """
    total_users: int
    total_documents: int
    total_conversations: int
    total_messages: int


class ChunkDetailResponse(BaseModel):
    """chunk 详情响应（含冲突与统计字段）"""
    id: int
    document_id: int
    chunk_type: str
    content: str
    milvus_id: Optional[str] = None
    content_hash: Optional[str] = None
    status: str = "active"
    # 冲突
    conflict_with_chunk_id: Optional[str] = None
    conflict_with_content: Optional[str] = None  # join 出来的新 chunk 内容
    conflict_detected_at: Optional[datetime] = None
    confidence: Optional[float] = None
    review_reason: Optional[str] = None
    superseded_at: Optional[datetime] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    # 统计
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    hit_count: int = 0
    avg_score: float = 0.0
    # 归档
    archived_reason: Optional[str] = None
    archived_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChunkPatchRequest(BaseModel):
    """admin PATCH chunk 动作"""
    action: str  # confirm | dismiss | archive | restore | hard_delete


class DocumentWithConflictStatusResponse(DocumentResponse):
    """文档响应（带冲突检测状态）"""
    conflict_check_status: str = "completed"
    conflict_check_progress: Optional[str] = None
    conflict_check_completed_at: Optional[datetime] = None
