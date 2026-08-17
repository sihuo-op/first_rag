"""
数据库模型定义

定义了 RAG 系统的 SQLAlchemy ORM 模型，包括用户、文档、文档片段、会话和消息。
"""

from sqlalchemy import (
    Column, Integer, String, Text, BigInteger, Boolean, DateTime,
    ForeignKey, JSON, Enum as SQLEnum, func, Float, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """
    用户角色枚举

    用于区分普通用户和管理员权限。
    """
    USER = "user"       # 普通用户，可上传文档和进行问答
    ADMIN = "admin"     # 管理员，拥有系统管理权限


class DocumentStatus(str, enum.Enum):
    """
    文档处理状态枚举

    追踪文档从上传到处理完成的整个生命周期。
    """
    PENDING = "pending"           # 待处理，文档刚上传
    PROCESSING = "processing"     # 处理中，正在分块和向量化
    COMPLETED = "completed"       # 处理完成，可用于检索
    FAILED = "failed"             # 处理失败，记录错误信息


class ChunkType(str, enum.Enum):
    """
    文档片段类型枚举

    三层分块策略，不同粒度用于不同检索场景：
    - SMALL: 精确匹配，适合短查询
    - MEDIUM: 平衡粒度，适合一般查询
    - LARGE: 完整上下文，适合需要背景信息的查询
    """
    LARGE = "large"     # 大片段（~2000字），提供完整语义上下文
    MEDIUM = "medium"   # 中片段（~500字），句子组
    SMALL = "small"     # 小片段（~150字），精确匹配


class MemoryType(str, enum.Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    CASE_FACT = "case_fact"


class MessageRole(str, enum.Enum):
    """
    对话消息角色枚举

    区分对话中的不同参与者角色。
    """
    USER = "user"           # 用户消息
    ASSISTANT = "assistant" # AI 助手回复
    SYSTEM = "system"       # 系统提示消息


class User(Base):
    """
    用户模型

    存储用户账户信息，支持用户认证和权限管理。

    关联：
    - documents: 用户上传的文档
    - conversations: 用户的对话会话
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    """
    文档模型

    存储用户上传的文档元数据，追踪处理状态。

    关联：
    - user: 文档所属用户
    - chunks: 文档切分后的片段
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(BigInteger)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ============ 新增：冲突检测状态 ============
    conflict_check_status = Column(String(20), default="completed", nullable=False)
    conflict_check_started_at = Column(DateTime(timezone=True), nullable=True)
    conflict_check_completed_at = Column(DateTime(timezone=True), nullable=True)
    conflict_check_progress = Column(String(20), nullable=True)

    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """
    文档片段模型

    存储文档切分后的片段，支持三层分块结构。
    每个片段在 Milvus 中有对应的向量表示。

    三层结构：
    - LARGE 片段是顶层，包含完整的语义单元
    - MEDIUM 片段是 LARGE 的子片段
    - SMALL 片段是 MEDIUM 的子片段

    关联：
    - document: 所属文档
    - parent_chunk: 父片段（用于层级结构）
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_type = Column(SQLEnum(ChunkType), nullable=False)
    parent_chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    content = Column(Text, nullable=False)
    metadata_ = Column(JSON, name="metadata")
    position = Column(Integer)
    token_count = Column(Integer)
    milvus_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ============ 新增：chunk 生命周期字段 ============
    # 增量更新 & 状态镜像
    content_hash = Column(String(64), nullable=True, index=True)
    status = Column(String(20), default="active", nullable=False, index=True)

    # ============ 新增：char_start/char_end 偏移（Task 4） ============
    # 切片在原文中的字符 range，供 KG Article<->chunk overlap 匹配使用
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)

    # 冲突作废元数据
    conflict_with_chunk_id = Column(String(100), nullable=True)
    conflict_detected_at = Column(DateTime(timezone=True), nullable=True)
    confidence = Column(Float, nullable=True)
    review_reason = Column(Text, nullable=True)
    superseded_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    # 命中统计（高频写）
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    hit_count = Column(Integer, default=0, nullable=False)
    total_score = Column(Float, default=0.0, nullable=False)
    avg_score = Column(Float, default=0.0, nullable=False)

    # 冷知识归档
    archived_reason = Column(String(30), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="chunks")
    parent_chunk = relationship("DocumentChunk", remote_side=[id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])


class Conversation(Base):
    """
    对话会话模型

    管理用户的对话会话，一个会话包含多条消息。

    关联：
    - user: 会话所属用户
    - messages: 会话中的所有消息
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


class ChatMessage(Base):
    """
    对话消息模型

    存储对话中的单条消息，包括用户问题和 AI 回复。
    回复消息会保存检索到的片段和调试信息用于追溯和分析。

    字段说明：
    - content: 消息内容
    - retrieved_chunks: 检索到的文档片段（JSON 格式，包含 content、score、chunk_type）
    - debug_info: RAG 检索调试信息（JSON 格式，包含各步骤详情）
    - process_time: 处理耗时（毫秒）

    关联：
    - conversation: 所属对话会话
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, nullable=True)
    debug_info = Column(JSON, nullable=True)  # RAG 检索调试信息
    process_time = Column(BigInteger, nullable=True)  # 处理耗时（毫秒）
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


class MessageFeedback(Base):
    """用户对 assistant 消息的点赞/点踩。

    同一 user 对同一 message 只保留最后一次反馈（upsert by (message_id, user_id)）。
    review 时 JOIN chat_messages 取 debug_info 做交叉验证。
    """
    __tablename__ = "message_feedback"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    polarity = Column(String(10), nullable=False)  # 'up' or 'down'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_message_user"),
    )


class ConversationSummary(Base):
    """会话滚动摘要。"""
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), unique=True, nullable=False, index=True)
    summary = Column(Text, nullable=False, default="")
    last_summarized_message_id = Column(Integer, nullable=True)
    message_count = Column(Integer, default=0)
    summary_token_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    conversation = relationship("Conversation")


class UserMemory(Base):
    """用户长期记忆。"""
    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    memory_type = Column(SQLEnum(MemoryType), nullable=False)
    content = Column(Text, nullable=False)
    source_message_ids = Column(JSON, nullable=True)
    importance = Column(Float, default=0.5)
    status = Column(String(20), default="active", index=True)
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User")
    conversation = relationship("Conversation")
