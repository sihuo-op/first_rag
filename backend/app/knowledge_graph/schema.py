"""KG 数据模型：节点/边类型枚举、Pydantic 模型、Cypher 标签常量。"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    LAW = "Law"
    ARTICLE = "Article"
    CONCEPT = "Concept"
    PARTY = "Party"
    REGION = "Region"
    DOCUMENT = "Document"


class EdgeType(str, Enum):
    CITES = "CITES"
    IS_A = "IS_A"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    APPLIES_TO = "APPLIES_TO"
    CONTAINS = "CONTAINS"
    EXPLAINS = "EXPLAINS"


class ConflictStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class ArticleStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


CYPHER_LABELS = {nt.value: nt.value for nt in NodeType}


class LawNode(BaseModel):
    id: str
    name: str
    level: str  # 法律/法规/规章/司法解释
    effective_date: Optional[str] = None
    issuer: Optional[str] = None
    region_id: Optional[str] = None


class ArticleNode(BaseModel):
    id: str
    law_id: str
    article_no: int
    content_hash: str
    content: str = ""  # 条款原文（text[char_start:char_end]），LLM 冲突判定用
    chunk_ids: list[str] = Field(default_factory=list)
    status: str = "active"
    char_start: int
    char_end: int


class ConceptNode(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float]  # 1024 维
    source_chunk_ids: list[str] = Field(default_factory=list)


class PartyNode(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class RegionNode(BaseModel):
    id: str
    name: str
    level: str  # 国家/省/市


class DocumentNode(BaseModel):
    id: str
    source_file: str
    uploaded_at: str
    doc_type: str
