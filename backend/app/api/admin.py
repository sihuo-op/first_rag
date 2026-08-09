from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.entities.schemas import (
    UserResponse,
    DocumentResponse,
    DocumentWithConflictStatusResponse,
    UserUpdate,
    StatsResponse,
    ChunkDetailResponse,
)
from app.entities.database import User, Document, DocumentChunk, Conversation, ChatMessage, UserRole
from app.core.security import get_current_admin_user
from app.services.user_service import UserService
from app.services.document_service import DocumentService
from app.services.conflict_service import ConflictService
from app.core.dependencies import get_vector_store, get_conflict_service
from app.rag.vector_store import MilvusStore

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    return user_service.get_users(skip=skip, limit=limit)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    user = user_service.update_user(user_id, data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )
    success = user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"message": "User deleted successfully"}


@router.get("/documents", response_model=List[DocumentWithConflictStatusResponse])
async def list_all_documents(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store),
    conflict_service: ConflictService = Depends(get_conflict_service)
):
    doc_service = DocumentService(db, vector_store, conflict_service=conflict_service)
    return doc_service.get_documents(skip=skip, limit=limit)


@router.get("/chunks", response_model=List[ChunkDetailResponse])
async def list_chunks(
    status: Optional[str] = None,
    archived_reason: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """列出 chunks，可按 status / archived_reason 过滤"""
    query = db.query(DocumentChunk)
    if status:
        query = query.filter(DocumentChunk.status == status)
    if archived_reason:
        query = query.filter(DocumentChunk.archived_reason == archived_reason)
    chunks = query.order_by(DocumentChunk.id.desc()).offset(skip).limit(limit).all()

    # join conflict_with_chunk 的 content（用于 pending_review / superseded 展示新 chunk 内容）
    result = []
    for c in chunks:
        item = ChunkDetailResponse.model_validate(c)
        if c.conflict_with_chunk_id:
            new_chunk = db.query(DocumentChunk).filter_by(milvus_id=c.conflict_with_chunk_id).first()
            if new_chunk:
                item.conflict_with_content = new_chunk.content
        result.append(item)
    return result


@router.delete("/documents/{doc_id}")
async def admin_delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store),
    conflict_service: ConflictService = Depends(get_conflict_service)
):
    doc_service = DocumentService(db, vector_store, conflict_service=conflict_service)
    success = doc_service.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return {"message": "Document deleted successfully"}


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    total_users = db.query(func.count(User.id)).scalar()
    total_documents = db.query(func.count(Document.id)).scalar()
    total_conversations = db.query(func.count(Conversation.id)).scalar()
    total_messages = db.query(func.count(ChatMessage.id)).scalar()

    return StatsResponse(
        total_users=total_users,
        total_documents=total_documents,
        total_conversations=total_conversations,
        total_messages=total_messages
    )
