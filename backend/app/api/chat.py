from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.entities.schemas import (
    ConversationResponse, ConversationCreate, ConversationUpdate,
    ChatMessageResponse, ChatRequest, ChatResponse, ChatMode,
    FeedbackCreate, FeedbackResponse
)
from app.entities.database import User, MessageFeedback
from app.core.security import get_current_active_user
from app.services.chat_service import ChatService
from app.core.dependencies import get_retriever
from app.rag.retriever import HybridRetriever

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_chat_service(
    db: Session = Depends(get_db),
    retriever: HybridRetriever = Depends(get_retriever)
) -> ChatService:
    return ChatService(db, retriever)


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    return chat_service.get_conversations(user_id=current_user.id, skip=skip, limit=limit)


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    return chat_service.create_conversation(current_user.id, data)


@router.get("/conversations/{conv_id}", response_model=ConversationResponse)
async def get_conversation(
    conv_id: int,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    conversation = chat_service.get_conversation_by_id(conv_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


@router.put("/conversations/{conv_id}", response_model=ConversationResponse)
async def update_conversation(
    conv_id: int,
    data: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    conversation = chat_service.update_conversation(conv_id, data, user_id=current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: int,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    success = chat_service.delete_conversation(conv_id, user_id=current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return {"message": "Conversation deleted successfully"}


@router.get("/conversations/{conv_id}/messages", response_model=List[ChatMessageResponse])
async def get_messages(
    conv_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    return chat_service.get_messages(conv_id, user_id=current_user.id, skip=skip, limit=limit)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service)
):
    """对话接口 - 支持传统 RAG、Agentic RAG 和流式输出"""
    if request.stream:
        return StreamingResponse(
            chat_service.agentic_chat_stream(
                user_id=current_user.id,
                query=request.query,
                conv_id=request.conversation_id,
                background_tasks=background_tasks
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            },
            background=background_tasks
        )

    try:
        result = await chat_service.agentic_chat(
            user_id=current_user.id,
            query=request.query,
            conv_id=request.conversation_id,
            background_tasks=background_tasks
        )
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    data: FeedbackCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """点赞/点踩接口 - 同一 user 对同一 message 的反馈会被覆盖（upsert）。"""
    existing = db.query(MessageFeedback).filter_by(
        message_id=data.message_id, user_id=current_user.id
    ).first()
    if existing:
        existing.polarity = data.polarity
        db.commit()
        db.refresh(existing)
        return existing
    fb = MessageFeedback(
        message_id=data.message_id,
        user_id=current_user.id,
        polarity=data.polarity
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb
