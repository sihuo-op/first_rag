from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_conflict_service, get_retriever, get_vector_store
from app.core.security import get_current_active_user
from app.db.session import get_db
from app.entities.database import User
from app.entities.schemas import DocumentResponse, DocumentUpdate
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import MilvusStore
from app.services.conflict_service import ConflictService
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_document_service(
    db: Session = Depends(get_db),
    vector_store: MilvusStore = Depends(get_vector_store),
    retriever: HybridRetriever = Depends(get_retriever),
    conflict_service: ConflictService = Depends(get_conflict_service)
) -> DocumentService:
    return DocumentService(db, vector_store, retriever, conflict_service=conflict_service)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    return doc_service.get_documents(user_id=current_user.id, skip=skip, limit=limit)


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: int,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    document = doc_service.get_document_by_id(doc_id, user_id=current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    document = await doc_service.upload_document(file, current_user.id, title, background_tasks=background_tasks)
    return document


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    success = doc_service.delete_document(doc_id, user_id=current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return {"message": "Document deleted successfully"}


@router.put("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: int,
    data: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    document = doc_service.update_document(doc_id, data, user_id=current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document


@router.get("/{doc_id}/status")
async def get_document_status(
    doc_id: int,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service)
):
    document = doc_service.get_document_by_id(doc_id, user_id=current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return {
        "status": document.status,
        "error_message": document.error_message
    }
