from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token, create_refresh_token, decode_token
from app.entities.database import User
from app.entities.schemas import Token
from app.services.user_service import UserService

settings = get_settings()


class AuthService:
    """
    认证服务

    负责用户认证和 JWT 令牌管理。

    功能：
    - 用户登录验证，生成访问令牌和刷新令牌
    - 令牌刷新，延长用户会话
    - 从令牌解析用户身份

    令牌类型：
    - access_token: 访问令牌，有效期短（默认30分钟）
    - refresh_token: 刷新令牌，有效期长（默认7天）
    """

    def __init__(self, db: Session):
        self.db = db
        self.user_service = UserService(db)

    def login(self, username: str, password: str) -> Optional[Token]:
        user = self.user_service.authenticate_user(username, password)
        if not user:
            return None

        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )

        refresh_token_expires = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = create_refresh_token(
            data={"sub": user.username},
            expires_delta=refresh_token_expires
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            refresh_token=refresh_token
        )

    def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        payload = decode_token(refresh_token)
        if not payload:
            return None
        if payload.get("type") != "refresh":
            return None

        username: str = payload.get("sub")
        if not username:
            return None

        user = self.user_service.get_user_by_username(username)
        if not user or not user.is_active:
            return None

        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )

        return Token(
            access_token=new_access_token,
            token_type="bearer"
        )

    def get_current_user_from_token(self, token: str) -> Optional[User]:
        payload = decode_token(token)
        if not payload:
            return None
        if payload.get("type") != "access":
            return None

        username: str = payload.get("sub")
        if not username:
            return None

        return self.user_service.get_user_by_username(username)
