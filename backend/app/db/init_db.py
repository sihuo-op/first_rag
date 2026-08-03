import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.entities.database import Base, User, UserRole
from app.core.security import get_password_hash
from app.core.config import get_settings
from app.db.session import engine, SessionLocal

settings = get_settings()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # 数据库迁移：添加新列（如果不存在）
    _migrate_add_columns()

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == UserRole.ADMIN).first()
        if not admin:
            admin_user = User(
                username=settings.FIRST_ADMIN_USERNAME,
                email=settings.FIRST_ADMIN_EMAIL,
                hashed_password=get_password_hash(settings.FIRST_ADMIN_PASSWORD),
                role=UserRole.ADMIN,
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print(f"Created admin user: {settings.FIRST_ADMIN_USERNAME}")
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()


def _migrate_add_columns():
    """数据库迁移：添加新列"""
    with engine.connect() as conn:
        # 检查并添加 debug_info 列
        try:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN debug_info JSON"))
            conn.commit()
            print("Added column: debug_info")
        except Exception:
            pass  # 列已存在

        # 检查并添加 process_time 列
        try:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN process_time BIGINT"))
            conn.commit()
            print("Added column: process_time")
        except Exception:
            pass  # 列已存在
