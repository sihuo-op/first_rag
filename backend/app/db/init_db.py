from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.core.security import get_password_hash
from app.db.session import SessionLocal, engine
from app.entities.database import Base, User, UserRole

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


def _migrate_add_columns(engine=None):
    """数据库迁移：添加新列

    Args:
        engine: 可选的 SQLAlchemy engine；不传时使用 app.db.session.engine（生产 engine）。
                传入便于单元测试用独立 in-memory SQLite 验证迁移路径。
    """
    if engine is None:
        from app.db.session import engine as _engine
        engine = _engine

    inspector = inspect(engine)

    # document_chunks 表迁移：添加 char_start / char_end 列（Task 4）
    if "document_chunks" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("document_chunks")]
        with engine.connect() as conn:
            if "char_start" not in columns:
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN char_start INTEGER"))
                conn.commit()
                print("Added column: document_chunks.char_start")
            if "char_end" not in columns:
                conn.execute(text("ALTER TABLE document_chunks ADD COLUMN char_end INTEGER"))
                conn.commit()
                print("Added column: document_chunks.char_end")

    # chat_messages 表迁移：debug_info / process_time 列（旧迁移，保留）
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
