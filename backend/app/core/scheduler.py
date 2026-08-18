"""
APScheduler 调度器

注册两个定时任务：
- cold_knowledge_sweep: 每天 3 点扫描归档冷知识
- hard_delete_sweep: 每天 4 点硬删除过期归档

使用 SQLAlchemyJobStore 持久化到 PG/SQLite，重启不丢。
多副本部署时通过 PG 行锁保证单实例执行（TODO: 后续如需多副本再加）。
"""
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.core.observability import get_tracer

tracer = get_tracer("scheduler")

_scheduler: BackgroundScheduler = None


def _run_cold_knowledge_sweep():
    """定时任务：冷知识扫描"""
    from app.core.dependencies import get_vector_store
    from app.db.session import SessionLocal
    from app.services.cold_knowledge_service import ColdKnowledgeService
    db = SessionLocal()
    try:
        with tracer.start_as_current_span("scheduler.cold_knowledge_sweep"):
            svc = ColdKnowledgeService(db, get_vector_store())
            stats = svc.sweep()
            print(f"[Scheduler] cold_knowledge_sweep: {stats}")
    except Exception as e:
        print(f"[Scheduler] cold_knowledge_sweep failed: {e}")
    finally:
        db.close()


def _run_hard_delete_sweep():
    """定时任务：硬删除过期归档"""
    from app.core.dependencies import get_vector_store
    from app.db.session import SessionLocal
    from app.services.cold_knowledge_service import ColdKnowledgeService
    db = SessionLocal()
    try:
        with tracer.start_as_current_span("scheduler.hard_delete_sweep"):
            svc = ColdKnowledgeService(db, get_vector_store())
            count = svc.hard_delete_sweep()
            print(f"[Scheduler] hard_delete_sweep: deleted {count} chunks")
    except Exception as e:
        print(f"[Scheduler] hard_delete_sweep failed: {e}")
    finally:
        db.close()


def setup_scheduler():
    """初始化并启动调度器（应用启动时调用一次）"""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    settings = get_settings()
    # 复用 settings.DATABASE_URL（与 app.db.session.engine 一致，避免硬编码）
    jobstore_url = settings.DATABASE_URL

    _scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=jobstore_url)},
        timezone="Asia/Shanghai"
    )

    # 冷知识扫描
    trigger = CronTrigger.from_crontab(settings.COLD_KNOWLEDGE_SWEEP_CRON)
    _scheduler.add_job(
        _run_cold_knowledge_sweep,
        trigger=trigger,
        id="cold_knowledge_sweep",
        replace_existing=True
    )

    # 硬删除扫描
    trigger2 = CronTrigger.from_crontab(settings.HARD_DELETE_SWEEP_CRON)
    _scheduler.add_job(
        _run_hard_delete_sweep,
        trigger=trigger2,
        id="hard_delete_sweep",
        replace_existing=True
    )

    _scheduler.start()
    print(f"[Scheduler] started: cold_sweep={settings.COLD_KNOWLEDGE_SWEEP_CRON}, hard_delete={settings.HARD_DELETE_SWEEP_CRON}")
    return _scheduler


def shutdown_scheduler():
    """关闭调度器（应用关闭时调用）"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
