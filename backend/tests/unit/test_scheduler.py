"""验证 APScheduler 注册了 cold_knowledge_sweep 和 hard_delete_sweep 两个定时任务，
且使用 settings 中的 cron 表达式。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.core.scheduler import setup_scheduler, shutdown_scheduler, _scheduler as _sched_singleton


def _reset_singleton():
    """清除 scheduler 单例，便于多次 setup。"""
    import app.core.scheduler as mod
    mod._scheduler = None


def test_setup_scheduler_registers_both_jobs_with_correct_cron():
    """setup_scheduler 后应注册两个 job，cron 与 settings 一致。"""
    _reset_singleton()
    try:
        sched = setup_scheduler()
        job_ids = {j.id for j in sched.get_jobs()}
        assert job_ids == {"cold_knowledge_sweep", "hard_delete_sweep"}

        settings = get_settings()
        cold_job = sched.get_job("cold_knowledge_sweep")
        hard_job = sched.get_job("hard_delete_sweep")

        # CronTrigger.from_crontab 解析 5 字段 cron
        expected_cold = CronTrigger.from_crontab(settings.COLD_KNOWLEDGE_SWEEP_CRON)
        expected_hard = CronTrigger.from_crontab(settings.HARD_DELETE_SWEEP_CRON)

        # 比较关键字段（直接对比 trigger 不一定相等，比较字段更稳定）
        assert _cron_fields(cold_job.trigger) == _cron_fields(expected_cold)
        assert _cron_fields(hard_job.trigger) == _cron_fields(expected_hard)
    finally:
        shutdown_scheduler()


def test_setup_scheduler_idempotent():
    """重复调用 setup_scheduler 不应创建多个调度器或多个同名 job。"""
    _reset_singleton()
    try:
        s1 = setup_scheduler()
        s2 = setup_scheduler()
        assert s1 is s2  # 同一单例
        cold_jobs = [j for j in s1.get_jobs() if j.id == "cold_knowledge_sweep"]
        assert len(cold_jobs) == 1  # replace_existing=True
    finally:
        shutdown_scheduler()


def test_shutdown_scheduler_clears_singleton():
    """shutdown_scheduler 后单例应被清空。"""
    _reset_singleton()
    setup_scheduler()
    shutdown_scheduler()
    import app.core.scheduler as mod
    assert mod._scheduler is None


def _cron_fields(trigger):
    """提取 CronTrigger 的字段值用于比较。"""
    return {
        "minute": str(trigger.fields[0]),
        "hour": str(trigger.fields[1]),
        "day": str(trigger.fields[2]),
        "month": str(trigger.fields[3]),
        "day_of_week": str(trigger.fields[4]),
    }
