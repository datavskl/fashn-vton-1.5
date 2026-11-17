"""In-process queue and status for one GPU and one Uvicorn worker."""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import JOB_TTL_SECONDS, QUEUE_SIZE
from app.services.inference import _infer

LOG = logging.getLogger("shrivastra.vton")
_jobs: dict[str, "Job"] = {}
_job_queue: asyncio.Queue | None = None
_worker_task: asyncio.Task | None = None

@dataclass
class Job:
    id: str
    created_at: float
    status: str = "queued"
    result: str | None = None
    error: str | None = None
    completed_at: float | None = None
    inputs: tuple = field(default_factory=tuple, repr=False)


def _purge_jobs():
    now = time.time()
    for job_id, job in list(_jobs.items()):
        if job.completed_at is not None and now - job.completed_at > JOB_TTL_SECONDS:
            del _jobs[job_id]


async def _job_worker():
    assert _job_queue is not None
    while True:
        job_id = await _job_queue.get()
        try:
            job = _jobs.get(job_id)
            if job:
                job.status = "processing"
                try:
                    job.result = await _infer(job.inputs)
                    job.status = "completed"
                except Exception:
                    LOG.exception("Inference failed for job %s", job.id)
                    job.error = "Inference failed; check server logs"
                    job.status = "failed"
                finally:
                    job.inputs = ()
                    job.completed_at = time.time()
        finally:
            _job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_queue, _worker_task
    _job_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    _worker_task = asyncio.create_task(_job_worker())
    try:
        yield
    finally:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        _job_queue = None


