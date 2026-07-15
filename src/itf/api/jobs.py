"""Minimal in-process background-job manager for extract/train tasks.

Jobs run on daemon threads. Status is kept in memory and also mirrored to the
run directory (for training) so a fresh process / the web app can poll it.
Suitable for a single-worker dev/local deployment; swap for Celery/RQ if you
need multi-worker durability.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"          # pending | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[], Any], meta: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result = fn()
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - report any failure to the caller
                job.status = "error"
                job.error = f"{exc}\n{traceback.format_exc()}"
            finally:
                job.finished_at = time.time()

        threading.Thread(target=_run, name=f"job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)


JOBS = JobManager()
