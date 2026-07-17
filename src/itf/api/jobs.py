"""X — background jobs.

**Deliberately small, and deliberately not thread-per-job.** The real queue is
fase 7 (persistence, cooperative cancellation, resume). This is the minimum that
`POST /patch-datasets` needs (R3: anything that can take more than ~1s returns a
job, not a result) -- but it is built with the worker limit from the first line,
because that is the part that is a trap rather than a feature.

The trap, measured on the old code: `jobs.py` called `threading.Thread` per
`submit()`, with no limit. A 20-point sweep meant 20 trainings fighting over the
same cores -- torch already uses all of them, so each ran ~20x slower -- each one
holding its whole `PatchDataset` resident in RAM. It ran out of memory long
before it finished. **On CPU the worker limit is 1**, and adding it later would
mean touching the queue while jobs are running.

What is honestly missing, and is fase 7's: the state lives in memory, so a
restart forgets the history, and there is no cancel.
"""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

JobState = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = "queued"
    detail: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    created_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "detail": self.detail,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobQueue:
    """A FIFO queue with a hard worker limit.

    `max_workers=1` is not a placeholder: on CPU, concurrency does not help.
    torch already saturates every core inside a single run, so N at once each go
    ~N times slower and N datasets sit in RAM at the same time.
    """

    def __init__(self, max_workers: int = 1):
        if max_workers < 1:
            raise ValueError("max_workers debe ser >= 1")
        self.max_workers = max_workers
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._fns: dict[str, Callable[[], Any]] = {}
        self._lock = threading.Lock()
        self._workers = [
            threading.Thread(target=self._work, name=f"itf-worker-{i}", daemon=True)
            for i in range(max_workers)
        ]
        for w in self._workers:
            w.start()

    def submit(self, kind: str, fn: Callable[[], Any], detail: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, detail=detail or {}, created_at=_now())
        with self._lock:
            self._jobs[job.id] = job
            self._fns[job.id] = fn
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs[job_id]
                fn = self._fns.pop(job_id)
                job.state = "running"
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 -- a job must not kill its worker
                with self._lock:
                    job.state = "error"
                    # The message alone loses where it broke, and a job that fails
                    # in a worker thread has no other way to tell you.
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.detail = {**job.detail, "traceback": traceback.format_exc()}
                    job.finished_at = _now()
            else:
                with self._lock:
                    job.state = "done"
                    job.result = result
                    job.finished_at = _now()
            finally:
                self._queue.task_done()
