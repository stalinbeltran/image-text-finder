"""X — background jobs: a FIFO queue with a hard worker limit.

**This is the queue fase 7 promised, and it is the seed of `jobq`** (librerias.md
§1). Three things it now does that the fase-2 stub did not, and each one is a trap
rather than a feature:

  - **Worker limit** (=1 on CPU). Was there from the first line, and stays: torch
    already saturates every core inside one run, so N trainings at once each go
    ~N times slower and N `PatchDataset`s sit in RAM. A 20-point sweep with no
    limit ran out of memory long before it finished.
  - **Cooperative cancellation.** `Future.cancel()` only cancels what has not
    started; a training that is already running has to be *asked* to stop. So a
    job may carry a `cancel` callback -- for a run it marks `stop.json`, for a
    sweep it marks the sweep's stop -- and the work function cuts at its own safe
    point (end of epoch / end of trial). Nothing is killed mid-batch.
  - **Persistence.** The old state lived in memory with daemon threads, so a
    restart forgot the history -- and a CPU sweep runs for hours. With a
    `persist_dir` every job record is written on each transition and reloaded on
    start; a job that was live when the process died reloads as `interrupted`,
    because its thread is gone and it is not coming back on its own. (What DOES
    come back is the sweep behind it: its state is on disk, so the API re-enqueues
    it -- see `itf.sweeps.resume`.)

What it deliberately does NOT do: resurrect the *work* of an interrupted job. The
thread is gone. The durable things -- runs on disk, the sweep's optuna study --
are what resume; a bare `build-patch-dataset` that was interrupted is simply
re-run by the user.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

#: `interrupted` is not `error`: the job did not fail, the process it ran in
#: stopped under it. Telling them apart matters -- an interrupted sweep is
#: resumable, a failed one is not.
JobState = Literal["queued", "running", "done", "error", "cancelled", "interrupted"]

#: A job in one of these is finished: cancelling it does nothing, and it is never
#: reloaded as `interrupted`.
_TERMINAL: frozenset[str] = frozenset({"done", "error", "cancelled", "interrupted"})


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

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            id=d["id"],
            kind=d["kind"],
            state=d.get("state", "queued"),
            detail=d.get("detail") or {},
            result=d.get("result"),
            error=d.get("error"),
            created_at=d.get("created_at", ""),
            finished_at=d.get("finished_at", ""),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobQueue:
    """A FIFO queue with a hard worker limit, cooperative cancel and persistence.

    `max_workers=1` is not a placeholder: on CPU, concurrency does not help.
    torch already saturates every core inside a single run, so N at once each go
    ~N times slower and N datasets sit in RAM at the same time.

    `persist_dir` is optional: without it the queue is purely in-memory (tests
    that do not care about restart pass nothing). With it, every transition is
    written, and on construction any record left `queued`/`running` reloads as
    `interrupted`.
    """

    def __init__(self, max_workers: int = 1, persist_dir: Path | None = None):
        if max_workers < 1:
            raise ValueError("max_workers debe ser >= 1")
        self.max_workers = max_workers
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._fns: dict[str, Callable[[], Any]] = {}
        #: The cooperative-stop callback per job, if it has one. Called by
        #: `cancel()`; the work function decides where it is safe to act on it.
        self._cancels: dict[str, Callable[[], None]] = {}
        #: Jobs someone asked to stop. A running job that returns after this was
        #: set closes as `cancelled`, not `done`.
        self._cancel_requested: set[str] = set()
        self._lock = threading.Lock()
        self._persist_dir = persist_dir
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        self._workers = [
            threading.Thread(target=self._work, name=f"itf-worker-{i}", daemon=True)
            for i in range(max_workers)
        ]
        for w in self._workers:
            w.start()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _record_path(self, job_id: str) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / f"{job_id}.json"

    def _persist(self, job: Job) -> None:
        """Write one job's record. Called under `self._lock`.

        Best-effort: a job whose record cannot be written must not take the queue
        down with it -- the in-memory copy is still authoritative for this run.
        """
        if self._persist_dir is None:
            return
        try:
            self._record_path(job.id).write_text(json.dumps(job.as_dict()), encoding="utf-8")
        except OSError:
            pass

    def _load(self) -> None:
        """Reload persisted job records, marking live ones `interrupted`.

        A record still `queued`/`running` means the process died with it in
        flight: the thread that was running it is gone, so it cannot report its
        own end. Reloading it as `interrupted` (not `error`) is what lets the
        resumer tell "the box was rebooted" from "the work failed".
        """
        assert self._persist_dir is not None
        for path in sorted(self._persist_dir.glob("*.json")):
            try:
                job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
            if job.state not in _TERMINAL:
                job.state = "interrupted"
                if not job.finished_at:
                    job.finished_at = _now()
            self._jobs[job.id] = job
            self._persist(job)

    # ── Submitting ─────────────────────────────────────────────────────────────

    def submit(
        self,
        kind: str,
        fn: Callable[[], Any],
        detail: dict | None = None,
        cancel: Callable[[], None] | None = None,
    ) -> Job:
        """Queue `fn` to run on a worker. `cancel` is its cooperative-stop signal.

        `cancel` is called by `JobQueue.cancel(job_id)`; it does not kill anything,
        it asks -- `RunStore.request_stop` for a training, the sweep's stop for a
        sweep. The work function is responsible for checking and cutting cleanly.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, detail=detail or {}, created_at=_now())
        with self._lock:
            self._jobs[job.id] = job
            self._fns[job.id] = fn
            if cancel is not None:
                self._cancels[job.id] = cancel
            self._persist(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ── Cancelling ─────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop. Returns the job, or None if there is no such id.

        - **Queued** (not started): marked `cancelled` now; the worker skips it
          when it reaches the front.
        - **Running**: its `cancel` callback is signalled (cooperative); it closes
          as `cancelled` when the work function returns at its next safe point.
        - **Terminal**: nothing to do -- the caller turns that into a 409.

        The callback is called OUTSIDE the lock: it may touch disk (write
        `stop.json`), and holding the queue lock across that would stall every
        other job.
        """
        cancel_cb: Callable[[], None] | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state in _TERMINAL:
                return job
            self._cancel_requested.add(job_id)
            cancel_cb = self._cancels.get(job_id)
            if job.state == "queued":
                # Not started yet: the worker has not popped it, so mark it
                # cancelled here and let the worker drop it on sight.
                job.state = "cancelled"
                job.finished_at = _now()
                self._fns.pop(job_id, None)
                self._persist(job)
        if cancel_cb is not None:
            cancel_cb()
        return job

    # ── The worker loop ──────────────────────────────────────────────────────────

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    job = self._jobs[job_id]
                    if job.state == "cancelled":
                        # Cancelled while it sat in the queue. Nothing ran.
                        continue
                    fn = self._fns.pop(job_id)
                    job.state = "running"
                    self._persist(job)
                try:
                    result = fn()
                except Exception as exc:  # noqa: BLE001 -- a job must not kill its worker
                    with self._lock:
                        job.state = "error"
                        # The message alone loses where it broke, and a job that
                        # fails in a worker thread has no other way to tell you.
                        job.error = f"{type(exc).__name__}: {exc}"
                        job.detail = {**job.detail, "traceback": traceback.format_exc()}
                        job.finished_at = _now()
                        self._persist(job)
                else:
                    with self._lock:
                        # `cancelled`, not `done`, if a stop was asked and the work
                        # function honoured it: the job did not run to completion.
                        job.state = "cancelled" if job_id in self._cancel_requested else "done"
                        job.result = result
                        job.finished_at = _now()
                        self._persist(job)
                finally:
                    with self._lock:
                        self._cancels.pop(job_id, None)
                        self._cancel_requested.discard(job_id)
            finally:
                self._queue.task_done()
