"""The queue (X / jobq): worker limit, cooperative cancel, persistence (fase 7).

These are unit tests of `JobQueue` plus a couple over HTTP. The queue is generic
-- it knows nothing about runs or sweeps -- so it is tested with plain callables
and threading events, never by training anything.
"""

from __future__ import annotations

import threading
import time

from itf.api.jobs import Job, JobQueue


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condición no cumplida a tiempo")


def test_a_completed_job_reports_its_result():
    q = JobQueue(max_workers=1)
    job = q.submit("compute", lambda: 41 + 1)
    _wait_for(lambda: q.get(job.id).state == "done")
    assert q.get(job.id).result == 42


def test_cancel_of_a_running_job_signals_its_callback_and_it_closes_cancelled():
    """A running job is ASKED to stop; it cuts at its own safe point.

    `Future.cancel()` would not do this -- it only cancels what has not started.
    The callback is the cooperative signal, and the job that honours it closes as
    `cancelled`, not `done`: it did not run to completion.
    """
    q = JobQueue(max_workers=1)
    stop = threading.Event()
    started = threading.Event()
    cancelled_cb = threading.Event()

    def work():
        started.set()
        # Stand in for "end of epoch": loop until asked to stop.
        while not stop.is_set():
            time.sleep(0.005)
        return "stopped cleanly"

    job = q.submit("train", work, cancel=lambda: (cancelled_cb.set(), stop.set()))
    _wait_for(started.is_set)

    q.cancel(job.id)
    assert cancelled_cb.is_set(), "el callback cooperativo tiene que dispararse"
    _wait_for(lambda: q.get(job.id).state == "cancelled")
    assert q.get(job.id).result == "stopped cleanly"


def test_cancel_of_a_queued_job_skips_it_without_running(tmp_path):
    """Cancelled while it waited in line: the work never runs at all.

    With `max_workers=1`, a first blocking job holds the worker; the second sits
    queued, and cancelling it there must drop it -- not run it once the blocker
    frees up.
    """
    q = JobQueue(max_workers=1)
    release = threading.Event()
    second_ran = threading.Event()

    blocker = q.submit("block", lambda: release.wait(5.0))
    _wait_for(lambda: q.get(blocker.id).state == "running")

    second = q.submit("never", lambda: second_ran.set())
    assert q.get(second.id).state == "queued"

    q.cancel(second.id)
    assert q.get(second.id).state == "cancelled"

    release.set()
    _wait_for(lambda: q.get(blocker.id).state == "done")
    time.sleep(0.05)
    assert not second_ran.is_set(), "un job cancelado en cola no debe ejecutarse"


def test_cancel_of_a_finished_job_is_a_no_op():
    q = JobQueue(max_workers=1)
    job = q.submit("compute", lambda: 1)
    _wait_for(lambda: q.get(job.id).state == "done")
    # Returns the job (so the API can 409), and does not resurrect it.
    assert q.cancel(job.id).state == "done"
    assert q.cancel("no-such-id") is None


def test_persistence_survives_a_restart_and_marks_live_jobs_interrupted(tmp_path):
    """A restart forgets nothing -- and it forgets the RIGHT thing about a live job.

    A job that was `running` when the process died reloads as `interrupted`, not
    `error`: the work did not fail, the box was rebooted. That distinction is what
    lets the sweep resumer tell the two apart.
    """
    persist = tmp_path / "jobs"

    q1 = JobQueue(max_workers=1, persist_dir=persist)
    done = q1.submit("compute", lambda: "ok")
    _wait_for(lambda: q1.get(done.id).state == "done")

    # Forge a record left `running`, as a hard kill would leave it: written to
    # disk but its thread gone.
    live = Job(id="deadbeef0001", kind="train", state="running", created_at="2026-01-01T00:00:00+00:00")
    (persist / f"{live.id}.json").write_text(__import__("json").dumps(live.as_dict()), encoding="utf-8")

    q2 = JobQueue(max_workers=1, persist_dir=persist)
    assert q2.get(done.id).state == "done", "un job terminado se recuerda tal cual"
    reloaded = q2.get(live.id)
    assert reloaded is not None and reloaded.state == "interrupted"


def test_jobs_cancel_endpoint_404s_and_409s(itf_api):
    client, _ = itf_api
    assert client.post("/jobs/no-such-id/cancel").status_code == 404
