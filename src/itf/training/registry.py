"""E — the run registry: what is on disk under `runs/`.

It lives here and not in `api/app.py` for the mechanical reason of api.md §0
(nothing here mentions HTTP), and not in `itf.patches` for a layering one: B must
not know that runs exist. The API composes the two.

**`create()` is the gate, and it is the only one.** Never overwriting a run is
not the API's rule to enforce: `itf-train` does not go through the API, so a
check that lived up there would guard one of the two doors. Both callers reserve
through `create()`, which refuses on a directory that already exists -- the old
code's `mkdir(exist_ok=True)` plus truncating `metrics.jsonl` destroyed results
without a word, and a sweep that auto-generates names is exactly who steps on it.

**The state is on disk, not in memory** (organizacion.md §3): `status.json` says
what a run is doing, and `stop.json` is how it is asked to quit. Both survive a
restart of the API, which the in-memory `JOBS` dict of the old code did not --
and a CPU sweep runs for hours. The real queue (persistence, resume) is fase 7;
this is the part of it that belongs to the RUN rather than to the job.

This module is the seed of `exp-registry` (librerias.md). It is not extracted
yet, and must not be: **the rule is to extract on the second use, not the
first.** It has exactly one user.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

RunState = Literal["queued", "running", "done", "error", "cancelled"]

#: A run in one of these is doing something: it must not be renamed or deleted
#: from under itself.
LIVE_STATES: frozenset[str] = frozenset({"queued", "running"})


class RunNotFound(KeyError):
    """No such run."""


class RunExists(FileExistsError):
    """A run by that name is already on disk.

    Its own type because the answer is a refusal, not a crash: the API turns it
    into a 409 and the CLI into a message. Silently continuing is what destroyed
    results in the old code.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: How long to keep trying when a reader and a writer collide (see below). A
#: DEADLINE, not a fixed number of attempts: the collisions clear in
#: microseconds, so what this really guards against is a *busy* reader, and there
#: the question is "has it had a gap yet?", not "how many times have I asked?".
#: Five seconds is invisible next to a 20 s epoch, and long enough that losing
#: the race for the whole window means something is actually wrong.
_RETRY_BUDGET_S = 5.0
_RETRY_STEP_S = 0.002


def _retrying(attempt: Callable[[], Any]) -> Any:
    """Run `attempt()` until it stops losing the Windows sharing race.

    Both sides of `write_json_atomic`/`read_text_retrying` need this, and for the
    same reason, so they share it rather than each growing their own loop.
    """
    deadline = time.monotonic() + _RETRY_BUDGET_S
    while True:
        try:
            return attempt()
        except PermissionError:
            if time.monotonic() >= deadline:
                # Out of budget: raise. A status that silently failed to land is
                # exactly the lie `status.json` exists to prevent, and five
                # seconds of collisions is not a race any more -- it is a file
                # someone else is holding open.
                raise
            time.sleep(_RETRY_STEP_S)


def write_json_atomic(path: Path, payload: dict, indent: int | None = None) -> None:
    """Write JSON so a reader never sees it half-written.

    **A run's files are read while they are being written**: the UI polls every
    2 s and a run rewrites `status.json` every epoch. `Path.write_text` truncates
    first and writes after, so there is a real window where the file is empty or
    cut in half -- and a reader landing in it gets a `JSONDecodeError` and
    concludes the run is CORRUPT, which is the one thing it is not.
    `GET /runs/{name}` answered **404 "no tiene un config.json legible"** about a
    perfectly healthy run, and a poll could show a running run as `error`. Found
    by a flaky test, not by reasoning: it failed about one run in three.

    So: temp file in the SAME directory, then `os.replace`, which is atomic
    everywhere -- a reader sees the old file or the new one, never a torn one.

    **And then the retry, which is not belt-and-braces: on Windows it is the
    whole point.** Windows refuses to replace a file another handle has open, and
    CPython opens for reading without `FILE_SHARE_DELETE`, so a poll landing on
    the wrong microsecond makes `os.replace` raise `PermissionError [WinError 5]`
    -- and that lands *inside the training thread*, killing a healthy run. It is
    not rare: measured here, a reader and a writer racing for 4 s produced **5111**
    failed replaces (and 1130 failed reads, which is why `read_json` retries too).
    Atomic-write-via-replace without retry is a POSIX habit that does not port.
    """
    # The thread id too: two threads writing one path would otherwise fight over
    # the same temp name and one would replace from a file the other deleted.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    try:
        _retrying(lambda: os.replace(tmp, path))
    except BaseException:
        # Do not leave the temp behind: `checkpoints()` and friends glob this
        # directory, and a litter of `.tmp` files is the kind of thing that gets
        # committed (D5 versions this directory).
        tmp.unlink(missing_ok=True)
        raise


def read_text_retrying(path: Path) -> str:
    """Read a file a live run may be writing underneath us.

    The other half of `write_json_atomic`: on Windows the reader loses the race
    just as often as the writer does (1130 failures in the same 4 s measurement),
    because `open()` fails while a replace is in flight. The file is fine; the
    timing is not. Retry briefly rather than report a healthy run as broken.
    """
    return _retrying(lambda: path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(read_text_retrying(path))


@dataclass(frozen=True)
class RunStore:
    """`runs/<name>/`, one subdirectory per E."""

    root: Path

    def path(self, name: str) -> Path:
        # `name` reaches this from an URL, so it must not be able to escape the
        # store: a name is one directory component, no separators, no `..`.
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de run inválido: {name!r}")
        return self.root / name

    def exists(self, name: str) -> bool:
        return self.path(name).is_dir()

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        # Every directory, not just those with a `config.json`: a queued run is a
        # run, and a run that died before writing its config still occupies its
        # name. Listing only the complete ones would report a name as free while
        # `create()` refuses it.
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # ── Creating ─────────────────────────────────────────────────────────────

    def create(self, name: str, config: dict) -> Path:
        """Reserve `runs/<name>/` and freeze its config. **Refuses to overwrite.**

        The reservation is what makes the run visible before the first epoch: it
        lands in `GET /runs` as `queued`, and `used_by` can already see which B
        it points at -- so deleting that B answers 409 instead of pulling the
        dataset out from under a run that is about to start.
        """
        path = self.path(name)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunExists(name) from exc
        write_json_atomic(path / "config.json", config, indent=2)
        self.set_status(name, "queued")
        return path

    # ── Reading ──────────────────────────────────────────────────────────────

    def config(self, name: str) -> dict:
        path = self.path(name) / "config.json"
        if not path.exists():
            raise RunNotFound(name)
        return read_json(path)

    def status(self, name: str) -> dict:
        """The EXPLICIT state (formatos.md §4.2).

        Deduced from which files exist -- as the old code did -- a crashed run
        reads as "running" forever, because the deduction has no way to tell "no
        more epochs are coming" from "the next one has not landed yet".
        """
        if not self.exists(name):
            raise RunNotFound(name)
        path = self.path(name) / "status.json"
        if not path.exists():
            return {"state": "error", "error": "el run no tiene status.json"}
        try:
            return read_json(path)
        except json.JSONDecodeError:
            return {"state": "error", "error": "status.json ilegible"}

    def set_status(self, name: str, state: RunState, **extra: Any) -> None:
        # Atomic: this is rewritten every epoch while the UI polls it, so a
        # non-atomic write makes a healthy run flick to `error` on screen.
        write_json_atomic(
            self.path(name) / "status.json", {"state": state, "updated_at": _now(), **extra}
        )

    def is_live(self, name: str) -> bool:
        return self.status(name).get("state") in LIVE_STATES

    @contextmanager
    def marking_failures(self, name: str) -> Iterator[None]:
        """Whatever goes wrong in here, the run says so instead of lying.

        It closes a real window. `train()` marks its own crashes, but only from
        the first epoch on -- everything before that (a `.npz` that will not
        load, a device that does not exist) raises while `status.json` still
        says `queued`, which is the crash-reads-as-running trap wearing a
        different word. Whoever reserved the run is who can close it, so both
        callers wrap the call in this.

        Re-raises: the caller still decides how to report. This only makes sure
        that what is on disk is true before it does.
        """
        try:
            yield
        except BaseException as exc:
            self.set_status(name, "error", error=f"{type(exc).__name__}: {exc}")
            raise

    def summary(self, name: str) -> dict | None:
        """The summary, or None if the run has not finished. None is a FACT here."""
        path = self.path(name) / "summary.json"
        if not path.exists():
            return None
        return read_json(path)

    def checkpoints(self, name: str) -> list[str]:
        return sorted(p.name for p in self.path(name).glob("*.pt"))

    def metrics(self, name: str, since: int = 0) -> dict:
        """Epoch records from `since` on -> `{records, next}` (R5 of api.md).

        **Incremental, never the whole history.** The old `GET /runs/{name}`
        returned every metric on every call while the UI polled in a loop, so the
        cost of watching a run grew with the epoch it was on.

        The last line may be half-written -- a live run is being read while it is
        written -- so only whole lines are parsed. Reading a torn line would
        raise mid-poll on a run that is perfectly healthy.
        """
        path = self.path(name) / "metrics.jsonl"
        if not path.exists():
            return {"records": [], "next": since}
        text = read_text_retrying(path)
        cut = text.rfind("\n")
        whole = text[: cut + 1] if cut >= 0 else ""
        records = [json.loads(line) for line in whole.splitlines() if line.strip()]
        return {"records": records[since:], "next": len(records)}

    def seconds_per_epoch(self, name: str) -> float | None:
        """Mean wall-clock seconds per epoch, or None if it has not run one.

        What Entrenar estimates the cost with. **None, not 0**: no epochs means
        "cannot tell", and a 0 would read as "instant" (formatos.md §2).
        """
        records = self.metrics(name)["records"]
        times = [r["seconds"] for r in records if r.get("seconds") is not None]
        return sum(times) / len(times) if times else None

    def using_patch_dataset(self, patch_dataset: str) -> list[str]:
        """Which runs were trained on this B.

        The question `DELETE /patch-datasets/{name}` must ask before destroying
        anything. Getting it wrong is not loud: removing a B in use leaves every
        run that pointed at it without provenance, silently -- the old code
        followed `run.config.data` -> `manifest.config.source` and just returned
        `None` when the directory had gone, so the Predict tab lost its dataset
        without saying why.

        Reads the provenance by NAME (D2). A run whose `config.json` has no
        provenance is not a legacy case to tolerate -- D3 killed that -- but this
        query still must not explode on a half-written run: a config being
        written right now would take the whole delete down with it, and "I could
        not read one run" is not an answer to "is this in use".
        """
        used_by = []
        for name in self.names():
            try:
                config = self.config(name)
            except (RunNotFound, json.JSONDecodeError, OSError):
                continue
            provenance = config.get("provenance") or {}
            referenced = (provenance.get("patch_dataset") or {}).get("name")
            if referenced == patch_dataset:
                used_by.append(name)
        return used_by

    # ── Stopping ─────────────────────────────────────────────────────────────

    def request_stop(self, name: str, reason: str = "petición del usuario") -> None:
        """Ask a run to stop. **Cooperative**: it cuts at the next safe point.

        A file rather than an in-memory event, for the same reason `status.json`
        is a file: the state of a run belongs to the run. It also means the CLI
        can be stopped the same way as the API, and that a stop survives an API
        restart -- and on CPU a run lasts long enough for that to happen.

        Nothing is killed. The loop finishes its epoch, writes its metrics and
        its checkpoint, and closes as `cancelled`. Killing the thread mid-batch
        would leave a half-written `last.pt`.
        """
        if not self.exists(name):
            raise RunNotFound(name)
        (self.path(name) / "stop.json").write_text(
            json.dumps({"requested_at": _now(), "reason": reason}), encoding="utf-8"
        )

    def stop_requested(self, name: str) -> bool:
        return (self.path(name) / "stop.json").exists()

    # ── Renaming and deleting ────────────────────────────────────────────────

    def rename(self, name: str, new_name: str) -> Path:
        source, target = self.path(name), self.path(new_name)
        if not source.is_dir():
            raise RunNotFound(name)
        if target.exists():
            raise RunExists(new_name)
        source.rename(target)
        return target

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.is_dir():
            raise RunNotFound(name)
        shutil.rmtree(path)
