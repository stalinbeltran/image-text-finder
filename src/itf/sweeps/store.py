"""H — the sweep registry on disk: `sweeps/<name>/`.

    sweeps/<name>/
    ├── spec.json     # what is fixed, the space, the objective, the budget
    ├── optuna.db     # optuna's SQLite storage: the trials and their values
    └── stop.json     # the cooperative-stop request, like a run's

**The border formatos.md §4.5 draws lives here**: `spec.json` is OURS (versioned,
it is the description of the experiment), `optuna.db` is the ENGINE's (load, like a
checkpoint -- derivable from re-running, ignored by git). A trial is not a run: a
trial launches a run and stores its name.

No optuna import in this module on purpose -- it manages files and paths. Reading
the trials themselves (which needs the engine) is `runner.read_progress`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from itf.sweeps.spec import SweepSpec


class SweepNotFound(KeyError):
    """No such sweep."""


class SweepExists(FileExistsError):
    """A sweep by that name is already on disk. Never overwritten (like a run)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SweepStore:
    """`sweeps/<name>/`, one subdirectory per H."""

    root: Path

    def path(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de barrido inválido: {name!r}")
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "spec.json").exists()

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "spec.json").exists())

    def storage_url(self, name: str) -> str:
        """The SQLite URL optuna persists the study to. Forward slashes so the URL
        is valid on Windows too."""
        db = (self.path(name) / "optuna.db").resolve()
        return "sqlite:///" + db.as_posix()

    def db_path(self, name: str) -> Path:
        return self.path(name) / "optuna.db"

    def create(self, spec: SweepSpec) -> Path:
        """Reserve `sweeps/<name>/` and freeze its spec. **Refuses to overwrite.**

        Same rule as a run: a sweep that auto-generates run names must never step
        on an earlier sweep's record.
        """
        path = self.path(spec.name)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SweepExists(spec.name) from exc
        (path / "spec.json").write_text(json.dumps(spec.as_dict(), indent=2), encoding="utf-8")
        return path

    def spec(self, name: str) -> SweepSpec:
        path = self.path(name) / "spec.json"
        if not path.exists():
            raise SweepNotFound(name)
        return SweepSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ── Stopping ─────────────────────────────────────────────────────────────

    def request_stop(self, name: str, reason: str = "petición del usuario") -> None:
        """Ask the sweep to stop. Cooperative: it cuts between trials (and the
        running trial's run is asked to stop at its epoch end)."""
        if not self.exists(name):
            raise SweepNotFound(name)
        (self.path(name) / "stop.json").write_text(
            json.dumps({"requested_at": _now(), "reason": reason}), encoding="utf-8"
        )

    def stop_requested(self, name: str) -> bool:
        return (self.path(name) / "stop.json").exists()

    def clear_stop(self, name: str) -> None:
        (self.path(name) / "stop.json").unlink(missing_ok=True)

    def delete(self, name: str) -> None:
        import shutil

        path = self.path(name)
        if not path.is_dir():
            raise SweepNotFound(name)
        shutil.rmtree(path)
