"""Where a run came from (contract ③, shape fixed by D2).

**The name AND the value, and that is not redundancy** -- it is exactly what
contract ③ found missing. The value reproduces; the name groups. With only the
value you must diff dictionaries by hand to ask *"which runs used network X?"*,
which is the question a sweep asks constantly (organizacion.md §2-③).

The five fields are mandatory and **none of them is ever filled in silently**
(formatos.md §2): if the commit cannot be known, the reason goes in, not a quiet
`null`. A `null` there would read as "no commit", which is not a thing -- what
happened is "I could not find out", and those are different facts.

`environment` closes the hole `git_commit` leaves: the commit fixes **the code**,
not **the interpreter**. Going from torch 2.13 to 2.14 moves results just like
touching the loss does, without moving the commit -- and the plan includes moving
to GPU, where the environment changes wholesale. The CPU runs of today are
exactly the ones that will be compared against tomorrow's GPU runs, so they are
the ones that need to be able to say what they ran on.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Where to ask git. The repo root, from src/itf/training/provenance.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The five mandatory fields plus `sweep` (formatos.md §4.2.1). Declared once so
#: that "is this provenance complete?" has a single answer -- the reader, the
#: writer and the test all ask this tuple.
PROVENANCE_FIELDS = (
    "patch_dataset",
    "network",
    "recipe",
    "sweep",
    "git_commit",
    "environment",
)


def git_commit(cwd: Path | None = None) -> str:
    """The commit this code is at, or **the reason it cannot be known**.

    Never `null` and never an empty string: both read as "there is no commit",
    when the fact is "I could not find out" (formatos.md §2). A dirty tree is
    reported as such, because rule 1 of protocolo.md §7 is *same git commit* --
    and with uncommitted changes the sha does not identify the code that ran, so
    a bare sha would be a lie the comparison rules rest on.
    """
    root = cwd or _REPO_ROOT

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10
        )

    try:
        # `rev-parse HEAD` fails BOTH when this is not a repo and when it is one
        # with no commits yet -- same exit code, different fact -- so the two are
        # told apart by asking. Reporting "not a repo" for an empty one would put
        # a false reason in the provenance of every run, which is worse than
        # saying nothing: formatos.md §2 asks for the reason, not for a reason.
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            inside = git("rev-parse", "--is-inside-work-tree")
            if inside.returncode == 0 and inside.stdout.strip() == "true":
                return "desconocido: el repositorio no tiene ningún commit todavía"
            return f"desconocido: '{root}' no es un repositorio git"
        sha = head.stdout.strip()
        status = git("status", "--porcelain")
        if status.returncode != 0:
            return f"desconocido: git status falló ({status.stderr.strip()[:80]})"
        dirty = status.stdout.strip()
    except FileNotFoundError:
        return "desconocido: git no está instalado"
    except subprocess.SubprocessError as exc:
        return f"desconocido: git falló ({type(exc).__name__})"
    # The suffix is deliberate rather than a separate boolean: whoever compares
    # two runs compares this string, and an equality check must not pass for two
    # dirty trees that were different.
    return f"{sha}+sucio" if dirty else sha


def environment() -> dict[str, str]:
    """The interpreter the run happened on. What the commit cannot capture."""
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": sys.platform,
    }


def build_provenance(
    *,
    patch_dataset: dict[str, Any],
    network: dict[str, Any],
    recipe: dict[str, Any],
    sweep: str | None = None,
) -> dict[str, Any]:
    """The provenance block of `runs/<name>/config.json` (formatos.md §4.2.1).

    Args:
        patch_dataset: ``{"name", "fingerprint"}`` -- B by name AND by content.
            The fingerprint is what tells a rebuilt B from the original: a path
            rebuilt under the same name still points the same way, so without it
            a half-finished sweep goes incomparable in silence (contract ⑧).
        network: ``{"name", "value"}`` -- C.
        recipe: ``{"name", "value"}`` -- D.
        sweep: the parent sweep's name, or None. `None` here is a real fact
            ("this run has no sweep"), unlike an unknown commit -- which is why
            it is the one field allowed to be null.
    """
    for label, ref in (("network", network), ("recipe", recipe)):
        if not ref.get("name"):
            # R7 of api.md: if it was trained with, it has a name. Refusing here
            # is what makes contract ③ hold by itself -- a run that cannot say
            # which C it came from is the exact hole the contract describes.
            raise ValueError(f"la procedencia de '{label}' necesita un nombre, no solo el valor")
    if not patch_dataset.get("fingerprint"):
        raise ValueError("la procedencia de 'patch_dataset' necesita la huella (contrato ⑧)")

    return {
        "patch_dataset": {
            "name": patch_dataset["name"],
            "fingerprint": patch_dataset["fingerprint"],
        },
        "network": {"name": network["name"], "value": network["value"]},
        "recipe": {"name": recipe["name"], "value": recipe["value"]},
        "sweep": sweep,
        "git_commit": git_commit(),
        "environment": environment(),
    }
