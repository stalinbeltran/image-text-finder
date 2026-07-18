"""Where sources live, and how an id maps to a directory. **Once.**

Since D19 there are **two** source roots -- the external, read-only one and the
local one the resize writes -- and three callers that must agree on how an id
resolves: `GET /sources`, `itf-extract` and `itf-resize`. Three copies of "try
this root, then that one, mind the prefix" is the exact shape of contract ⑤: they
would be identical, so it would work, and the day one of them learned about a
third root the others would silently keep finding nothing.

So the mapping lives here and takes **explicit paths**, never `Settings`: the
domain does not read the environment, entry points do (settings.py).
"""

from __future__ import annotations

from pathlib import Path

from itf.datasets.loader import discover_sources

#: Ids under the local root wear this prefix. It is what stops a derived source
#: from shadowing an original with the same name -- the two roots are not
#: interchangeable, and one of them we must never write to.
DERIVED_PREFIX = "derived"


def source_roots(datasets_root: Path, derived_root: Path) -> tuple[tuple[str, Path], ...]:
    """The roots to search, each with the prefix its ids carry. Order is lookup order."""
    return (("", Path(datasets_root)), (DERIVED_PREFIX + "/", Path(derived_root)))


def split_id(source_id: str, roots: tuple[tuple[str, Path], ...]) -> tuple[Path, str]:
    """Which root an id addresses, and the path within it. Does not touch disk."""
    for prefix, root in roots:
        if prefix and source_id.startswith(prefix):
            return root, source_id[len(prefix) :]
    return roots[0][1], source_id


def resolve(value: str, roots: tuple[tuple[str, Path], ...]) -> Path | None:
    """A directory for ``value``: an absolute path, a relative one, or an id.

    `None` rather than an exception: each caller words its own refusal, and the
    CLI's wording (listing what does exist) is half the value of the failure.
    """
    as_path = Path(value).expanduser()
    if as_path.is_dir():
        return as_path.resolve()

    root, rel = split_id(value, roots)
    candidate = (root / rel).expanduser()
    return candidate.resolve() if candidate.is_dir() else None


def list_ids(roots: tuple[tuple[str, Path], ...]) -> list[tuple[str, Path]]:
    """Every source in every root, as ``(id, path)``.

    `discover_sources`, not `iterdir`: a source is a directory holding a
    `labels.jsonl`, and the generator NESTS them (`clean-paragraphs-01/reducido`).
    Listing top-level directories would offer `clean-paragraphs-01`, which is not
    itself a source -- and an error message that lies is worse than none.
    """
    out: list[tuple[str, Path]] = []
    for prefix, root in roots:
        for path in discover_sources(root):
            out.append((prefix + path.relative_to(root).as_posix(), path))
    return out
