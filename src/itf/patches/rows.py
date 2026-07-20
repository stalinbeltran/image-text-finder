"""Row-addressable access to a B, so looking at ONE patch costs one page of disk.

**Why this exists.** `patches.npz` is written with `np.savez_compressed`
(`extract.py`), which is a zip of deflated `.npy` members. A deflate stream has no
positions: to reach the bytes of patch 1 245 659 you must inflate everything
before it. NumPy does not try -- `data["X"][idx]` inflates the **whole member**
and then indexes the result. The comment this module replaces said "`np.load`
does not read until you index it", which is true and misleading: the *load* is
lazy, the *index* is not.

Measured on `dirty-20` (6 283 620 patches, `X` = 2,5 GB raw, 134 MB on disk):

    np.load(...)        0,001 s
    data["X"][idx]      6,5 s and 2,5 GB of RAM

Clicking one patch in the worst-first gallery fires four probes at once (V2, V4,
V10, V16 -- `Gallery.tsx`), each resolving the patch through `_patch_from_body`.
Four threads inflating 2,5 GB each is ~10 GB live, and the machine goes to swap:
the screen did not fail, it **hung**. This is the `labels.jsonl` trap of
`itf.datasets.index` wearing a different hat, and it has the same cure: **to look
at one row you do not touch the whole array**.

The fix is an uncompressed copy, memory-mapped. An uncompressed `.npy` is a
header plus raw bytes, so the offset of any row is arithmetic and the OS faults
in only the page it needs: ~0,1 ms, no allocation.

**It is a cache, like the per-patch tables (D1) and the source index**:
recomputable from the `.npz`, keyed by that file's size and mtime, and deleting
`data/cache/patch-rows/` loses nothing but the ~30 s of the first rebuild. The
key is not decoration -- a `patches.npz` rebuilt under the same name must not be
served through the previous file's rows, and a stale row does not fail loudly, it
returns **the wrong patch**, which is the only silent failure this could add.

**What it does NOT replace.** `PatchDatasetStore.arrays()` stays, for the one
consumer that legitimately wants every row at once: `diagnostics/service.py`
joining the truth for a whole split. One bulk inflate beats millions of random
page faults. The rule is the same one `samples()` got: *bulk goes through the
`.npz`, a single row goes through here.*
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import numpy as np

#: Bump when the member list or layout changes: an old cache directory is then a
#: miss and gets rebuilt, rather than a `KeyError` inside a view.
ROWS_VERSION = 1

#: The members that are one-row-per-patch. All of them, because `GET
#: /patch-datasets/{name}/patches/{index}` reads all of them, and the four
#: besides `X` add 380 MB to X's 2,5 GB -- not worth a second code path.
MEMBERS = ("X", "y", "border", "sample_idx", "patch_xy", "split")

#: Building inflates 2,5 GB. The four probes of one gallery click arrive
#: together, so without this they would each build their own copy and reproduce
#: the very blow-up this module exists to remove. Blocking is correct here: the
#: losers wait for the winner's result, which is what they wanted anyway.
_BUILD_LOCK = threading.Lock()

#: In-process memo, keyed by path AND fingerprint. The memmaps are cheap to keep
#: (they are views on disk, not buffers), and re-opening six files per probe is
#: work to learn something we knew a millisecond ago. Keyed by fingerprint so a
#: rebuilt B invalidates itself here exactly as it does on disk, instead of
#: serving dead rows for as long as the process lives.
_MEMO: dict[Path, tuple[dict, dict[str, np.memmap]]] = {}


class PatchRows:
    """One B's per-row members, memory-mapped. Index it; do not slice it whole."""

    def __init__(self, arrays: dict[str, np.memmap]):
        self._arrays = arrays

    def __len__(self) -> int:
        return int(self._arrays["X"].shape[0])

    def __getitem__(self, member: str) -> np.memmap:
        return self._arrays[member]

    def row(self, index: int) -> dict[str, np.ndarray]:
        """Every member at one index, as real arrays.

        `np.array(...)` on purpose: a memmap slice stays a view on the file, and
        handing one to a caller that outlives this call is a promise about the
        file we cannot keep. A single row is 400 bytes.
        """
        return {name: np.array(arr[index]) for name, arr in self._arrays.items()}


def _fingerprint(npz_path: Path) -> dict:
    st = npz_path.stat()
    return {"version": ROWS_VERSION, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _slug(npz_path: Path) -> str:
    """The B's path, readable. Not a hash, for the reason
    `itf.datasets.index.cache_path` gives: when a cache misbehaves the first
    thing anyone does is look in the directory, and `data_patch-datasets_dirty-20/`
    answers what `a3f19c.../` does not.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", npz_path.parent.resolve().as_posix()).strip("_")[-120:]


def cache_dir(npz_path: Path, cache_root: Path) -> Path:
    """`<slug>/<size>-<mtime>/` — **the fingerprint is in the path**, deliberately.

    The obvious layout is one directory per B, overwritten when the `.npz`
    changes. It does not work on Windows: a memory-mapped file cannot be
    deleted or replaced while any mapping is live, and after a rebuild ours are
    live (the memo holds them, and so may a request in flight). `rmtree` then
    raises `PermissionError` and the rebuild fails -- on the one path that
    matters, a B rewritten under the same name.

    Putting the fingerprint in the directory name removes the delete entirely: a
    new version of a B is a new directory, and the old one is garbage to be
    collected when nothing holds it, not an obstacle in the way of a build.
    """
    st = npz_path.stat()
    return cache_root / _slug(npz_path) / f"{st.st_size}-{st.st_mtime_ns}"


def _read_cached(npz_path: Path, cache_root: Path) -> dict[str, np.memmap] | None:
    """The memmaps if a cache exists *for this exact `.npz`*, else `None`.

    A stale fingerprint is a miss, never a repair.
    """
    path = cache_dir(npz_path, cache_root)
    stamp = path / "fingerprint.json"
    if not stamp.exists():
        return None
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) != _fingerprint(npz_path):
            return None
        return {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in MEMBERS}
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _build(npz_path: Path, cache_root: Path) -> dict[str, np.memmap]:
    """Inflate once, write uncompressed, then map it back.

    Member by member, so the peak is one member (X, 2,5 GB) and not their sum.
    The fingerprint is written **last**: it is what makes the directory valid, so
    a build killed halfway is a miss on the next read rather than a directory of
    truncated arrays that reads as good.
    """
    path = cache_dir(npz_path, cache_root)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Fingerprint taken BEFORE reading, so a `.npz` rewritten *during* the build
    # produces a stamp that no longer matches it -- a miss, not a silent mix of
    # rows from two datasets.
    fp = _fingerprint(npz_path)
    with np.load(npz_path) as data:
        for name in MEMBERS:
            np.save(tmp / f"{name}.npy", data[name])
    (tmp / "fingerprint.json").write_text(json.dumps(fp), encoding="utf-8")

    tmp.replace(path)
    _sweep_old_versions(path)
    return {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in MEMBERS}


def _sweep_old_versions(current: Path) -> None:
    """Drop this B's other versions. Best effort, on purpose.

    Each version is gigabytes, so leaving them forever is not an option -- but
    neither is failing a build over one. On Windows a directory still mapped by
    a request in flight cannot be removed, and that is a perfectly normal state,
    not an error: skip it and the next build collects it.
    """
    for sibling in current.parent.iterdir():
        if sibling == current or not sibling.is_dir():
            continue
        try:
            shutil.rmtree(sibling)
        except OSError:
            pass


def load_rows(npz_path: str | Path, cache_root: Path) -> PatchRows:
    """Row access to a B: from memory, else from disk, else built (and cached).

    The first call on a dataset pays the inflate (~30 s for `dirty-20`) and the
    disk (2,9 GB). Every call after it is a `mmap` of six files.
    """
    npz_path = Path(npz_path).resolve()
    fp = _fingerprint(npz_path)

    memo = _MEMO.get(npz_path)
    if memo is not None and memo[0] == fp:
        return PatchRows(memo[1])

    arrays = _read_cached(npz_path, cache_root)
    if arrays is None:
        with _BUILD_LOCK:
            # Re-check inside the lock: while we queued, the thread ahead of us
            # very likely built exactly this. Otherwise the four probes of one
            # click build it four times, serially, which is worse than the bug.
            arrays = _read_cached(npz_path, cache_root)
            if arrays is None:
                arrays = _build(npz_path, cache_root)

    _MEMO[npz_path] = (fp, arrays)
    return PatchRows(arrays)
