"""A byte-offset index over a source's `labels.jsonl`, so one image costs one seek.

**Why this exists.** `labels.jsonl` is one JSON object per image, and the object
carries the whole nested geometry -- `blocks[]`, and inside them `lines[]` and
`words[]`. For `dirty-paragraphs-80ancho` that is **522 MB over 20 000 lines**,
and `json.loads` on all of it takes **30 s**. Every read of domain A went through
`SourceDataset.samples()`, which parses the file whole: listing the sources,
listing the images, *and each thumbnail, each predict, each scrubber drag*. A
gallery of 24 thumbnails was 12 minutes of CPU, and the screen looked broken
rather than slow.

The fix is not to parse less carefully -- it is to parse **once**. This module
records, per image, where its line starts and the handful of fields a listing
needs, and persists that. After the first build, `/samples` is a file read of a
~1 MB JSON and one image is `seek` + one 26 KB `json.loads`.

**It is a cache, exactly like the per-patch tables (D1)**: recomputable from the
file it indexes, keyed by that file's size and mtime, and deleting
`data/cache/sources/` loses nothing. The key matters -- a source that is rewritten
(the resize writes A', D19) must not serve the previous file's offsets, and an
offset into a changed file does not fail loudly, it decodes the *wrong image*.

Domain A stays read-only: the index is written under our cache root, never
beside `labels.jsonl`, because `datasets_root` belongs to the generator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

#: Bump when the record shape changes: an old cache file is then simply a miss,
#: rather than a `KeyError` deep inside a listing.
INDEX_VERSION = 1


@dataclass(frozen=True)
class SampleSummary:
    """What a listing needs about one image, plus where to find the rest.

    Deliberately not a `Sample`: this is what `GET /sources/{id}/samples`
    answers, and it holds no geometry. The blocks are the 26 KB we are avoiding.
    """

    index: int
    width: int
    height: int
    has_overlap: bool
    num_blocks: int
    #: Byte offset of this image's line in `labels.jsonl`.
    offset: int


class SourceIndex:
    """The summaries of one source, addressable by image index."""

    def __init__(self, entries: list[SampleSummary]):
        self.entries = entries
        self._by_index = {e.index: e for e in entries}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_overlapping(self) -> int:
        return sum(1 for e in self.entries if e.has_overlap)

    def get(self, index: int) -> SampleSummary | None:
        return self._by_index.get(index)


def count_lines(labels_path: Path) -> int:
    """How many images, without parsing any of them (~1 s for 522 MB).

    This is what lets `GET /sources` stay cheap on a source nobody has opened
    yet: the count is honest and the expensive fields are reported absent.
    """
    n = 0
    with labels_path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def build_index(labels_path: Path) -> SourceIndex:
    """Read the file once, keeping only the summary and the offset.

    Full `json.loads` per line, not a regex over the raw bytes: this runs once
    per version of the file, and a hand-rolled JSON scanner that is subtly wrong
    on an escaped quote would mis-index silently -- which is the failure mode
    this module exists to prevent.
    """
    entries: list[SampleSummary] = []
    with labels_path.open("rb") as fh:
        offset = 0
        for raw in fh:
            start = offset
            offset += len(raw)
            if not raw.strip():
                continue
            rec = json.loads(raw.decode("utf-8"))
            lab = rec["labels"]
            entries.append(
                SampleSummary(
                    index=int(rec["index"]),
                    width=int(lab["width"]),
                    height=int(lab["height"]),
                    has_overlap=bool(lab.get("has_overlap", False)),
                    num_blocks=len(lab["blocks"]),
                    offset=start,
                )
            )
    return SourceIndex(entries)


def _fingerprint(labels_path: Path) -> dict:
    st = labels_path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def cache_path(labels_path: Path, cache_root: Path) -> Path:
    """One file per source, named after its path so it is greppable by a human.

    Not a hash: when the cache misbehaves the first thing anyone does is look in
    the directory, and `data_sources_dirty-paragraphs-80ancho.json` answers the
    question that `a3f19c...json` does not.
    """
    # `resolve()` first: `data/sources/x` and `C:/Desarrollo/.../data/sources/x`
    # are the same source, and without this they get two cache files -- so the
    # CLI builds one index and the API, which resolves its roots, rebuilds it
    # from scratch. Caught by timing the real endpoint: a warm cache still cost
    # its full 27 s.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", labels_path.parent.resolve().as_posix()).strip("_")
    return cache_root / f"{slug[-120:]}.json"


def read_cached(labels_path: Path, cache_root: Path) -> SourceIndex | None:
    """The index if one is on disk *for this exact file*, else `None`.

    A stale fingerprint is a miss, never a repair: an index that half-matches is
    an index that returns the wrong image (see the module docstring).
    """
    path = cache_path(labels_path, cache_root)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if blob.get("version") != INDEX_VERSION or blob.get("fingerprint") != _fingerprint(labels_path):
        return None
    return SourceIndex([SampleSummary(*row) for row in blob["entries"]])


def write_cached(labels_path: Path, cache_root: Path, index: SourceIndex) -> None:
    """Persist, atomically. A half-written index would read as a valid one."""
    path = cache_path(labels_path, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "version": INDEX_VERSION,
        "fingerprint": _fingerprint(labels_path),
        # Tuples, not objects: 20 000 rows of five keys each is 3 MB of field
        # names for nothing. The shape is pinned by INDEX_VERSION.
        "entries": [
            [e.index, e.width, e.height, e.has_overlap, e.num_blocks, e.offset] for e in index.entries
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8")
    tmp.replace(path)


#: In-process memo on top of the disk cache, keyed by path AND fingerprint.
#: Without it a 24-thumbnail gallery re-reads and re-parses the same ~1.5 MB
#: cache file 24 times -- much better than 24 × 522 MB, but still work done to
#: learn something we knew a millisecond ago. Keyed by the fingerprint and not
#: just the path, so a rewritten source (the resize writes A', D19) invalidates
#: itself exactly as the disk cache does, instead of serving dead offsets for as
#: long as the process lives -- the failure this whole module guards against.
_MEMO: dict[Path, tuple[dict, SourceIndex]] = {}


def peek_index(labels_path: Path, cache_root: Path) -> SourceIndex | None:
    """The index **if it already exists**, from memory or disk. Never builds.

    This is what keeps `GET /sources` cheap: the picker reports what it happens
    to know and refuses to spend 30 s per source computing what it does not.
    """
    labels_path = Path(labels_path).resolve()
    fp = _fingerprint(labels_path)
    memo = _MEMO.get(labels_path)
    if memo is not None and memo[0] == fp:
        return memo[1]
    index = read_cached(labels_path, cache_root)
    if index is not None:
        _MEMO[labels_path] = (fp, index)
    return index


def load_index(labels_path: Path, cache_root: Path) -> SourceIndex:
    """The index: from memory, else from disk, else built (and then cached)."""
    # Resolved for the same reason `cache_path` resolves: two spellings of one
    # path must be one entry, not two.
    labels_path = Path(labels_path).resolve()
    index = peek_index(labels_path, cache_root)
    if index is None:
        index = build_index(labels_path)
        try:
            write_cached(labels_path, cache_root, index)
        except OSError:
            # A cache we cannot write is slow, not broken.
            pass
        # Fingerprint taken AFTER the build, from the file we actually read.
        _MEMO[labels_path] = (_fingerprint(labels_path), index)
    return index
