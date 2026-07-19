"""The source offset index (`itf.datasets.index`).

The index exists for speed, but what needs a test is not speed -- it is that the
shortcut answers the SAME thing the long way answers. An offset into a
`labels.jsonl` is a raw byte position: if it is off, nothing raises, the wrong
image is decoded and every screen quietly shows image 41's geometry over image
40's pixels. So these tests are all one question in different clothes: **does
`sample_at(index.get(i).offset)` equal `samples()[i]`?**

Same shape as the D1 cache tests: it is a cache, so deleting it must lose
nothing, and a stale one must be a miss rather than a repair.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import write_tiny_source

from itf.datasets import SourceDataset
from itf.datasets.index import (
    build_index,
    cache_path,
    count_lines,
    load_index,
    read_cached,
)
import itf.datasets.index as index_module


def _source(tmp_path: Path, **kwargs) -> SourceDataset:
    write_tiny_source(tmp_path / "src", **kwargs)
    return SourceDataset(tmp_path / "src")


def test_index_addresses_the_same_samples_as_a_full_parse(tmp_path: Path):
    """The shortcut and the long way agree, image by image.

    Not "is the offset plausible" but "does it decode the right record" --
    including the LAST line, which is where an off-by-one in the byte accounting
    shows up first.
    """
    ds = _source(tmp_path, num_samples=9)
    index = build_index(ds.labels_path)
    full = ds.samples()

    assert len(index) == len(full)
    for entry, expected in zip(index.entries, full):
        got = ds.sample_at(entry.offset)
        assert got.index == expected.index == entry.index
        assert (got.width, got.height) == (entry.width, entry.height)
        assert len(got.blocks) == entry.num_blocks
        assert [b.quad.tolist() for b in got.blocks] == [b.quad.tolist() for b in expected.blocks]


def test_count_lines_matches_the_number_of_samples(tmp_path: Path):
    """`GET /sources` counts images without parsing them; it must not undercount."""
    ds = _source(tmp_path, num_samples=7)
    assert count_lines(ds.labels_path) == len(ds.samples()) == 7


def test_deleting_the_cache_loses_nothing(tmp_path: Path):
    """Same claim as the per-patch tables (D1): it is derived, so it is disposable."""
    ds = _source(tmp_path, num_samples=5)
    cache = tmp_path / "cache"

    first = load_index(ds.labels_path, cache)
    assert cache_path(ds.labels_path, cache).exists()

    cache_path(ds.labels_path, cache).unlink()
    index_module._MEMO.clear()

    rebuilt = load_index(ds.labels_path, cache)
    assert [vars(e) for e in rebuilt.entries] == [vars(e) for e in first.entries]


def test_a_rewritten_source_invalidates_the_index(tmp_path: Path):
    """The one failure that would be silent, so it is the one that must be loud.

    Offsets from the previous version of a file do not raise when the file grows
    or shrinks -- they decode a different image. Both the disk cache and the
    in-process memo are keyed on size+mtime for exactly this.
    """
    ds = _source(tmp_path, num_samples=4)
    cache = tmp_path / "cache"
    load_index(ds.labels_path, cache)

    # Rewrite with the records in a different order and one more image: every
    # offset from the first build now points somewhere else.
    records = [json.loads(line) for line in ds.labels_path.read_text(encoding="utf-8").splitlines() if line]
    records.reverse()
    records.append({**records[0], "index": 99})
    ds.labels_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    assert read_cached(ds.labels_path, cache) is None, "a stale index must be a miss, not a repair"

    fresh = load_index(ds.labels_path, cache)
    assert len(fresh) == 5
    for entry in fresh.entries:
        assert ds.sample_at(entry.offset).index == entry.index


def test_api_lists_and_serves_a_source_through_the_index(itf_api):
    """End to end: the picker, the gallery and one image, over HTTP.

    `num_overlapping` is `null` until an index exists -- absent, not zero
    (formatos.md §2) -- and becomes a number once the gallery has built one.
    """
    client, layout = itf_api
    name = layout.write_source("tiny")

    listed = client.get("/sources").json()["sources"]
    entry = next(s for s in listed if s["id"] == name)
    assert entry["num_samples"] == 6
    assert entry["num_overlapping"] is None

    samples = client.get(f"/sources/{name}/samples").json()["samples"]
    assert [s["index"] for s in samples] == list(range(6))
    assert all(s["num_blocks"] == 1 for s in samples)

    # The geometry of one image still comes back whole, and it is the RIGHT one.
    geom = client.get(f"/sources/{name}/samples/3/geometry").json()
    assert geom["index"] == 3
    assert geom["blocks"][0]["quad"] == ds_quad(layout, 3)

    assert client.get(f"/sources/{name}/samples/999/image").status_code == 404
    assert client.get(f"/sources/{name}/samples/3/image").status_code == 200

    entry = next(s for s in client.get("/sources").json()["sources"] if s["id"] == name)
    assert entry["num_overlapping"] == 0


def ds_quad(layout, index: int) -> list[list[float]]:
    """The ground-truth quad of one image, read the slow, obvious way."""
    ds = SourceDataset(layout.datasets / "tiny")
    sample = next(s for s in ds.samples() if s.index == index)
    return sample.blocks[0].quad.tolist()
