"""`itf.patches.rows` — the memmap cache over a B's per-row arrays.

The failure this guards against is not a crash: a row cache keyed loosely serves
**the wrong patch**, and a wrong patch looks exactly like a right one. So the
tests are about identity (same bytes as the `.npz`) and invalidation (a rebuilt B
is a miss), not about speed.
"""

from __future__ import annotations

import numpy as np
import pytest

from itf.patches.rows import MEMBERS, cache_dir, load_rows


def _write_npz(path, n=64, size=8):
    rng = np.random.default_rng(0)
    arrays = {
        "X": rng.integers(0, 256, (n, size, size, 1), dtype=np.uint8),
        "y": rng.random((n, 4, 3)).astype(np.float32),
        "border": rng.integers(0, 2, (n, 4), dtype=np.uint8),
        "sample_idx": np.arange(n, dtype=np.int32),
        "patch_xy": rng.integers(0, 100, (n, 2), dtype=np.int32),
        "split": np.zeros(n, dtype=np.int8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def test_rows_are_the_same_bytes_as_the_npz(tmp_path):
    """The whole point: identical rows, without inflating the whole array."""
    npz = tmp_path / "b" / "patches.npz"
    expected = _write_npz(npz)
    rows = load_rows(npz, tmp_path / "cache")

    assert len(rows) == 64
    for i in (0, 17, 63):
        row = rows.row(i)
        for member in MEMBERS:
            np.testing.assert_array_equal(row[member], expected[member][i])


def test_deleting_the_cache_loses_nothing(tmp_path):
    """D1's property, and what makes this a cache and not an artifact."""
    npz = tmp_path / "b" / "patches.npz"
    expected = _write_npz(npz)
    cache = tmp_path / "cache"

    before = load_rows(npz, cache).row(5)

    # Drop the in-process memo FIRST -- and not only because the memo would
    # otherwise answer from RAM and the deleted directory never be missed: it
    # holds the memmaps, and on Windows a mapped file cannot be unlinked. This
    # ordering is the same constraint `cache_dir` is shaped around.
    import shutil

    import itf.patches.rows as rows_mod

    rows_mod._MEMO.clear()
    shutil.rmtree(cache_dir(npz, cache))

    after = load_rows(npz, cache).row(5)
    for member in MEMBERS:
        np.testing.assert_array_equal(before[member], after[member])
    np.testing.assert_array_equal(after["X"], expected["X"][5])


def test_a_rebuilt_dataset_is_a_miss_not_a_stale_row(tmp_path):
    """The only silent failure this module could introduce.

    A B rebuilt under the same name must not be read through the previous
    file's rows. It would not raise -- it would return another patch's pixels
    under this patch's index, which is indistinguishable from a real answer.
    """
    npz = tmp_path / "b" / "patches.npz"
    _write_npz(npz)
    cache = tmp_path / "cache"
    load_rows(npz, cache)  # warms both the disk cache and the memo

    # Same name, different content -- and a different size, which is what the
    # fingerprint sees even if the mtime resolution were coarse.
    rebuilt = _write_npz(npz, n=128)

    rows = load_rows(npz, cache)
    assert len(rows) == 128
    np.testing.assert_array_equal(rows.row(100)["X"], rebuilt["X"][100])


def test_a_row_survives_the_file_it_came_from(tmp_path):
    """`row()` copies. A memmap view handed out would be a promise about disk."""
    npz = tmp_path / "b" / "patches.npz"
    expected = _write_npz(npz)
    row = load_rows(npz, tmp_path / "cache").row(3)
    assert not isinstance(row["X"], np.memmap)
    np.testing.assert_array_equal(row["X"], expected["X"][3])


@pytest.mark.parametrize("member", MEMBERS)
def test_every_member_is_row_addressable(tmp_path, member):
    """If a member ever stops being one-row-per-patch, `row()` lies about it."""
    npz = tmp_path / "b" / "patches.npz"
    expected = _write_npz(npz)
    rows = load_rows(npz, tmp_path / "cache")
    assert rows[member].shape[0] == expected[member].shape[0] == len(rows)
