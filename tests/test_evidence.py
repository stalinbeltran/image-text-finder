"""V18 — the evidence a corner's label actually has behind it.

**Why this exists at all.** A sliding window labels a patch with any corner whose
*point* falls inside it, and says nothing about whether that corner's paragraph
does. A TL landing near the patch's bottom-right has its whole paragraph outside
the window: the label asks for something the pixels do not show. Measured on
`dirty-20` that is 14 % of all corners, and they carry 31 % of the position error.

These tests do not assert those numbers -- results of investigation belong in the
protocol, never in pytest (tests.md §1). What they assert is that the measurement
means one thing: one definition, one corner order, the same answer in numpy and in
torch, and partitions that actually partition.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from itf.diagnostics import evidence_split, open_diagnostics, rows
from itf.geometry import CORNER_NAMES
from itf.metrics import BLIND_EVIDENCE, corner_evidence
from itf.metrics import _EVIDENCE_TOWARDS

from test_diagnostics import _build


def _xy(**corners: tuple[float, float]) -> np.ndarray:
    """(1, 4, 2) with the named corners placed and the rest at the centre."""
    out = np.full((1, 4, 2), 0.5, dtype=np.float32)
    for name, (x, y) in corners.items():
        out[0, CORNER_NAMES.index(name)] = (x, y)
    return out


class TestDefinition:
    """The geometry itself. Hand-checkable cases, because the formula is the claim."""

    def test_corner_at_the_far_edge_has_no_evidence(self):
        # A TL at the bottom-right: its paragraph runs right and down, i.e. out of
        # the patch. This is the whole phenomenon, in one number.
        assert corner_evidence(_xy(TL=(1.0, 1.0)))[0, 0] == pytest.approx(0.0)
        # And a TL at the top-left has the entire patch to show its paragraph in.
        assert corner_evidence(_xy(TL=(0.0, 0.0)))[0, 0] == pytest.approx(1.0)

    def test_each_corner_points_its_own_way(self):
        """The four are NOT the same function -- mixing them up is the easy bug.

        Placed at the same (0.9, 0.9), a TL is blind (its body left the patch) and
        a BR is nearly fully visible (its body is behind it, inside). A "distance
        to the edge" measure would call both of them edge cases and be wrong about
        half of them.
        """
        evidence = corner_evidence(_xy(TL=(0.9, 0.9), TR=(0.9, 0.9), BR=(0.9, 0.9), BL=(0.9, 0.9)))[0]
        by_name = dict(zip(CORNER_NAMES, evidence))
        assert by_name["TL"] == pytest.approx(0.01)
        assert by_name["BR"] == pytest.approx(0.81)
        assert by_name["TR"] == pytest.approx(0.09)
        assert by_name["BL"] == pytest.approx(0.09)

    def test_corner_order_matches_geometry(self):
        """The seam. `itf.metrics` imports nothing from `itf` **by design**, so the
        corner order is restated there -- and a restated constant that nobody
        checks is contract ⑤ waiting to happen. If someone reorders
        `CORNER_NAMES`, every evidence number silently starts describing the
        wrong corner: no crash, no shape change, just a different meaning.
        """
        assert len(_EVIDENCE_TOWARDS) == len(CORNER_NAMES) == 4
        expected = {"TL": (1, 1), "TR": (0, 1), "BR": (0, 0), "BL": (1, 0)}
        assert dict(zip(CORNER_NAMES, _EVIDENCE_TOWARDS)) == expected

    def test_same_answer_in_numpy_and_torch(self):
        """B's dataloader is torch, the diagnóstico is numpy, and this must not be
        two functions. It is written with arithmetic only for exactly that reason.
        """
        xy = np.random.default_rng(0).random((7, 4, 2), dtype=np.float32)
        np.testing.assert_allclose(
            corner_evidence(xy), corner_evidence(torch.from_numpy(xy)).numpy(), rtol=1e-6
        )

    def test_bounded(self):
        xy = np.random.default_rng(1).random((200, 4, 2), dtype=np.float32)
        evidence = corner_evidence(xy)
        assert evidence.min() >= 0.0 and evidence.max() <= 1.0


class TestOverATable:
    """Against a real run and a real B: the properties the payload promises."""

    def test_absent_corners_are_nan_never_zero(self, layout):
        """`0` here would mean "blind", and absent is not blind (formatos.md §2).

        This is the one that would rot silently: a 0 for "no corner" puts every
        empty slot into the blind band, which is the population the whole view is
        about -- it would inflate it with patches that have no corner at all.
        """
        diag = _open(layout)
        assert np.isnan(diag.evidence[~diag.exists]).all()
        assert not np.isnan(diag.evidence[diag.exists]).any()

    def test_bands_partition_the_corners(self, layout):
        out = evidence_split(_open(layout))
        assert sum(b["corners"] for b in out["bands"]) == out["corners"]
        assert out["blind"]["corners"] + out["seen"]["corners"] == out["corners"]

    def test_error_shares_sum_to_one(self, layout):
        """Each band's slice of the total error. If these do not close, the
        headline number ("14 % of the corners, 31 % of the error") is arithmetic
        nobody can trust."""
        out = evidence_split(_open(layout))
        shares = [b["error_share"] for b in out["bands"] if b["error_share"] is not None]
        assert sum(shares) == pytest.approx(1.0, abs=1e-6)
        assert out["blind"]["error_share"] + out["seen"]["error_share"] == pytest.approx(1.0, abs=1e-6)

    def test_threshold_is_free(self, layout):
        """Same free sweep as V8: recall moves, nothing recomputes. The cache key
        does not mention the threshold, so if this ever needed a pass it would be
        a silent 10⁵-forward regression."""
        diag = _open(layout)
        low = evidence_split(diag, threshold=0.1)
        high = evidence_split(diag, threshold=0.9)
        assert low["blind"]["corners"] == high["blind"]["corners"]
        assert low["blind"]["recall"] >= high["blind"]["recall"]

    def test_gallery_filters_are_complementary(self, layout):
        """`max_evidence` and `min_evidence` at the same cut must split the
        population, not overlap it -- otherwise "solo ciegas" and "solo visibles"
        would both show the same patch and neither label would mean anything."""
        diag = _open(layout)
        everything = rows(diag, outcome="fn", limit=1)["total"]
        blind = rows(diag, outcome="fn", max_evidence=BLIND_EVIDENCE, limit=1)["total"]
        seen = rows(diag, outcome="fn", min_evidence=BLIND_EVIDENCE, limit=1)["total"]
        assert blind <= everything and seen <= everything
        # A patch can hold two corners on opposite sides of the cut, so the two
        # sets may overlap by patch -- but neither may exceed the whole.
        assert blind + seen >= everything

    def test_rows_carry_their_evidence(self, layout):
        page = rows(_open(layout), limit=4)
        for row in page["rows"]:
            for exists, evidence in zip(row["exists"], row["evidence"]):
                assert (evidence is None) == (not exists)


class TestApi:
    def test_route_serves_the_split(self, itf_api):
        client, layout = itf_api
        _build(layout, run_name="ev-01")
        body = client.get("/runs/ev-01/diagnostics/evidence?split=val").json()
        assert body["blind_cut"] == BLIND_EVIDENCE
        assert body["patch_size"] == 40
        assert body["bands"]

    def test_unknown_corner_is_400(self, itf_api):
        client, layout = itf_api
        _build(layout, run_name="ev-02")
        response = client.get("/runs/ev-02/diagnostics/evidence?split=val&corner=XX")
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "unknown_corner"


def _open(layout):
    _build(layout, run_name="ev-table")
    from itf.diagnostics import TableCache
    from itf.patches import PatchDatasetStore
    from itf.training.registry import RunStore

    return open_diagnostics(
        runs=RunStore(layout.runs),
        patch_datasets=PatchDatasetStore(layout.patch_datasets),
        cache=TableCache(layout.cache),
        run="ev-table",
        split="val",
    )
