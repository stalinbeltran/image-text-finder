"""A' — the derived source (D19).

What these assert is the SEAM, not the arithmetic (tests.md §1.2): that the
pixels and the geometry end up describing the same thing. `img.resize()` being
correct is PIL's problem; a quad landing where its corner actually went is ours.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import write_tiny_source

from itf.datasets import SourceDataset
from itf.datasets.resize import ResizeRefused, ResizeRequest, check_resize, resize_source, source_sizes
from itf.imageops import resize as resize_image
from itf.imageops import target_size

SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- #
# The mechanism: itf.imageops
# --------------------------------------------------------------------------- #


def test_imageops_imports_nothing_from_itf():
    """The whole reason it is its own module (tests.md §4, librerias.md §2).

    It has to stay usable on an arbitrary test image with no dataset anywhere
    near it. One import of `itf` and that stops being true -- and it would be a
    convenient import, which is how every layering trap in this project started.
    """
    tree = ast.parse((SRC / "itf" / "imageops.py").read_text(encoding="utf-8"))
    found = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("itf")
    ]
    assert found == []


def test_aspect_ratio_is_kept_and_one_dimension_derived():
    assert target_size(640, 480, to_width=320) == (320, 240)
    assert target_size(640, 480, to_height=240) == (320, 240)


def test_asking_for_both_dimensions_is_refused():
    """Two dimensions is a distortion request, and this module does not do those."""
    with pytest.raises(ValueError):
        target_size(640, 480, to_width=320, to_height=200)
    with pytest.raises(ValueError):
        target_size(640, 480)


def test_scale_is_measured_from_the_output_not_the_request():
    """The trap the module exists to avoid.

    641 -> 320 is a factor of 0.4992 on x, but the derived height rounds to
    240/480 = 0.5000 exactly. Reusing the requested factor for both axes puts
    every quad a fraction of a pixel off -- too small to see, big enough to be
    wrong, and nothing anywhere would report it.
    """
    result = resize_image(Image.new("L", (641, 480)), to_width=320)
    sx, sy = result.scale
    assert (sx, sy) == (320 / 641, 240 / 480)
    assert sx != sy, "si estos coincidieran, el test no estaría probando nada"


def test_a_thin_image_does_not_collapse_to_zero_rows():
    """A 0-pixel image is not a smaller image, it is a corrupt one."""
    assert target_size(1000, 3, to_width=10)[1] == 1


# --------------------------------------------------------------------------- #
# The composition: a derived source is a source
# --------------------------------------------------------------------------- #


def test_a_derived_source_reads_with_the_same_parser(tmp_path):
    """The point of reusing the format: `SourceDataset` cannot tell the difference.

    If this needed a second loader, or a flag, or a `if derived:` anywhere, the
    derivation would have created a new format while claiming not to.
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=3, width=120, height=100)
    dst = tmp_path / "derived" / "tiny-w60"
    resize_source(src, dst, ResizeRequest(name="tiny-w60", to_width=60))

    ds = SourceDataset(dst)
    samples = ds.samples()
    assert len(samples) == 3
    for s in samples:
        assert (s.width, s.height) == (60, 50)
        with Image.open(s.image_path) as img:
            assert img.size == (60, 50)


def test_the_geometry_follows_the_pixels(tmp_path):
    """**The seam.** The quad must land where the corner actually went.

    Not "the quad was multiplied by something" -- that a unit test of
    `scale_quad` would cover, and it could not fail. This reads the corner out of
    the resized IMAGE and checks the LABEL agrees with it, which is the only
    version of the question that can catch a wrong factor, a swapped axis, or a
    resize applied to one and not the other.
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=1, width=120, height=100)
    dst = tmp_path / "derived" / "half"
    resize_source(src, dst, ResizeRequest(name="half", to_width=60))

    sample = SourceDataset(dst).samples()[0]
    quad = sample.blocks[0].quad

    # The paragraph is a black rectangle on white: find its extent in the pixels.
    pixels = np.asarray(Image.open(sample.image_path).convert("L"))
    ys, xs = np.nonzero(pixels < 128)
    tl_x, tl_y = quad[0]
    br_x, br_y = quad[2]

    # One pixel of tolerance: LANCZOS softens the edge, so the darkest run is not
    # exactly the geometric rectangle. Anything larger than that would mean the
    # label and the image disagree about where the paragraph is.
    assert abs(tl_x - xs.min()) <= 1 and abs(tl_y - ys.min()) <= 1
    assert abs(br_x - xs.max()) <= 1.5 and abs(br_y - ys.max()) <= 1.5


def test_every_pixel_field_is_rescaled_not_only_the_ones_we_read(tmp_path):
    """`box` is geometry we never consume, and leaving it stale is a silent lie.

    The record would load, `quad` would be at the new resolution and `box` at the
    old one, and the only symptom would be a downstream tool drawing rectangles
    in the wrong place. A resize is all-or-nothing (formatos.md §4.6).
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=1, width=120, height=100)
    dst = tmp_path / "derived" / "half"
    resize_source(src, dst, ResizeRequest(name="half", to_width=60))

    before = json.loads((src / "labels.jsonl").read_text(encoding="utf-8").splitlines()[0])
    after = json.loads((dst / "labels.jsonl").read_text(encoding="utf-8").splitlines()[0])

    box_before = before["labels"]["blocks"][0]["box"]
    box_after = after["labels"]["blocks"][0]["box"]
    assert box_after == pytest.approx([v * 0.5 for v in box_before])

    # And the fields that are NOT pixels ride through untouched.
    assert after["labels"]["blocks"][0]["text"] == before["labels"]["blocks"][0]["text"]
    assert after["labels"]["image_id"] == before["labels"]["image_id"]


def test_nested_geometry_is_rescaled_at_any_depth(tmp_path):
    """`blocks[].lines[].words[].box` exists in the real format and not in the tiny one.

    A version of the walk that only looked at `labels.blocks` would pass every
    other test in this file and be wrong on `mixed-layout` -- right by accident,
    which is worse than wrong.
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=1, width=120, height=100)
    records = [json.loads(l) for l in (src / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    records[0]["labels"]["blocks"][0]["lines"] = [
        {"box": [10.0, 20.0, 30.0, 8.0], "words": [{"box": [10.0, 20.0, 12.0, 8.0]}]}
    ]
    (src / "labels.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    dst = tmp_path / "derived" / "half"
    resize_source(src, dst, ResizeRequest(name="half", to_width=60))

    line = json.loads((dst / "labels.jsonl").read_text(encoding="utf-8"))["labels"]["blocks"][0]["lines"][0]
    assert line["box"] == pytest.approx([5.0, 10.0, 15.0, 4.0])
    assert line["words"][0]["box"] == pytest.approx([5.0, 10.0, 6.0, 4.0])


def test_the_derived_source_records_its_parent_and_real_scale(tmp_path):
    """Without this, two B at different resolutions have provenance that reads identical.

    The directory name is not a datum (organizacion.md ⑧).
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=2, width=120, height=100)
    dst = tmp_path / "derived" / "tiny-w60"
    derived = resize_source(src, dst, ResizeRequest(name="tiny-w60", to_width=60))

    assert derived["op"] == "resize"
    assert derived["request"] == {"width": 60}
    assert derived["size"] == [60, 50]
    assert derived["scale"] == [0.5, 0.5]

    meta = json.loads((dst / "dataset.json").read_text(encoding="utf-8"))
    assert meta["derived"]["from"] == derived["from"]
    assert meta["id"] == "tiny-w60"


def test_the_parent_is_recorded_by_its_addressable_id_not_its_declared_one(tmp_path):
    """Measured on the real data, and it is the 14.5x trap wearing a new hat.

    `clear-paragraphs-02-reducidos` and `clear-paragraphs-02-8ea1ac04` declare
    the SAME `id` in their own `dataset.json` -- the reduced one kept the big
    one's. So a `from` built from the declared id names the wrong parent, in
    silence, in exactly the case organizacion.md §3 says is dangerous.
    """
    src = write_tiny_source(tmp_path / "reducido", num_samples=1, width=120, height=100)
    (src / "dataset.json").write_text(json.dumps({"id": "el-grande"}), encoding="utf-8")

    derived = resize_source(
        src,
        tmp_path / "out",
        ResizeRequest(name="out", to_width=60),
        source_id="reducido",
    )
    assert derived["from"] == "reducido"
    assert derived["from_declared_id"] == "el-grande"


def test_an_original_source_has_no_derived_block(tmp_path):
    """Absent means ORIGINAL, and that is why the absence is legal (formatos.md §2)."""
    src = write_tiny_source(tmp_path / "orig", num_samples=1)
    assert SourceDataset(src).meta.get("derived") is None


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #


def test_upscaling_is_refused(tmp_path):
    """Ampliar interpola: the dataset would measure LANCZOS, not the model."""
    src = write_tiny_source(tmp_path / "orig", num_samples=2, width=120, height=100)
    problems = check_resize(source_sizes(src), ResizeRequest(name="big", to_width=240))
    assert [p["code"] for p in problems] == ["upscale_not_allowed"]
    assert problems[0]["hint"]


def test_a_mixed_size_source_is_refused_whole_if_any_sample_would_grow(tmp_path):
    """Checked against every sample, not the first.

    Accepting this would leave ONE interpolated sample inside an otherwise
    legitimate dataset -- the kind of thing nobody ever finds (api.md §3).
    """
    sizes = [(400, 300), (400, 300), (100, 75)]  # the last one is smaller than the target
    problems = check_resize(sizes, ResizeRequest(name="x", to_width=200))
    assert [p["code"] for p in problems] == ["upscale_not_allowed"]
    assert "1 de 3" in problems[0]["message"]


def test_both_or_neither_dimension_is_refused():
    assert check_resize([(10, 10)], ResizeRequest(name="x"))[0]["code"] == "resize_needs_one_dimension"
    both = ResizeRequest(name="x", to_width=5, to_height=5)
    assert check_resize([(10, 10)], both)[0]["code"] == "resize_needs_one_dimension"


def test_an_existing_source_is_never_overwritten(tmp_path):
    """Trap nº5 of organizacion.md §3, with another subject."""
    src = write_tiny_source(tmp_path / "orig", num_samples=1, width=120, height=100)
    dst = tmp_path / "derived" / "taken"
    resize_source(src, dst, ResizeRequest(name="taken", to_width=60))

    with pytest.raises(ResizeRefused) as exc:
        resize_source(src, dst, ResizeRequest(name="taken", to_width=30))
    assert exc.value.problems[0]["code"] == "source_exists"


def test_nothing_partial_survives_a_cancellation(tmp_path):
    """A half-written source parses fine and silently has fewer samples than its parent.

    That is worse than no source at all, so cancelling leaves nothing behind
    rather than something plausible.
    """
    src = write_tiny_source(tmp_path / "orig", num_samples=5, width=120, height=100)
    dst = tmp_path / "derived" / "cancelled"

    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(KeyboardInterrupt):
        resize_source(src, dst, ResizeRequest(name="cancelled", to_width=60), should_stop=stop_after_two)
    assert not dst.exists()


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #


def _await_job(client, job: dict) -> dict:
    for _ in range(200):
        current = client.get(f"/jobs/{job['id']}").json()
        if current["state"] in ("done", "error", "cancelled", "interrupted"):
            return current
        time.sleep(0.05)
    raise AssertionError(f"el job no terminó: {job}")


def test_post_resize_produces_a_listable_derived_source(itf_api):
    """End to end: the derived source appears in `GET /sources`, prefixed."""
    client, layout = itf_api
    layout.write_source("tiny", num_samples=2, width=120, height=100)

    job = client.post("/sources/tiny/resize", json={"name": "tiny-w60", "width": 60})
    assert job.status_code == 202, job.text
    assert _await_job(client, job.json())["state"] == "done"

    listing = client.get("/sources").json()
    ids = {s["id"]: s for s in listing["sources"]}
    assert "tiny" in ids and "derived/tiny-w60" in ids
    assert ids["derived/tiny-w60"]["derived"]["from"] is not None
    assert ids["tiny"]["derived"] is None


def test_the_derived_source_is_addressable_like_any_other(itf_api):
    """A derived id works everywhere a source id works, or the reuse was a lie."""
    client, layout = itf_api
    layout.write_source("tiny", num_samples=2, width=120, height=100)
    job = client.post("/sources/tiny/resize", json={"name": "tiny-w60", "width": 60})
    _await_job(client, job.json())

    geometry = client.get("/sources/derived/tiny-w60/samples/0/geometry")
    assert geometry.status_code == 200
    assert (geometry.json()["width"], geometry.json()["height"]) == (60, 50)


def test_post_resize_refuses_upscaling_with_400_and_writes_nothing(itf_api):
    """The refusal lands BEFORE the job exists -- no job, no directory."""
    client, layout = itf_api
    layout.write_source("tiny", num_samples=2, width=120, height=100)

    response = client.post("/sources/tiny/resize", json={"name": "big", "width": 400})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "upscale_not_allowed"
    assert response.json()["detail"]["hint"]
    assert not (layout.derived_sources / "big").exists()
    assert client.get("/jobs").json()["jobs"] == []


def test_post_resize_refuses_both_dimensions(itf_api):
    client, layout = itf_api
    layout.write_source("tiny", num_samples=1, width=120, height=100)
    response = client.post("/sources/tiny/resize", json={"name": "x", "width": 60, "height": 50})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "resize_needs_one_dimension"


def test_post_resize_never_writes_into_the_read_only_root(itf_api):
    """A is external and read-only. The parent's root must be untouched (D19)."""
    client, layout = itf_api
    layout.write_source("tiny", num_samples=2, width=120, height=100)
    before = sorted(p.name for p in layout.datasets.iterdir())

    job = client.post("/sources/tiny/resize", json={"name": "tiny-w60", "width": 60})
    _await_job(client, job.json())

    assert sorted(p.name for p in layout.datasets.iterdir()) == before
    assert (layout.derived_sources / "tiny-w60" / "labels.jsonl").exists()
