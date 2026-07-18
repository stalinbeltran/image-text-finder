"""Pixels: proportional resize, and the scale it actually achieved.

**Imports nothing from `itf`, and must not** (tests.md §4). It does not know what
a quad is, what a paragraph is, or that a CNN exists -- it takes an image and one
dimension. That boundary is the whole point of the module rather than a tidiness
preference: it is what lets the same function resize an arbitrary test image
later, with no dataset anywhere near it (D19).

**Why it returns the scale.** The caller needs to move geometry along with the
pixels, and the honest factor is not the one that was requested: asking for
`width=320` on a 641x480 image gives a height of `round(480 * 320 / 641) = 240`,
so x scaled by `320/641 = 0.4992` and y by `240/480 = 0.5000`. Close, and that is
exactly the danger -- reusing the requested factor for both axes puts every quad
a fraction of a pixel off, in a way no test notices and no eye catches. So the
scale is **measured from the output**, and there are **two** of them.

Library-shaped on purpose but **not extracted**: it exists in one project, and
librerias.md §0 extracts on the second sighting, not the first.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Resized:
    """An image and the scale that was actually applied to it."""

    image: Image.Image
    #: Output size, ``(width, height)``.
    size: tuple[int, int]
    #: ``(sx, sy)`` measured as out/in -- **not** the requested factor. See module docstring.
    scale: tuple[float, float]


def target_size(
    width: int,
    height: int,
    *,
    to_width: int | None = None,
    to_height: int | None = None,
) -> tuple[int, int]:
    """The output size for ``width x height``, given exactly one target dimension.

    Pure arithmetic, no image: this is what lets a caller reject a whole batch
    before touching a single file. `POST /sources/{id}/resize` needs precisely
    that -- it checks every sample for upscaling *before* writing anything, so a
    mixed-size source with one image that would grow is refused as a whole rather
    than leaving one interpolated sample inside an otherwise legitimate dataset
    (api.md §3).

    One dimension, never two: the aspect ratio is preserved by construction, so
    passing both would be asking for a distortion this module does not do.
    """
    if (to_width is None) == (to_height is None):
        raise ValueError("pide exactamente una dimensión: o `to_width` o `to_height`, no ambas")
    if width <= 0 or height <= 0:
        raise ValueError(f"tamaño de origen inválido: {width}x{height}")

    if to_width is not None:
        if to_width <= 0:
            raise ValueError(f"`to_width` debe ser positivo; me llegó {to_width}")
        # max(1, ...) so a very wide, very short image cannot collapse to zero
        # rows. A 0-pixel image is not a smaller image: it is a corrupt one.
        return to_width, max(1, round(height * to_width / width))

    if to_height <= 0:
        raise ValueError(f"`to_height` debe ser positivo; me llegó {to_height}")
    return max(1, round(width * to_height / height)), to_height


def resize(
    image: Image.Image,
    *,
    to_width: int | None = None,
    to_height: int | None = None,
    resample: int = Image.LANCZOS,
) -> Resized:
    """Resize keeping the aspect ratio, and report the scale actually achieved.

    `resample` is the caller's call because it is the caller who knows what the
    numbers mean. LANCZOS for a photograph or a render; **NEAREST for a label
    mask**, where interpolating fabricates classes that do not exist -- the
    continuous version of `ausente != cero` (formatos.md §2). This module cannot
    know which it has, so it does not guess; it is the same reasoning that keeps
    `matrixview` from choosing its own colours (librerias.md §1).
    """
    out_w, out_h = target_size(
        image.width, image.height, to_width=to_width, to_height=to_height
    )
    resized = image.resize((out_w, out_h), resample)
    return Resized(
        image=resized,
        size=(out_w, out_h),
        scale=(out_w / image.width, out_h / image.height),
    )
