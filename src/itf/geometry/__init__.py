"""G — the shared vocabulary, and the sliding window.

This module exists because of two contracts, and both were violations of the same
shape: something everyone shares had been left living inside somebody's house.

**⑦ — the vocabulary lived in domain A.** `CORNER_NAMES`, `BORDER_NAMES` and
their counts were declared in `datasets/loader.py`, so `models/builder.py` (the
network!) imported from the source-dataset loader just to learn that there are 4
borders. Nobody decided that: the constant was there and the import worked.

**⑤ — the window was duplicated between B and F.** Extraction and inference must
see the *identical* window or the model meets a geometry it never trained on.
That was held together by two seams: `inference/predict.py` imported the
**private** `_positions` from `patches/extract.py`, and the border flags were
duplicated *literally* in both files. Identical, so it worked -- and nothing
would have noticed if someone touched one. The failure would not have been an
exception: just slightly worse predictions near the image edges.

So: B and F both import the window FROM HERE. Re-typing those six lines is what
feels natural when writing inference, which is why `test_contract_05` asserts
both sides resolve to this module's function and not to a copy of it.

Imports nothing from `itf`, and must not (tests.md §4). It is the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

# The quad in SAMPLE_FORMAT is clockwise from the top-left, so the corner order
# is fixed project-wide. `y[:, c]` means CORNER_NAMES[c] everywhere, and the
# manifest carries this order because it is semantics, not decoration: lose it
# and the arrays still load and mean something else (formatos.md §4.1).
CORNER_NAMES: tuple[str, ...] = ("TL", "TR", "BR", "BL")
NUM_CORNERS = len(CORNER_NAMES)

# Whether the patch sits flush against the SOURCE IMAGE's edge on each side --
# not the edge of a paragraph. Clockwise from the top, like CSS margins.
#
# This context is destroyed by cropping, and it is real information: a paragraph
# corner cannot sit just outside the image, so a patch flush against the top has
# a different prior than one in the middle.
BORDER_NAMES: tuple[str, ...] = ("top", "right", "bottom", "left")
NUM_BORDERS = len(BORDER_NAMES)


@dataclass(frozen=True)
class Window:
    """One position of the sliding window over one image."""

    x0: int
    y0: int
    #: Flags in BORDER_NAMES order: (top, right, bottom, left).
    border: tuple[int, int, int, int]


def positions(size: int, n: int, stride: int) -> list[int]:
    """Top-left offsets so the window covers ``[0, size)``, including a flush edge.

    The flush edge is the part worth keeping: when ``size - n`` is not a multiple
    of ``stride`` the plain range stops short and the last strip of the image is
    never seen by any window. So the last position is appended even though it
    overlaps its neighbour by more than ``stride``.
    """
    if n > size:
        raise ValueError(f"patch_size ({n}) es mayor que la dimensión de la imagen ({size})")
    last = size - n
    xs = list(range(0, last + 1, stride))
    if xs[-1] != last:
        xs.append(last)
    return xs


def windows(width: int, height: int, n: int, stride: int) -> list[Window]:
    """Every window over a ``width x height`` image, with its border flags.

    **The single definition of the window.** B calls it to extract; F calls it to
    infer. If these two ever disagree the model sees, at inference, a geometry it
    never trained on -- silently.
    """
    return [
        Window(
            x0=x0,
            y0=y0,
            border=(
                int(y0 == 0),
                int(x0 + n >= width),
                int(y0 + n >= height),
                int(x0 == 0),
            ),
        )
        for y0 in positions(height, n, stride)
        for x0 in positions(width, n, stride)
    ]
