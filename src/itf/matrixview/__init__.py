"""matrixview — a matrix of numbers, serialised so a browser can paint it.

**A library that has not been extracted yet, and is built to be one**
([librerias.md](../../../docs/librerias.md) §5 puts the extraction in fase 6, next
to V1 and V2, its first two consumers). The rule it obeys meanwhile is the
condition that makes extracting cheap: **it imports nothing from `itf`**. It takes
arrays and returns dicts; if writing it ever required naming `TL`, a patch or a
run, the boundary would be in the wrong place (librerias.md §0).

*Deferred on purpose, 2026-07-17: `claude-libs/` does not exist yet -- fases 3 and
4 did not extract `convspec` or `exp-registry` either -- and pulling the monorepo
into being also means closing D9 (names) and D10 (layout). Isolated here, the
extraction stays a `git mv` plus a `pyproject.toml`.*

**The code is 30 lines; the value is the decisions inside it.** That is not a
figure of speech -- the sibling project re-derived these from scratch and **got
one wrong**:

- **Numbers travel, not images.** The server sends `matrix` + `min`/`max`/`mean`;
  the client decides the colour. It is also what makes R5's number table possible
  at zero extra cost: the table and the map read the same payload.
- **Normalisation is per map**, against its own min/max, never global. Without it
  the quiet feature maps all render black -- and "the deep layers are dead" is a
  conclusion someone would then draw about the network.
- **Truncation warns** (`max_maps`, `truncated: true`). A layer of 128 filters
  serialised whole is enough JSON to hang the tab; dropping them silently would
  quietly answer "how many filters fired?" with a lie.
- **The painter does not choose the colour: it is told.** This is the one the
  sibling got wrong, and it is why `job` is a required argument with no default.
  It painted kernels -- which have sign -- on a `min→max` ramp, so zero landed
  wherever it fell and the sign structure (what excites, what inhibits: *what a
  kernel is*) was invisible. Signed data is **diverging, centred on 0**; magnitude
  is **sequential**. This module cannot know which: only the caller knows whether
  the array is a weight or a ReLU'd activation, so the caller says (ui.md §4.0
  R2/R3, api.md §3).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

#: What the colour has to do. Declared by whoever knows the data, never guessed.
ColorJob = Literal["sequential", "diverging"]

#: Maps per layer beyond which the payload is truncated. 64 is the sibling's, and
#: it matters more here: a backbone block of 128 filters over a 40×40 patch is
#: 128 × 1600 rounded floats in one response.
MAX_MAPS_PER_LAYER = 64

#: Decimals kept in the JSON. Enough to read a weight, short enough that a 40×40
#: map is not mostly float noise: `0.0123456789` and `0.0123` paint identically
#: and the second is a third of the bytes.
DIGITS = 4


def map_payload(array: np.ndarray, index: int, *, digits: int = DIGITS) -> dict[str, Any]:
    """One matrix: its numbers, plus the stats the client normalises against.

    `min`/`max` ride along with **each** map because the normalisation is per map.
    Compute them client-side from the whole layer instead and you have silently
    made it global, which is the failure mode described above.
    """
    values = np.asarray(array, dtype=np.float64)
    return {
        "index": int(index),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "matrix": np.round(values, digits).tolist(),
    }


def layer_payload(
    *,
    layer: int,
    maps: np.ndarray,
    job: ColorJob,
    max_maps: int = MAX_MAPS_PER_LAYER,
    **extra: Any,
) -> dict[str, Any]:
    """A layer's worth of matrices: `(K, H, W)` → the payload V1 and V2 read.

    One shape for both views, which is why the front renders them with one
    component: a layer of kernels and a layer of feature maps differ in what the
    numbers mean, not in what they are.

    `count` vs `shown` is the truncation, said out loud: `count` is how many the
    layer really has, `shown` is how many are in this response. A reader that only
    got `shown` would have no way to know it was looking at part of a layer.
    """
    stack = np.asarray(maps)
    if stack.ndim != 3:
        raise ValueError(f"un layer_payload es (K, H, W); me llegó {stack.shape}")

    count = int(stack.shape[0])
    shown = min(count, max_maps)
    return {
        "layer": int(layer),
        "count": count,
        "shown": shown,
        "truncated": shown < count,
        # The colour work, declared. The client cannot know whether this is a
        # signed weight or a non-negative activation, and guessing is the bug.
        "job": job,
        "maps": [map_payload(stack[i], i) for i in range(shown)],
        **extra,
    }
