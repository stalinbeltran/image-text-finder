"""F — inference: applying a trained E to a patch or to a whole image.

Complete since fase 6. What is here, and which question each one answers:

| | |
|---|---|
| `load_model` | contract ④: a `.pt` describes itself, so F never needs C's YAML |
| `ModelCache` | the same network, loaded once. Keyed on **mtime**: `best.pt` moves |
| `kernels`, `feature_maps` | V1 and V2 — **input = a patch** (contract ①) |
| `predict_image` | the three stages over a whole image (V11), `raw` included |

May import `models` and `geometry`. **Not `patches`** (tests.md §4): B and F share
a window, and sharing it means both taking it from G -- not F reaching into B,
which is what the old `from patches.extract import _positions` did, importing a
private name across a domain border to hold up contract ⑤.
"""

from itf.inference.cache import MAX_MODELS, ModelCache
from itf.inference.checkpoint import load_model
from itf.inference.introspect import NotInspectable, feature_maps, kernels
from itf.inference.predict import (
    DEFAULT_MIN_SIZE,
    DEFAULT_THRESHOLD,
    Detection,
    default_stride,
    detect_corners,
    merge,
    predict_image,
    reconstruct_boxes,
)

__all__ = [
    "DEFAULT_MIN_SIZE",
    "DEFAULT_THRESHOLD",
    "Detection",
    "MAX_MODELS",
    "ModelCache",
    "NotInspectable",
    "default_stride",
    "detect_corners",
    "feature_maps",
    "kernels",
    "load_model",
    "merge",
    "predict_image",
    "reconstruct_boxes",
]
