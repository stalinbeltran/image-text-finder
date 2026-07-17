"""F — inference: applying a trained E to a whole image.

Fase 4 brings only `load_model`, which is contract ④ and is what everything else
in F will stand on. The sliding window and the paragraph reconstruction are fase
6 (`predict.py`), and when they arrive **the window must come from `itf.geometry`**
-- contract ⑤ is about exactly this module, and the natural thing to do while
writing inference is to re-type those six lines instead of importing them.

May import `models` and `geometry`. **Not `patches`** (tests.md §4): B and F share
a window, and sharing it means both taking it from G, not F reaching into B --
which is what the old `from patches.extract import _positions` did, importing a
private name across a domain border.
"""

from itf.inference.checkpoint import load_model

__all__ = ["load_model"]
