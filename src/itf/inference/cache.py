"""The model cache. `(checkpoint, device, mtime)` → a loaded network.

**It lived in `api/app.py` and it is not the API's** (api.md §0): nothing about
caching a torch module mentions HTTP. The mechanical rule that moved it is the
same one that moved `_discover_datasets` and `_run_status` -- and the proof it
belongs here rather than there is that `predict_image` from a script wants it
just as much as a route does.

**Why it exists**: V2 and V5 repaint live. A patch's feature maps are one forward
pass -- microseconds -- but `load_model` reads a file and rebuilds a network, and
doing that on every slider tick is what makes a live view feel broken.

**Why `mtime` is in the key, and this is the load-bearing part**: a run is not
immutable while it trains. `best.pt` is rewritten on every improving epoch, so a
cache keyed on the path alone serves epoch 5's weights forever -- you would watch
a run improve and its kernels would never change. That is the same reasoning that
put `checkpoint_mtime_ns` in `TableKey`, and the old code's model cache already
keyed on mtime for exactly this (organizacion.md §2-④): it is a lesson the
project keeps re-learning, so it is written down in both places.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock

from itf.inference.checkpoint import load_model
from itf.models import ConfigurableCNN

#: Networks kept before the least recently used is dropped. Small on purpose: a
#: CNN of this size is a few MB, the cache exists to serve a live view of ONE run,
#: and an unbounded dict of every model ever opened is a leak with a good excuse.
MAX_MODELS = 4


class ModelCache:
    """LRU of loaded, `eval()`-mode networks. Thread-safe.

    The lock is not decoration: `JobQueue` runs training in a worker thread while
    the API serves reads on another, and `OrderedDict.move_to_end` during a
    concurrent `popitem` is a real race. It guards the bookkeeping only -- two
    threads racing on a cold key both load, and the second wins the slot. That is
    a wasted load, not a wrong answer, and it is worth far more than holding a
    lock across a file read.
    """

    def __init__(self, max_models: int = MAX_MODELS):
        self._models: OrderedDict[tuple, ConfigurableCNN] = OrderedDict()
        self._lock = Lock()
        self._max = max_models

    def get(self, checkpoint: str | Path, device: str = "cpu") -> ConfigurableCNN:
        """The network that checkpoint holds, loaded once per (path, device, mtime)."""
        path = Path(checkpoint)
        if not path.exists():
            # Before stat(), so the error is `load_model`'s -- which says what a
            # missing checkpoint means -- and not a bare OSError from the key.
            return load_model(path, device=device)
        key = (str(path.resolve()), device, path.stat().st_mtime_ns)

        with self._lock:
            hit = self._models.get(key)
            if hit is not None:
                self._models.move_to_end(key)
                return hit

        model = load_model(path, device=device)

        with self._lock:
            self._models[key] = model
            self._models.move_to_end(key)
            while len(self._models) > self._max:
                self._models.popitem(last=False)
        return model

    def clear(self) -> None:
        """Drop everything. For rename/delete, and for tests.

        The old cache needed a `_drop_model_cache` for renames because it keyed on
        the run's NAME. This one keys on the resolved path and mtime, so a rename
        misses on its own and a rebuilt checkpoint invalidates itself -- there is
        nothing to remember to call. That is the difference between a cache that
        is correct and one that is correct as long as everyone remembers.
        """
        with self._lock:
            self._models.clear()
