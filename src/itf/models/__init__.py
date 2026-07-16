"""C — Red: the architecture. Config pure, zero data.

Imports only `itf.geometry` (the shared vocabulary), and that is contract ⑦'s
fix rather than an exception to it: what ⑦ forbids is C importing from the SOURCE
LOADER to learn there are 4 borders. See tests.md §4.
"""

from itf.models.builder import (
    ConfigurableCNN,
    NetworkConfig,
    build_model,
    count_params,
    flat_features,
    spatial_trace,
)
from itf.models.heads import CornerHead
from itf.models.store import NetworkNotFound, NetworkStore

__all__ = [
    "ConfigurableCNN",
    "CornerHead",
    "NetworkConfig",
    "NetworkNotFound",
    "NetworkStore",
    "build_model",
    "count_params",
    "flat_features",
    "spatial_trace",
]
