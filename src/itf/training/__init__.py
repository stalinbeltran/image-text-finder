"""D — Receta, and E — Run.

Fase 2 only needs the reading half of E (the registry), because deleting a B has
to know which runs reference it. The recipe store, the loss and the loop arrive
with fases 3 and 4.
"""

from itf.training.registry import RunStore

__all__ = ["RunStore"]
