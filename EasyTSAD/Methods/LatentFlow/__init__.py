"""LatentFlow method package (EasyTSAD adapter).

``LatentFlow`` is registered with ``BaseMethodMeta`` through this export; the
implementation lives in ``LatentFlow.py`` and delegates the algorithm to the
untouched official source tree in ``upstream/``.
"""

from .LatentFlow import LatentFlow

__all__ = ["LatentFlow"]
