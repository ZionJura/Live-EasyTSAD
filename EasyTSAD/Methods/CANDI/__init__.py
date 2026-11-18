"""EasyTSAD method package for CANDI.

``EasyTSAD/Harness`` imports ``EasyTSAD.Methods.CANDI.CANDI`` and looks up
``CANDI`` in the ``BaseMethodMeta`` registry, so the adapter class must be
re-exported from the package namespace.
"""

from .CANDI import CANDI

__all__ = ["CANDI"]
