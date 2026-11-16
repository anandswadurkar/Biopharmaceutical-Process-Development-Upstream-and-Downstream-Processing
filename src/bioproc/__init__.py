"""Biopharmaceutical process development: upstream, downstream, data-driven models.

The notebook in ``notebooks/`` is the interface; this package holds the logic so it can
be tested. Modules never print, plot or write files - those belong to the notebook
layer (ground rules section 5).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "downstream",
    "evaluation",
    "models",
    "plotting",
    "quality",
    "simulators",
    "ui",
    "upstream",
    "uq",
    "verification",
]
