"""Recommender implementations and the name -> class registry."""

from __future__ import annotations

from .als import ALS
from .bpr import BPR
from .itemknn import ItemKNN
from .popularity import Popularity
from .random_model import RandomModel

REGISTRY: dict[str, type] = {
    RandomModel.name: RandomModel,
    Popularity.name: Popularity,
    ItemKNN.name: ItemKNN,
    ALS.name: ALS,
    BPR.name: BPR,
}

# TwoTower needs torch (optional extra). It is left out of the registry when torch is missing;
# the reason is kept so a run that asks for it can say why instead of dropping it silently.
UNAVAILABLE: dict[str, str] = {}
try:
    from .twotower import TwoTower
except ImportError as exc:
    UNAVAILABLE["twotower"] = f"{type(exc).__name__}: {exc}"
else:
    REGISTRY[TwoTower.name] = TwoTower

__all__ = ["ALS", "BPR", "REGISTRY", "UNAVAILABLE", "ItemKNN", "Popularity", "RandomModel"]
