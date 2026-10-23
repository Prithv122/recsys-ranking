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

# TwoTower needs torch (optional extra). A missing module and a missing torch both
# surface as ImportError, and either way the model is simply not registered.
try:
    from .twotower import TwoTower
except ImportError:
    pass
else:
    REGISTRY[TwoTower.name] = TwoTower

__all__ = ["ALS", "BPR", "REGISTRY", "ItemKNN", "Popularity", "RandomModel"]
