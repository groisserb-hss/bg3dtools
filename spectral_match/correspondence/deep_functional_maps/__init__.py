"""
Deep functional maps for shape correspondence.

This package provides neural network operations and training utilities
for deep functional map learning.
"""

from .operations import (
    solve_lstsq, correspondence_matrix, soft_correspondence_ensemble,
    geodesic_error_ensemble, ResidualLayer, ResidualNet,
)
from .training import EnsembleTrainer, cache_and_train

__all__ = [
    "solve_lstsq", "correspondence_matrix", "soft_correspondence_ensemble",
    "geodesic_error_ensemble", "ResidualLayer", "ResidualNet",
    "EnsembleTrainer", "cache_and_train",
]
