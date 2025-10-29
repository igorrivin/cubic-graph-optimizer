"""
Planar cubic graph optimization.

This module extends the cubic graph optimizer to work specifically with
planar cubic graphs, using only planarity-preserving Whitehead flips.
"""

from .planar_ops import (
    is_planar_cubic,
    check_flip_preserves_planarity,
    find_planar_whitehead_flips,
    perform_planar_flip,
    random_planar_cubic_graph,
    random_planar_cubic_from_sphere,
)

__all__ = [
    'is_planar_cubic',
    'check_flip_preserves_planarity',
    'find_planar_whitehead_flips',
    'perform_planar_flip',
    'random_planar_cubic_graph',
    'random_planar_cubic_from_sphere',
]
