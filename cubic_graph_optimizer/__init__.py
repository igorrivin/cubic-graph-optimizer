"""Cubic Graph Optimizer package for maximizing spanning trees in cubic graphs."""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for convenience
from .core.spanning_trees import count_spanning_trees
from .optimization.methods import (
    gradient_ascent_greedy,
    gradient_ascent_first_improvement,
    simulated_annealing,
    optimize_with_restarts,
    run_optimization_sweep
)
from .analysis.properties import analyze_graph_properties, compare_graph_structures
from .utils.io import save_graph, load_graph

__all__ = [
    'count_spanning_trees',
    'gradient_ascent_greedy',
    'gradient_ascent_first_improvement',
    'simulated_annealing',
    'optimize_with_restarts',
    'run_optimization_sweep',
    'analyze_graph_properties',
    'compare_graph_structures',
    'save_graph',
    'load_graph'
]