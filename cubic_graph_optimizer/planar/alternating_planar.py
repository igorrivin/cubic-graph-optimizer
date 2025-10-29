"""
Alternating optimization for planar cubic graphs.

Alternates between optimizing:
1. Spanning trees (on the triangulation)
2. Expansion (λ₂ of the dual cubic graph)

Since the triangulation and its dual are related by planar duality,
optimizing one affects the other. Alternating can find better solutions
than single-objective optimization.
"""

import networkx as nx
import random
from typing import Tuple, Dict, List
from .triangulation import (
    find_flippable_edges,
    diagonal_flip,
    triangulation_to_dual_cubic,
)
from ..core.spanning_trees import count_spanning_trees, get_second_eigenvalue


def alternating_planar_optimization(
    G_tri_initial: nx.Graph,
    objectives: List[str] = ['spanning_trees', 'expansion'],
    max_cycles: int = 10,
    max_iterations_per_phase: int = 100,
    min_improvement: float = 1e-6,
    verbose: bool = False,
) -> Tuple[nx.Graph, nx.Graph, Dict]:
    """
    Alternating optimization for planar cubic graphs via triangulation flips.

    Alternates between optimizing:
    - Spanning trees (equivalent in triangulation and dual)
    - Expansion (λ₂ of the dual cubic graph)

    Args:
        G_tri_initial: Initial planar triangulation
        objectives: List of objectives to alternate between
        max_cycles: Maximum number of alternating cycles
        max_iterations_per_phase: Max iterations per objective phase
        min_improvement: Minimum improvement to continue
        verbose: Print progress

    Returns:
        (final_triangulation, final_dual_cubic, stats)
    """
    G_tri = G_tri_initial.copy()
    G_dual = triangulation_to_dual_cubic(G_tri)

    # Initialize tracking
    trajectory = []
    current_cycle = 0

    if verbose:
        trees = count_spanning_trees(G_tri)
        lambda2 = get_second_eigenvalue(G_dual)
        print(f"Initial: ln(trees)={trees:.6f}, dual λ₂={lambda2:.6f}")

    for cycle in range(max_cycles):
        cycle_improved = False

        for obj_idx, objective in enumerate(objectives):
            if verbose:
                print(f"\nCycle {cycle+1}, Phase {obj_idx+1}: Optimizing {objective}")

            # Determine what to optimize
            if objective == 'spanning_trees':
                # Optimize spanning trees on triangulation
                # (This also optimizes spanning trees on dual by duality)
                current_value = count_spanning_trees(G_tri)
                maximize = True

                def eval_func(G):
                    return count_spanning_trees(G)

            elif objective == 'expansion':
                # Optimize λ₂ of the DUAL cubic graph
                # Need to compute dual after each flip
                current_value = get_second_eigenvalue(G_dual)
                maximize = False

                def eval_func(G_tri_test):
                    G_dual_test = triangulation_to_dual_cubic(G_tri_test)
                    return get_second_eigenvalue(G_dual_test)

            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Try to improve via diagonal flips
            phase_improved = False
            for iteration in range(max_iterations_per_phase):
                # Find flippable edges
                flippable = find_flippable_edges(G_tri)

                if not flippable:
                    if verbose:
                        print(f"  No flippable edges at iteration {iteration}")
                    break

                # Shuffle for randomization
                random.shuffle(flippable)

                # Try each flip
                improved_this_iter = False
                for edge in flippable:
                    # Test flip
                    G_tri_test = G_tri.copy()
                    diagonal_flip(G_tri_test, edge)

                    new_value = eval_func(G_tri_test)

                    # Check if improved
                    is_better = (new_value > current_value) if maximize else (new_value < current_value)

                    if is_better and abs(new_value - current_value) > min_improvement:
                        # Accept flip
                        G_tri = G_tri_test
                        G_dual = triangulation_to_dual_cubic(G_tri)
                        current_value = new_value
                        phase_improved = True
                        improved_this_iter = True
                        cycle_improved = True

                        if verbose and iteration % 10 == 0:
                            trees = count_spanning_trees(G_tri)
                            lambda2 = get_second_eigenvalue(G_dual)
                            print(f"    Iter {iteration}: trees={trees:.6f}, λ₂={lambda2:.6f}")

                        break

                if not improved_this_iter:
                    # No improving flip found
                    if verbose:
                        print(f"  Local optimum at iteration {iteration}")
                    break

            # Record trajectory
            trees = count_spanning_trees(G_tri)
            lambda2 = get_second_eigenvalue(G_dual)
            trajectory.append({
                'cycle': cycle,
                'phase': obj_idx,
                'objective': objective,
                'trees': trees,
                'lambda2': lambda2,
            })

            if verbose:
                print(f"  After {objective}: trees={trees:.6f}, λ₂={lambda2:.6f}")

        # Check if any improvement this cycle
        if not cycle_improved:
            if verbose:
                print(f"\nConverged at cycle {cycle+1} (no improvement)")
            break

        current_cycle = cycle + 1

    # Final stats
    final_trees = count_spanning_trees(G_tri)
    final_lambda2 = get_second_eigenvalue(G_dual)

    stats = {
        'cycles_completed': current_cycle + 1,
        'final_trees': final_trees,
        'final_lambda2': final_lambda2,
        'trajectory': trajectory,
        'converged': not cycle_improved,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Alternating optimization complete")
        print(f"  Cycles: {stats['cycles_completed']}")
        print(f"  Final ln(trees): {final_trees:.6f}")
        print(f"  Final dual λ₂: {final_lambda2:.6f}")
        print(f"{'='*70}")

    return G_tri, G_dual, stats
