"""
Optimization for planar cubic graphs using planarity-preserving Whitehead flips.

This module adapts the standard optimization methods to work exclusively with
planar cubic graphs, ensuring all Whitehead flips preserve planarity.
"""

import networkx as nx
import random
from typing import Callable, Tuple, Optional
from ..core.spanning_trees import count_spanning_trees, get_second_eigenvalue
from .planar_ops import find_planar_whitehead_flips, perform_planar_flip, is_planar_cubic


def gradient_ascent_planar(
    G: nx.Graph,
    objective_func: Callable[[nx.Graph], float],
    maximize: bool = True,
    max_iterations: int = 100,
    max_attempts_per_iteration: int = 100,
    verbose: bool = False,
) -> Tuple[nx.Graph, float, dict]:
    """
    Gradient ascent/descent optimization using only planarity-preserving Whitehead flips.

    This is similar to the standard first-improvement method, but restricts to flips
    that preserve planarity.

    Args:
        G: Initial planar cubic graph
        objective_func: Function to optimize (takes graph, returns float)
        maximize: If True, maximize objective; if False, minimize
        max_iterations: Maximum number of improvement iterations
        max_attempts_per_iteration: Max attempts to find improving flip per iteration
        verbose: Print progress

    Returns:
        (optimized_graph, final_value, stats_dict)
    """
    if not is_planar_cubic(G):
        raise ValueError("Input graph must be planar and cubic!")

    G = G.copy()
    current_value = objective_func(G)
    stats = {
        'initial_value': current_value,
        'iterations': 0,
        'flips_performed': 0,
        'planar_flips_tested': 0,
    }

    if verbose:
        print(f"Initial value: {current_value:.6f}")

    for iteration in range(max_iterations):
        improved = False

        # Find all valid planar flips
        valid_flips = find_planar_whitehead_flips(G, max_flips=max_attempts_per_iteration)
        stats['planar_flips_tested'] += len(valid_flips)

        if not valid_flips:
            if verbose:
                print(f"Iteration {iteration}: No valid planar flips available")
            break

        # Shuffle to randomize
        random.shuffle(valid_flips)

        # Try flips until we find an improvement
        for e1, e2 in valid_flips:
            # Perform flip
            G_test = G.copy()
            perform_planar_flip(G_test, e1, e2)

            # Evaluate
            new_value = objective_func(G_test)

            # Check if improved
            is_better = (new_value > current_value) if maximize else (new_value < current_value)

            if is_better:
                # Accept the flip
                G = G_test
                current_value = new_value
                stats['flips_performed'] += 1
                improved = True

                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: {current_value:.6f}")

                break

        stats['iterations'] += 1

        if not improved:
            # No improving flip found
            if verbose:
                print(f"Local optimum reached at iteration {iteration}")
            break

    stats['final_value'] = current_value

    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Final value: {current_value:.6f}")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Flips performed: {stats['flips_performed']}")
        print(f"  Planar flips tested: {stats['planar_flips_tested']}")

    return G, current_value, stats


def optimize_planar_cubic_graph(
    n: int,
    objective: str = 'spanning_trees',
    restarts: int = 10,
    max_iterations: int = 100,
    num_randomization_flips: int = 50,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, float, dict]:
    """
    Optimize a planar cubic graph with multiple random restarts.

    Args:
        n: Number of vertices
        objective: 'spanning_trees' or 'expansion'
        restarts: Number of random restarts
        max_iterations: Max iterations per restart
        num_randomization_flips: Flips to randomize starting graph
        verbose: Print progress
        seed: Random seed

    Returns:
        (best_graph, best_value, stats)
    """
    if seed is not None:
        random.seed(seed)

    # Set up objective function
    if objective == 'spanning_trees':
        objective_func = count_spanning_trees
        maximize = True
        obj_name = "ln(spanning trees)"
    elif objective == 'expansion':
        objective_func = get_second_eigenvalue
        maximize = False
        obj_name = "λ₂"
    else:
        raise ValueError(f"Unknown objective: {objective}")

    best_graph = None
    best_value = float('-inf') if maximize else float('inf')
    all_stats = []

    if verbose:
        print(f"Optimizing planar cubic graph (N={n})")
        print(f"Objective: {obj_name}")
        print(f"Restarts: {restarts}\n")

    from .planar_ops import random_planar_cubic_from_sphere

    for restart in range(restarts):
        if verbose:
            print(f"--- Restart {restart + 1}/{restarts} ---")

        # Generate random starting graph using sphere method
        # Note: n points → approximately 2n-4 vertices, so we solve for n
        # For target N vertices: n ≈ (N+4)/2
        n_points = (n + 4) // 2

        try:
            G_start = random_planar_cubic_from_sphere(
                n_points,
                seed=None  # Use current random state
            )

            # Verify we got approximately the right size
            actual_n = G_start.number_of_nodes()
            if abs(actual_n - n) > n * 0.3:  # Allow 30% deviation
                if verbose:
                    print(f"  Warning: requested N={n}, got N={actual_n}")

        except (NotImplementedError, ValueError) as e:
            print(f"Cannot generate planar cubic graph for N={n}: {e}")
            return None, None, {}

        # Optimize
        G_opt, value, stats = gradient_ascent_planar(
            G_start,
            objective_func,
            maximize=maximize,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        all_stats.append(stats)

        # Check if this is the best so far
        is_better = (value > best_value) if maximize else (value < best_value)
        if is_better:
            best_graph = G_opt
            best_value = value

            if verbose:
                print(f"  ⭐ New best: {best_value:.6f}\n")

    # Summary stats
    summary = {
        'best_value': best_value,
        'restarts': restarts,
        'all_restart_stats': all_stats,
        'objective': objective,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best result: {obj_name} = {best_value:.6f}")
        print(f"{'='*60}")

    return best_graph, best_value, summary
