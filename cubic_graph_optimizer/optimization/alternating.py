"""
Alternating optimization between multiple objectives.

This module implements alternating optimization schemes where we optimize
one objective until convergence, then switch to another objective, and repeat.
"""

import networkx as nx
from .methods import gradient_ascent_first_improvement, get_objective_function
from ..core.spanning_trees import count_spanning_trees, get_second_eigenvalue


def alternating_optimization(G, objectives=['spanning_trees', 'expansion'],
                            max_cycles=10, max_iterations_per_phase=100,
                            min_improvement=1e-6, verbose=True):
    """
    Alternate between optimizing different objectives until convergence or cycling.

    The algorithm:
    1. Optimize objective 1 until local optimum
    2. Optimize objective 2 until local optimum
    3. Repeat until:
       - Both objectives stop improving (converged to a fixed point)
       - We return to a previous state (detected a cycle)
       - Maximum cycles reached

    Args:
        G: NetworkX graph to optimize
        objectives: List of objectives to alternate between
                   (e.g., ['spanning_trees', 'expansion'])
        max_cycles: Maximum number of complete cycles through all objectives
        max_iterations_per_phase: Max iterations when optimizing each objective
        min_improvement: Minimum improvement to consider significant
        verbose: Print progress

    Returns:
        dict: {
            'final_graph': optimized graph,
            'trajectory': list of states visited,
            'converged': whether we converged to a fixed point,
            'cycled': whether we detected a cycle,
            'num_cycles': number of complete cycles performed
        }
    """
    if verbose:
        print("=" * 70)
        print("Alternating Optimization")
        print("=" * 70)
        print(f"Objectives: {' ↔ '.join(objectives)}")
        print(f"Max cycles: {max_cycles}")
        print("=" * 70)

    # Track the trajectory: list of (ln_trees, lambda2, graph_hash)
    trajectory = []

    # Current graph
    current_G = G.copy()

    # Track seen states to detect cycles
    seen_states = {}

    # Initial measurements
    ln_trees = count_spanning_trees(current_G)
    lambda2 = get_second_eigenvalue(current_G)

    if verbose:
        print(f"\nInitial state:")
        print(f"  ln(trees) = {ln_trees:.6f}")
        print(f"  λ₂        = {lambda2:.6f}")

    # Store initial state
    state_hash = _hash_graph_state(ln_trees, lambda2)
    trajectory.append({
        'cycle': 0,
        'phase': 'initial',
        'ln_trees': ln_trees,
        'lambda2': lambda2
    })
    seen_states[state_hash] = 0

    converged = False
    cycled = False
    cycle_start = -1

    for cycle in range(1, max_cycles + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Cycle {cycle}")
            print(f"{'='*70}")

        cycle_start_trees = ln_trees
        cycle_start_lambda2 = lambda2

        # Alternate through all objectives
        for phase_idx, objective in enumerate(objectives):
            obj_func, maximize, obj_name = get_objective_function(objective)

            if verbose:
                print(f"\nPhase {phase_idx + 1}: Optimizing {obj_name}")
                print("-" * 70)

            # Optimize this objective
            current_G, final_value = gradient_ascent_first_improvement(
                current_G,
                max_iterations=max_iterations_per_phase,
                verbose=False,  # Don't print detailed iteration info
                objective=objective
            )

            # Measure both objectives
            ln_trees = count_spanning_trees(current_G)
            lambda2 = get_second_eigenvalue(current_G)

            if verbose:
                print(f"  After optimization:")
                print(f"    ln(trees) = {ln_trees:.6f}")
                print(f"    λ₂        = {lambda2:.6f}")

            # Record trajectory
            trajectory.append({
                'cycle': cycle,
                'phase': objective,
                'ln_trees': ln_trees,
                'lambda2': lambda2
            })

            # Check if we've seen this state before (cycle detection)
            state_hash = _hash_graph_state(ln_trees, lambda2)
            if state_hash in seen_states:
                cycle_start = seen_states[state_hash]
                cycled = True
                if verbose:
                    print(f"\n⟳ Cycle detected! Returned to state from cycle {cycle_start}")
                break

            seen_states[state_hash] = cycle

        if cycled:
            break

        # Check convergence: did both objectives improve in this cycle?
        trees_improvement = abs(ln_trees - cycle_start_trees)
        lambda2_improvement = abs(lambda2 - cycle_start_lambda2)

        if verbose:
            print(f"\nCycle {cycle} summary:")
            print(f"  Δln(trees) = {ln_trees - cycle_start_trees:+.6f}")
            print(f"  Δλ₂        = {lambda2 - cycle_start_lambda2:+.6f}")

        if trees_improvement < min_improvement and lambda2_improvement < min_improvement:
            converged = True
            if verbose:
                print(f"\n✓ Converged! Both objectives stable (changes < {min_improvement})")
            break

    if verbose:
        print(f"\n{'='*70}")
        print("Final Results")
        print(f"{'='*70}")
        print(f"Total cycles: {cycle}")
        print(f"Converged: {converged}")
        print(f"Cycled: {cycled}")
        if cycled:
            print(f"  Cycle detected at cycle {cycle}, returning to state from cycle {cycle_start}")
        print(f"\nFinal state:")
        print(f"  ln(trees) = {ln_trees:.6f}")
        print(f"  λ₂        = {lambda2:.6f}")

        # Summary of trajectory
        print(f"\nTrajectory summary ({len(trajectory)} states):")
        print(f"  Initial: ln(trees)={trajectory[0]['ln_trees']:.6f}, λ₂={trajectory[0]['lambda2']:.6f}")
        print(f"  Final:   ln(trees)={trajectory[-1]['ln_trees']:.6f}, λ₂={trajectory[-1]['lambda2']:.6f}")
        print(f"  Δln(trees) = {trajectory[-1]['ln_trees'] - trajectory[0]['ln_trees']:+.6f}")
        print(f"  Δλ₂        = {trajectory[-1]['lambda2'] - trajectory[0]['lambda2']:+.6f}")

    return {
        'final_graph': current_G,
        'trajectory': trajectory,
        'converged': converged,
        'cycled': cycled,
        'cycle_start': cycle_start if cycled else -1,
        'num_cycles': cycle
    }


def _hash_graph_state(ln_trees, lambda2, precision=5):
    """
    Create a hash of the graph state based on objective values.

    Args:
        ln_trees: ln(spanning trees)
        lambda2: second eigenvalue
        precision: number of decimal places to round to

    Returns:
        tuple: rounded state for hashing
    """
    return (round(ln_trees, precision), round(lambda2, precision))
