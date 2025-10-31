#!/usr/bin/env python3
"""
Optimize planar cubic graph with N≈150 vertices using both objectives.
Run expansion and spanning_trees optimizers in parallel and compare results.
"""

import sys
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.planar.planar_optimizer import optimize_planar_cubic_graph
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue
from cubic_graph_optimizer.utils.io import save_graph
import networkx as nx


def run_optimization(n, objective, restarts, max_iterations, seed):
    """Run a single optimization (for parallel execution)."""
    print(f"\n{'='*70}")
    print(f"Starting {objective} optimization")
    print(f"{'='*70}\n")

    G_best, best_value, stats = optimize_planar_cubic_graph(
        n=n,
        objective=objective,
        restarts=restarts,
        max_iterations=max_iterations,
        verbose=True,
        seed=seed
    )

    if G_best is None:
        return None

    actual_n = G_best.number_of_nodes()
    ln_trees = count_spanning_trees(G_best)
    lambda2 = get_second_eigenvalue(G_best)
    is_planar = nx.is_planar(G_best)

    result = {
        'objective': objective,
        'graph': G_best,
        'actual_n': actual_n,
        'ln_trees': ln_trees,
        'lambda2': lambda2,
        'is_planar': is_planar,
        'stats': stats
    }

    print(f"\n{'='*70}")
    print(f"{objective.upper()} optimization complete")
    print(f"{'='*70}")
    print(f"Actual N: {actual_n}")
    print(f"ln(trees): {ln_trees:.6f}")
    print(f"λ₂: {lambda2:.6f}")
    print(f"{'='*70}\n")

    return result


def main():
    n = 150
    restarts = 20
    max_iterations = 100

    print(f"Optimizing planar cubic graph with N≈{n} vertices")
    print(f"Running TWO optimizations in parallel:")
    print(f"  1. Expansion optimizer (minimize λ₂)")
    print(f"  2. Spanning trees optimizer (maximize ln(trees))")
    print(f"  Restarts per optimizer: {restarts}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Run both optimizations in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_optimization, n, 'expansion', restarts, max_iterations, 42): 'expansion',
            executor.submit(run_optimization, n, 'spanning_trees', restarts, max_iterations, 43): 'spanning_trees'
        }

        results = {}
        for future in as_completed(futures):
            obj_name = futures[future]
            try:
                result = future.result()
                if result:
                    results[obj_name] = result
            except Exception as e:
                print(f"ERROR in {obj_name} optimization: {e}")

    if len(results) < 2:
        print("ERROR: Not all optimizations completed successfully!")
        return

    # Compare results
    expansion_result = results['expansion']
    trees_result = results['spanning_trees']

    print("\n" + "=" * 70)
    print("COMPARISON OF RESULTS")
    print("=" * 70)
    print(f"\nExpansion optimizer (targeting λ₂):")
    print(f"  N = {expansion_result['actual_n']}")
    print(f"  ln(trees) = {expansion_result['ln_trees']:.6f}")
    print(f"  λ₂ = {expansion_result['lambda2']:.6f}  ← optimized for this")

    print(f"\nSpanning trees optimizer (targeting ln(trees)):")
    print(f"  N = {trees_result['actual_n']}")
    print(f"  ln(trees) = {trees_result['ln_trees']:.6f}  ← optimized for this")
    print(f"  λ₂ = {trees_result['lambda2']:.6f}")

    # Determine which has better λ₂
    if trees_result['lambda2'] < expansion_result['lambda2']:
        print(f"\n⭐ WINNER for λ₂: Spanning trees optimizer!")
        print(f"   (λ₂ = {trees_result['lambda2']:.6f} vs {expansion_result['lambda2']:.6f})")
        best_for_expansion = trees_result
        best_objective = 'spanning_trees'
    else:
        print(f"\n⭐ WINNER for λ₂: Expansion optimizer!")
        print(f"   (λ₂ = {expansion_result['lambda2']:.6f} vs {trees_result['lambda2']:.6f})")
        best_for_expansion = expansion_result
        best_objective = 'expansion'

    # Save both graphs
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    for obj_name, result in results.items():
        G = result['graph']
        actual_n = result['actual_n']

        output_file = f"optimized_graphs/planar_n{actual_n}_{obj_name}.pkl"
        print(f"\nSaving {obj_name} graph to: {output_file}")
        save_graph(G, output_file, result['ln_trees'])

        stats_file = f"optimized_graphs/planar_n{actual_n}_{obj_name}_stats.json"
        stats_data = {
            'n_target': n,
            'n_actual': actual_n,
            'ln_trees': result['ln_trees'],
            'lambda2': result['lambda2'],
            'is_planar': result['is_planar'],
            'objective': obj_name,
            'restarts': result['stats']['restarts'],
            'best_value': result['stats']['best_value']
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"Saved stats to: {stats_file}")

    # Save the best one for expansion with a special name
    best_n = best_for_expansion['actual_n']
    best_file = f"optimized_graphs/planar_n{best_n}_best_expansion.pkl"
    print(f"\n⭐ Saving BEST for λ₂ (from {best_objective} optimizer) to: {best_file}")
    save_graph(best_for_expansion['graph'], best_file, best_for_expansion['ln_trees'])

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
