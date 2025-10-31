#!/usr/bin/env python3
"""
Optimize planar cubic graph with N≈150 vertices using TRIANGULATION optimization.
This is the correct approach that worked in the sweep!
"""

import sys
import json
from datetime import datetime

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.planar.triangulation import (
    optimize_triangulation_multi_restart_parallel,
    triangulation_to_dual_cubic,
)
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue
from cubic_graph_optimizer.utils.io import save_graph
import networkx as nx
import numpy as np


def objective_trees(G_tri):
    """Objective: maximize spanning trees on dual cubic graph."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return count_spanning_trees(G_dual)


def objective_expansion(G_tri):
    """Objective: minimize λ₂ on dual cubic graph."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return -get_second_eigenvalue(G_dual)  # Negative for minimization via maximization


def run_optimization(objective_name, objective_func, maximize, n_points, restarts):
    """Run a single optimization."""
    print(f"\n{'='*70}")
    print(f"Starting {objective_name} optimization (triangulation-based)")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    G_tri_best, G_dual_best, best_value, stats = optimize_triangulation_multi_restart_parallel(
        n_points=n_points,
        objective_func=objective_func,
        maximize=maximize,
        restarts=restarts,
        max_iterations=200,
        n_jobs=-1,  # Use all cores
        verbose=True,
        seed=None,
    )

    elapsed = datetime.now() - start_time

    # Compute both metrics
    ln_trees = count_spanning_trees(G_dual_best)
    lambda2 = get_second_eigenvalue(G_dual_best)

    print(f"\n{'='*70}")
    print(f"{objective_name.upper()} optimization complete")
    print(f"{'='*70}")
    print(f"Actual N: {G_dual_best.number_of_nodes()}")
    print(f"ln(trees): {ln_trees:.6f}")
    print(f"λ₂: {lambda2:.6f}")
    print(f"Time: {elapsed.total_seconds():.1f}s")
    print(f"{'='*70}\n")

    return {
        'objective': objective_name,
        'G_tri': G_tri_best,
        'G_dual': G_dual_best,
        'ln_trees': ln_trees,
        'lambda2': lambda2,
        'stats': stats,
        'elapsed': elapsed.total_seconds()
    }


def main():
    target_n = 150
    n_points = (target_n + 4) // 2  # 77 points
    restarts = 150  # Match the sweep

    print(f"Optimizing planar cubic graph with N≈{target_n} vertices")
    print(f"Using TRIANGULATION optimization (diagonal flips on primal)")
    print(f"  n_points = {n_points}")
    print(f"  restarts = {restarts}")
    print(f"  max_iterations = 200")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Run BOTH objectives
    results = {}

    # 1. Spanning trees optimizer
    results['trees'] = run_optimization(
        'spanning_trees',
        objective_trees,
        maximize=True,
        n_points=n_points,
        restarts=restarts
    )

    # 2. Expansion optimizer
    results['expansion'] = run_optimization(
        'expansion',
        objective_expansion,
        maximize=True,  # We negated it, so maximize the negative
        n_points=n_points,
        restarts=restarts
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON OF RESULTS")
    print("=" * 70)

    trees_result = results['trees']
    expansion_result = results['expansion']

    print(f"\nSpanning trees optimizer:")
    print(f"  N = {trees_result['G_dual'].number_of_nodes()}")
    print(f"  ln(trees) = {trees_result['ln_trees']:.6f}  ← optimized for this")
    print(f"  λ₂ = {trees_result['lambda2']:.6f}")
    print(f"  Time: {trees_result['elapsed']:.1f}s")

    print(f"\nExpansion optimizer:")
    print(f"  N = {expansion_result['G_dual'].number_of_nodes()}")
    print(f"  ln(trees) = {expansion_result['ln_trees']:.6f}")
    print(f"  λ₂ = {expansion_result['lambda2']:.6f}  ← optimized for this")
    print(f"  Time: {expansion_result['elapsed']:.1f}s")

    # Determine winners
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

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    for obj_name, result in results.items():
        G_dual = result['G_dual']
        actual_n = G_dual.number_of_nodes()

        output_file = f"optimized_graphs/planar_n{actual_n}_{obj_name}_triangulation.pkl"
        print(f"\nSaving {obj_name} graph to: {output_file}")
        save_graph(G_dual, output_file, result['ln_trees'])

        # Save full result with triangulation
        full_file = f"optimized_graphs/planar_n{actual_n}_{obj_name}_triangulation_full.json"
        adj_dual = [sorted(list(G_dual.neighbors(i))) for i in range(G_dual.number_of_nodes())]

        G_tri = result['G_tri']
        adj_tri = [sorted(list(G_tri.neighbors(i))) for i in range(G_tri.number_of_nodes())]

        full_data = {
            'target_n': target_n,
            'actual_n': actual_n,
            'n_points': n_points,
            'ln_trees': result['ln_trees'],
            'lambda2': result['lambda2'],
            'objective': obj_name,
            'dual_adjacency': adj_dual,
            'triangulation_adjacency': adj_tri,
            'stats': result['stats'],
            'elapsed': result['elapsed']
        }

        with open(full_file, 'w') as f:
            json.dump(full_data, f, indent=2)
        print(f"Saved full data to: {full_file}")

    # Save best for expansion
    best_n = best_for_expansion['G_dual'].number_of_nodes()
    best_file = f"optimized_graphs/planar_n{best_n}_best_expansion_triangulation.pkl"
    print(f"\n⭐ Saving BEST for λ₂ (from {best_objective} optimizer) to: {best_file}")
    save_graph(best_for_expansion['G_dual'], best_file, best_for_expansion['ln_trees'])

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
