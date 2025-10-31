#!/usr/bin/env python3
"""
Generate the champion planar cubic graph for the challenge dataset.

Uses triangulation-based diagonal flip optimization:
- n_points = 77 (gives N=150 vertices in dual)
- restarts = 60 (use all CPUs)
- max_iterations = 150
- objective = expansion (minimize λ₂)
"""

import sys
import pickle
import time

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.planar.triangulation import (
    optimize_triangulation_multi_restart_parallel,
    triangulation_to_dual_cubic,
)
from cubic_graph_optimizer.core.spanning_trees import get_second_eigenvalue

print("="*70)
print("GENERATING CHAMPION PLANAR CUBIC GRAPH")
print("="*70)
print(f"Target: N ≈ 150 vertices")
print(f"Objective: Minimize λ₂(A) = maximize expansion")
print(f"Method: Triangulation diagonal flips")
print(f"Parameters:")
print(f"  n_points = 77")
print(f"  restarts = 60")
print(f"  max_iterations = 150")
print("="*70)
print()

# Objective function: minimize λ₂ of dual cubic graph
def expansion_objective(G_tri):
    """Minimize λ₂ to maximize expansion."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    lambda2 = get_second_eigenvalue(G_dual)
    return -lambda2  # Negative because we maximize

start_time = time.time()

# Run optimization
G_tri_best, G_dual_best, best_value, stats = optimize_triangulation_multi_restart_parallel(
    n_points=77,
    objective_func=expansion_objective,
    maximize=True,  # Maximize -λ₂ = minimize λ₂
    restarts=60,
    max_iterations=150,
    n_jobs=-1,  # Use all CPUs
    verbose=True
)

elapsed = time.time() - start_time

lambda2_A = get_second_eigenvalue(G_dual_best)
lambda2_L = 3.0 - lambda2_A
n = G_dual_best.number_of_nodes()

print()
print("="*70)
print("CHAMPION GENERATED")
print("="*70)
print(f"N = {n}")
print(f"λ₂(A) = {lambda2_A:.6f} (adjacency)")
print(f"λ₂(L) = {lambda2_L:.6f} (Laplacian/algebraic connectivity)")
print(f"Spielman-Teng bound: 0.16")
print(f"Fraction of bound: {lambda2_L/0.16*100:.1f}%")
print(f"Time: {elapsed:.1f}s")
print()
print(f"Statistics:")
print(f"  Best value: {best_value:.6f}")
print(f"  Total flips: {stats.get('total_flips', 'N/A')}")
print("="*70)

# Save champion
output_pkl = 'optimized_graphs/planar_n150_champion.pkl'

with open(output_pkl, 'wb') as f:
    pickle.dump(G_dual_best, f)

print(f"\nChampion saved to: {output_pkl}")
print(f"  N = {n}")
print(f"  λ₂(L) = {lambda2_L:.6f}")
