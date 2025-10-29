#!/usr/bin/env python3
"""Investigate why planar optimization doesn't improve random graphs."""

import sys
sys.path.insert(0, '.')

from cubic_graph_optimizer.planar.triangulation import (
    random_triangulation_from_sphere,
    triangulation_to_dual_cubic,
    optimize_triangulation
)
from cubic_graph_optimizer.planar.planar_ops import random_planar_cubic_from_sphere
from cubic_graph_optimizer.planar.alternating_planar import alternating_planar_optimization
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue


def triangulation_objective_trees(G_tri):
    """Objective for triangulation: spanning trees of dual."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return count_spanning_trees(G_dual)


def triangulation_objective_expansion(G_tri):
    """Objective for triangulation: λ₂ of dual."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return get_second_eigenvalue(G_dual)


def main():
    print("="*70)
    print("Investigating Planar Graph Optimization")
    print("="*70)

    # Generate initial random triangulation
    print("\n1. Starting with random N≈40 planar cubic graph...")
    G_tri_init, _ = random_triangulation_from_sphere(22, seed=42)
    G_dual_init = triangulation_to_dual_cubic(G_tri_init)

    trees_init = count_spanning_trees(G_dual_init)
    lambda2_init = get_second_eigenvalue(G_dual_init)

    print(f"   Initial: N={G_dual_init.number_of_nodes()}")
    print(f"   ln(trees) = {trees_init:.6f}")
    print(f"   λ₂ = {lambda2_init:.6f}")

    # Try single-objective: maximize trees
    print("\n2. Single-objective: Maximize spanning trees only...")
    G_tri_trees, _, stats_trees = optimize_triangulation(
        G_tri_init.copy(),
        triangulation_objective_trees,
        maximize=True,
        max_iterations=200,
        verbose=False
    )
    G_dual_trees = triangulation_to_dual_cubic(G_tri_trees)

    trees_trees = count_spanning_trees(G_dual_trees)
    lambda2_trees = get_second_eigenvalue(G_dual_trees)

    print(f"   After trees optimization:")
    print(f"   ln(trees) = {trees_trees:.6f} (Δ = {trees_trees - trees_init:+.6f})")
    print(f"   λ₂ = {lambda2_trees:.6f} (Δ = {lambda2_trees - lambda2_init:+.6f})")
    print(f"   Flips performed: {stats_trees['flips_performed']}")

    # Try single-objective: minimize λ₂
    print("\n3. Single-objective: Minimize expansion (λ₂) only...")
    G_tri_lambda, _, stats_lambda = optimize_triangulation(
        G_tri_init.copy(),
        triangulation_objective_expansion,
        maximize=False,
        max_iterations=200,
        verbose=False
    )
    G_dual_lambda = triangulation_to_dual_cubic(G_tri_lambda)

    trees_lambda = count_spanning_trees(G_dual_lambda)
    lambda2_lambda = get_second_eigenvalue(G_dual_lambda)

    print(f"   After expansion optimization:")
    print(f"   ln(trees) = {trees_lambda:.6f} (Δ = {trees_lambda - trees_init:+.6f})")
    print(f"   λ₂ = {lambda2_lambda:.6f} (Δ = {lambda2_lambda - lambda2_init:+.6f})")
    print(f"   Flips performed: {stats_lambda['flips_performed']}")

    # Try alternating optimization
    print("\n4. Alternating optimization...")
    G_tri_alt, G_dual_alt, stats_alt = alternating_planar_optimization(
        G_tri_init.copy(),
        max_cycles=10,
        max_iterations_per_phase=100,
        verbose=False
    )

    trees_alt = count_spanning_trees(G_dual_alt)
    lambda2_alt = get_second_eigenvalue(G_dual_alt)

    print(f"   After alternating optimization:")
    print(f"   ln(trees) = {trees_alt:.6f} (Δ = {trees_alt - trees_init:+.6f})")
    print(f"   λ₂ = {lambda2_alt:.6f} (Δ = {lambda2_alt - lambda2_init:+.6f})")

    # Compare to several random graphs
    print("\n5. Comparing to fresh random graphs...")
    random_trees = []
    random_lambda2 = []

    for seed in [100, 200, 300, 400, 500]:
        G_random = random_planar_cubic_from_sphere(22, seed=seed)
        trees = count_spanning_trees(G_random)
        lambda2 = get_second_eigenvalue(G_random)
        random_trees.append(trees)
        random_lambda2.append(lambda2)
        print(f"   Random seed {seed}: trees={trees:.6f}, λ₂={lambda2:.6f}")

    avg_random_trees = sum(random_trees) / len(random_trees)
    avg_random_lambda2 = sum(random_lambda2) / len(random_lambda2)

    print(f"\n   Average random: trees={avg_random_trees:.6f}, λ₂={avg_random_lambda2:.6f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Initial (random):    trees={trees_init:.3f}, λ₂={lambda2_init:.3f}")
    print(f"Trees-optimized:     trees={trees_trees:.3f} ({trees_trees-trees_init:+.3f}), λ₂={lambda2_trees:.3f} ({lambda2_trees-lambda2_init:+.3f})")
    print(f"Expansion-optimized: trees={trees_lambda:.3f} ({trees_lambda-trees_init:+.3f}), λ₂={lambda2_lambda:.3f} ({lambda2_lambda-lambda2_init:+.3f})")
    print(f"Alternating:         trees={trees_alt:.3f} ({trees_alt-trees_init:+.3f}), λ₂={lambda2_alt:.3f} ({lambda2_alt-lambda2_init:+.3f})")
    print(f"Random average:      trees={avg_random_trees:.3f}, λ₂={avg_random_lambda2:.3f}")

    print("\nKEY FINDINGS:")
    max_change = max(abs(trees_trees - trees_init), abs(lambda2_lambda - lambda2_init))
    if max_change < 0.05:
        print("  • Single-objective optimizations show MINIMAL improvement")
        print("  • Diagonal flips on planar triangulations may not significantly")
        print("    change dual graph properties for sphere-based constructions")
        print("  • Random sphere → triangulation → dual is already near-optimal!")
    else:
        print("  • Optimizations DO make changes")
        if trees_trees > trees_init:
            print(f"  • Trees optimization improved by {trees_trees - trees_init:.3f}")
        if lambda2_lambda < lambda2_init:
            print(f"  • Expansion optimization improved by {lambda2_init - lambda2_lambda:.3f}")

    # Additional analysis
    print("\nCOMPARISON TO RANDOM:")
    if trees_trees < avg_random_trees:
        print(f"  ⚠ Trees-optimized ({trees_trees:.3f}) < Random average ({avg_random_trees:.3f})")
    if lambda2_lambda > avg_random_lambda2:
        print(f"  ⚠ Expansion-optimized λ₂ ({lambda2_lambda:.3f}) > Random average ({avg_random_lambda2:.3f})")
    if trees_alt < avg_random_trees:
        print(f"  ⚠ Alternating trees ({trees_alt:.3f}) < Random average ({avg_random_trees:.3f})")


if __name__ == '__main__':
    main()
