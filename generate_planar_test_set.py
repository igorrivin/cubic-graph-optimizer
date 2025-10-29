#!/usr/bin/env python3
"""Generate test set for planar cubic graphs with optimized graphs that beat random."""

import sys
import json
sys.path.insert(0, '.')

from cubic_graph_optimizer.planar.triangulation import (
    random_triangulation_from_sphere,
    triangulation_to_dual_cubic,
    optimize_triangulation_multi_restart
)
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue


def triangulation_objective_trees(G_tri):
    """Objective for triangulation: spanning trees of dual."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return count_spanning_trees(G_dual)


def graph_to_adjacency_list(G):
    """Convert networkx graph to adjacency list format."""
    n = G.number_of_nodes()
    adj_list = [sorted(list(G.neighbors(i))) for i in range(n)]
    return adj_list


def generate_test_case(n_points, restarts=10, num_random_comparisons=5):
    """
    Generate a test case: one optimized graph and several random graphs for comparison.

    Returns:
        List of test case dictionaries
    """
    print(f"\n{'='*70}")
    print(f"Generating test case for n_points={n_points} (target N≈{2*n_points-4})")
    print(f"{'='*70}")

    # Generate optimized graph using multi-restart
    print(f"\nOptimizing with {restarts} restarts...")
    G_tri_opt, G_dual_opt, best_value, summary = optimize_triangulation_multi_restart(
        n_points=n_points,
        objective_func=triangulation_objective_trees,
        maximize=True,
        restarts=restarts,
        max_iterations=200,
        verbose=True,
    )

    n = G_dual_opt.number_of_nodes()
    trees_opt = count_spanning_trees(G_dual_opt)
    lambda2_opt = get_second_eigenvalue(G_dual_opt)

    print(f"\nOptimized graph: N={n}, ln(trees)={trees_opt:.3f}, λ₂={lambda2_opt:.3f}")

    # Generate random graphs for comparison
    print(f"\nGenerating {num_random_comparisons} random comparison graphs...")
    random_graphs = []
    for i in range(num_random_comparisons):
        G_tri_random, _ = random_triangulation_from_sphere(n_points)
        G_dual_random = triangulation_to_dual_cubic(G_tri_random)
        trees_random = count_spanning_trees(G_dual_random)
        lambda2_random = get_second_eigenvalue(G_dual_random)

        random_graphs.append({
            'G_dual': G_dual_random,
            'trees': trees_random,
            'lambda2': lambda2_random,
        })

        print(f"  Random {i+1}: ln(trees)={trees_random:.3f}, λ₂={lambda2_random:.3f}")

    # Compute statistics
    random_trees = [g['trees'] for g in random_graphs]
    avg_random_trees = sum(random_trees) / len(random_trees)
    max_random_trees = max(random_trees)
    min_random_trees = min(random_trees)

    print(f"\nRandom graphs: ln(trees) range=[{min_random_trees:.3f}, {max_random_trees:.3f}], avg={avg_random_trees:.3f}")
    print(f"Improvement over average: {trees_opt - avg_random_trees:+.3f}")
    print(f"Improvement over best random: {trees_opt - max_random_trees:+.3f}")

    # Verify that optimized beats average (and ideally best random)
    if trees_opt <= avg_random_trees:
        print(f"⚠ WARNING: Optimized graph does NOT beat random average!")
        print(f"   Try increasing restarts or max_iterations")
    elif trees_opt <= max_random_trees:
        print(f"⚠ Note: Optimized beats average but not the best random graph")
    else:
        print(f"✓ Success: Optimized graph beats all random graphs!")

    # Build test cases
    test_cases = []

    # Add optimized graph (expected=true)
    test_cases.append({
        'name': f'planar_n{n}_optimized_trees',
        'adjacency_list': graph_to_adjacency_list(G_dual_opt),
        'expected': True,
        'description': f'Optimized planar cubic N={n} (trees={trees_opt:.3f}, λ₂={lambda2_opt:.3f})',
        'ln_trees': trees_opt,
        'lambda2': lambda2_opt,
    })

    # Add random graphs (expected=false)
    for i, g in enumerate(random_graphs, 1):
        test_cases.append({
            'name': f'planar_n{n}_random_{i}',
            'adjacency_list': graph_to_adjacency_list(g['G_dual']),
            'expected': False,
            'description': f'Random planar cubic N={n} (trees={g["trees"]:.3f}, λ₂={g["lambda2"]:.3f})',
            'ln_trees': g['trees'],
            'lambda2': g['lambda2'],
        })

    return test_cases


def main():
    print("="*70)
    print("Generating Planar Cubic Graph Test Set")
    print("="*70)

    all_test_cases = []

    # Generate test cases for different sizes
    # n_points=11 → N=20 (dodecahedron size)
    # n_points=22 → N=40
    # n_points=32 → N=60

    test_configs = [
        (11, 10, 5),  # (n_points, restarts, num_random)
        (22, 10, 5),
        (32, 10, 5),
    ]

    for n_points, restarts, num_random in test_configs:
        test_cases = generate_test_case(n_points, restarts, num_random)
        all_test_cases.extend(test_cases)

    # Save to file
    output_file = 'test_planar_graphs.json'
    with open(output_file, 'w') as f:
        json.dump(all_test_cases, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Test set generated: {len(all_test_cases)} test cases")
    print(f"Saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    optimized_cases = [tc for tc in all_test_cases if tc['expected']]
    random_cases = [tc for tc in all_test_cases if not tc['expected']]

    print(f"\nSummary:")
    print(f"  Optimized graphs: {len(optimized_cases)}")
    print(f"  Random graphs: {len(random_cases)}")
    print(f"\nOptimized graphs:")
    for tc in optimized_cases:
        print(f"  {tc['name']}: ln(trees)={tc['ln_trees']:.3f}, λ₂={tc['lambda2']:.3f}")


if __name__ == '__main__':
    main()
