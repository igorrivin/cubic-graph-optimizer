#!/usr/bin/env python
"""
Test the conjecture: High hyperbolic volume ↔ High spanning trees

Approach:
1. Optimize hyperbolic volume for 22 vertices (→ N=40 cubic graph)
2. Read off the resulting triangulation
3. Compute spanning trees of the dual cubic graph
4. Compare to our champion (optimized for spanning trees)

If conjecture holds: volume-optimal ≈ trees-optimal
"""

import sys
import numpy as np
import torch
from scipy.optimize import differential_evolution

sys.path.insert(0, '.')
sys.path.insert(0, '../ideal_poly_volume_toolkit')

from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees
from cubic_graph_optimizer.planar.triangulation import triangulation_to_dual_cubic
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_bloch_wigner,
)
import networkx as nx


def spherical_to_complex(theta, phi):
    """Convert spherical coordinates to complex via stereographic projection."""
    return np.tan(theta/2) * np.exp(1j * phi)


def compute_volume_from_params(params, n_vertices):
    """
    Compute hyperbolic volume from optimization parameters.

    Based on gui.py compute_volume function.
    """
    # Build complex points: 0, 1, ∞ are fixed, rest parameterized
    complex_points = [complex(0, 0), complex(1, 0)]
    n_params = n_vertices - 3

    for i in range(n_params):
        theta = params[2*i]
        phi = params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)

    # Get Delaunay triangulation
    try:
        idx = delaunay_triangulation_indices(Z_np)
    except:
        return 1000.0  # Invalid configuration

    # Compute volume
    Z_torch = torch.tensor(Z_np, dtype=torch.complex128, device='cpu')

    total_volume = 0.0
    for (i, j, k) in idx:
        try:
            vol = triangle_volume_bloch_wigner(
                Z_torch[i], Z_torch[j], Z_torch[k]
            )
            total_volume += vol.item()
        except:
            return 1000.0

    return -total_volume  # Negative for minimization


def optimize_volume(n_vertices, seed=42):
    """
    Optimize hyperbolic volume for n_vertices.

    Returns:
        (best_params, best_volume, complex_points, triangulation_indices)
    """
    n_params = n_vertices - 3
    n_vars = 2 * n_params  # theta, phi pairs

    print(f'Optimizing hyperbolic volume for {n_vertices} vertices...')
    print(f'  Free parameters: {n_params} points × 2 coords = {n_vars} variables')
    print()

    # Bounds: theta in [0, π], phi in [0, 2π]
    bounds = [(0, np.pi), (0, 2*np.pi)] * n_params

    # Run optimization
    result = differential_evolution(
        lambda params: compute_volume_from_params(params, n_vertices),
        bounds,
        seed=seed,
        maxiter=300,
        popsize=15,
        workers=1,
        disp=True,
    )

    best_params = result.x
    best_volume = -result.fun  # Undo negation

    # Reconstruct complex points
    complex_points = [complex(0, 0), complex(1, 0)]
    for i in range(n_params):
        theta = best_params[2*i]
        phi = best_params[2*i + 1]
        z = spherical_to_complex(theta, phi)
        complex_points.append(z)

    Z_np = np.array(complex_points, dtype=np.complex128)
    idx = delaunay_triangulation_indices(Z_np)

    return best_params, best_volume, Z_np, idx


def triangulation_indices_to_graph(indices, n_vertices):
    """Convert triangulation indices to NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    for (i, j, k) in indices:
        G.add_edge(i, j)
        G.add_edge(j, k)
        G.add_edge(k, i)

    return G


def main():
    print('='*70)
    print('TESTING: VOLUME-OPTIMAL → TREES?')
    print('='*70)
    print()

    n_vertices = 22  # Corresponds to N=40 cubic graph

    # Optimize for volume
    params, volume, Z_complex, tri_indices = optimize_volume(n_vertices, seed=42)

    print()
    print('='*70)
    print('VOLUME OPTIMIZATION RESULT')
    print('='*70)
    print(f'Optimized volume: {volume:.6f}')
    print(f'Number of triangles: {len(tri_indices)}')
    print()

    # Build triangulation graph
    G_tri = triangulation_indices_to_graph(tri_indices, n_vertices)

    print('Triangulation properties:')
    print(f'  Vertices: {G_tri.number_of_nodes()}')
    print(f'  Edges: {G_tri.number_of_edges()}')
    print(f'  Connected: {nx.is_connected(G_tri)}')
    print()

    # Convert to dual cubic graph
    G_dual = triangulation_to_dual_cubic(G_tri)

    print('Dual cubic graph properties:')
    print(f'  Vertices: {G_dual.number_of_nodes()}')
    print(f'  Edges: {G_dual.number_of_edges()}')
    degrees = [G_dual.degree(v) for v in G_dual.nodes()]
    print(f'  Cubic: {all(d == 3 for d in degrees)}')
    print()

    # Compute spanning trees
    ln_trees = count_spanning_trees(G_dual)

    print('='*70)
    print('SPANNING TREE COUNT')
    print('='*70)
    print(f'Volume-optimized graph: ln(trees) = {ln_trees:.6f}')
    print(f'                        trees = {np.exp(ln_trees):.3e}')
    print()

    # Compare to our champion
    champion_ln_trees = 31.354421
    champion_volume = None  # Unknown

    print('='*70)
    print('COMPARISON')
    print('='*70)
    print(f'Our champion (trees-optimized):')
    print(f'  ln(trees) = {champion_ln_trees:.6f}')
    print(f'  volume = unknown')
    print()
    print(f'Volume-optimized graph:')
    print(f'  ln(trees) = {ln_trees:.6f}')
    print(f'  volume = {volume:.6f}')
    print()

    diff = ln_trees - champion_ln_trees
    print(f'Difference: {diff:+.6f}')

    if abs(diff) < 0.05:
        print()
        print('🎯 STRONG EVIDENCE FOR CONJECTURE!')
        print('   Volume-optimal ≈ Trees-optimal (within 0.05)')
    elif abs(diff) < 0.2:
        print()
        print('✓ MODERATE EVIDENCE FOR CONJECTURE')
        print('   Volume-optimal and trees-optimal are close')
    else:
        print()
        if diff > 0:
            print('⚠️  Volume-optimal has MORE trees than our champion!')
            print('    This would mean our champion is not optimal for trees.')
        else:
            print('✗ CONJECTURE LIKELY FALSE')
            print('   Volume-optimal has significantly fewer trees.')

    print()
    print('Random baseline (from previous test):')
    print(f'  Mean: ln(trees) = 31.03 ± 0.10')
    print(f'  Volume-optimal improvement: {ln_trees - 31.03:+.3f}')
    print(f'  Champion improvement: {champion_ln_trees - 31.03:+.3f}')


if __name__ == '__main__':
    main()
