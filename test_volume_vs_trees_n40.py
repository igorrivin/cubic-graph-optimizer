#!/usr/bin/env python
"""
Test the conjecture: High spanning trees ↔ High hyperbolic volume

For N=40 planar cubic graph (dual to 22-vertex triangulation):
1. Load champion graph optimized for spanning trees
2. Generate random Delaunay triangulations (22 vertices)
3. Compute hyperbolic volumes
4. Compare: Does max spanning trees → max volume?
"""

import sys
import json
import numpy as np
import networkx as nx
import torch

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, '../ideal_poly_volume_toolkit')

from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees
from cubic_graph_optimizer.planar.triangulation import (
    random_triangulation_from_sphere,
    triangulation_to_dual_cubic,
)
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch
)


def compute_hyperbolic_volume(points_3d):
    """
    Compute hyperbolic volume from 3D points on sphere.

    Args:
        points_3d: Nx3 array of points on unit sphere

    Returns:
        Total hyperbolic volume
    """
    # Convert sphere points to complex via stereographic projection
    # Project from north pole to complex plane
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Stereographic projection: (x,y,z) -> (x/(1-z), y/(1-z))
    # Handle points near north pole carefully
    Z = (x + 1j*y) / (1 - z + 1e-10)

    # Get Delaunay triangulation
    try:
        idx = delaunay_triangulation_indices(Z)
    except:
        return np.nan

    # Compute volume using Bloch-Wigner (same as GUI)
    Z_torch = torch.tensor(Z, dtype=torch.complex128, device='cpu')

    total_volume = 0.0
    for (i, j, k) in idx:
        try:
            from ideal_poly_volume_toolkit.geometry import triangle_volume_bloch_wigner
            vol = triangle_volume_bloch_wigner(
                Z_torch[i], Z_torch[j], Z_torch[k]
            )
            total_volume += vol.item()
        except:
            continue

    return total_volume


def main():
    print('='*70)
    print('TESTING: SPANNING TREES vs HYPERBOLIC VOLUME (N=40)')
    print('='*70)
    print()

    # Load champion N=40 graph
    print('Loading N=40 champion graph (optimized for spanning trees)...')
    with open('planar_n40_record.json') as f:
        record = json.load(f)

    champion_ln_trees = record['ln_trees']
    print(f'  Champion ln(trees) = {champion_ln_trees:.6f}')
    print()

    # Note: We don't have the original triangulation saved!
    # But we can create random ones and compare

    print('Generating random Delaunay triangulations (22 vertices)...')
    print('Computing spanning trees and hyperbolic volumes...')
    print()

    n_points = 22  # For N=40 cubic
    n_samples = 20

    results = []

    for i in range(n_samples):
        # Generate random Delaunay triangulation from sphere points
        G_tri, points_3d = random_triangulation_from_sphere(n_points, seed=1000+i)
        G_dual = triangulation_to_dual_cubic(G_tri)

        # Compute spanning trees
        ln_trees = count_spanning_trees(G_dual)

        # Compute hyperbolic volume from the SAME 3D points
        volume = compute_hyperbolic_volume(points_3d)

        results.append({
            'ln_trees': ln_trees,
            'volume': volume,
        })

        print(f'  Sample {i+1:2d}: ln(trees)={ln_trees:.4f}, volume={volume:.6f}')

    print()
    print('='*70)
    print('CORRELATION ANALYSIS')
    print('='*70)

    ln_trees_vals = np.array([r['ln_trees'] for r in results])
    volume_vals = np.array([r['volume'] for r in results])

    # Compute correlation
    correlation = np.corrcoef(ln_trees_vals, volume_vals)[0, 1]

    print(f'Correlation (ln(trees) vs volume): {correlation:.6f}')
    print()

    if abs(correlation) > 0.7:
        print('✓ STRONG CORRELATION!')
        if correlation > 0:
            print('  High spanning trees → High hyperbolic volume')
        else:
            print('  High spanning trees → Low hyperbolic volume')
    elif abs(correlation) > 0.3:
        print('~ MODERATE CORRELATION')
    else:
        print('✗ WEAK OR NO CORRELATION')

    print()
    print('Summary statistics:')
    print(f'  ln(trees): {ln_trees_vals.mean():.4f} ± {ln_trees_vals.std():.4f}')
    print(f'  volume:    {volume_vals.mean():.6f} ± {volume_vals.std():.6f}')
    print()
    print(f'  ln(trees) range: [{ln_trees_vals.min():.4f}, {ln_trees_vals.max():.4f}]')
    print(f'  volume range:    [{volume_vals.min():.6f}, {volume_vals.max():.6f}]')


if __name__ == '__main__':
    main()
