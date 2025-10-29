#!/usr/bin/env python
"""
Comprehensive sweep of planar cubic graph optimization for N=4 to N=400.

This script runs parallel multi-restart optimization for all even N from 4 to 400,
saving results incrementally with full metadata.

Results are saved to: planar_sweep_results/planar_n{N}_result.json

Can be resumed if interrupted - checks for existing results.
"""

import json
import os
import time
import numpy as np
import networkx as nx
from pathlib import Path

from cubic_graph_optimizer.planar.triangulation import (
    optimize_triangulation_multi_restart_parallel,
    triangulation_to_dual_cubic,
)
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees


def determine_restarts(N):
    """Determine number of restarts based on graph size."""
    if N <= 40:
        return 100
    elif N <= 80:
        return 200
    elif N <= 120:
        return 300
    elif N <= 200:
        return 150
    elif N <= 300:
        return 100
    else:
        return 50


def save_result(N, G_dual, ln_trees, stats, output_dir):
    """Save result with full metadata."""
    # Convert graph to adjacency list
    adj_list = [sorted(list(G_dual.neighbors(i))) for i in range(G_dual.number_of_nodes())]

    # Compute expansion (λ₂)
    eigenvalues = nx.adjacency_spectrum(G_dual)
    eigenvalues_sorted = sorted(eigenvalues, reverse=True)
    lambda2 = float(eigenvalues_sorted[1].real) if len(eigenvalues_sorted) > 1 else 0.0

    result = {
        'N': N,
        'n_points': (N + 4) // 2,
        'ln_trees': float(ln_trees),
        'trees': float(np.exp(ln_trees)),
        'lambda2': lambda2,
        'adjacency_list': adj_list,
        'optimization_stats': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    filename = output_dir / f'planar_n{N:03d}_result.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

    return filename


def objective_trees(G_tri):
    """Objective function: maximize spanning trees (via dual)."""
    G_dual = triangulation_to_dual_cubic(G_tri)
    return count_spanning_trees(G_dual)


def main():
    # Setup
    output_dir = Path('planar_sweep_results')
    output_dir.mkdir(exist_ok=True)

    print('='*80)
    print('PLANAR CUBIC GRAPH OPTIMIZATION SWEEP: N=4 to N=400')
    print('='*80)
    print()
    print(f'Output directory: {output_dir}')
    print(f'Using parallel optimization on all available cores')
    print()

    # Summary tracking
    summary = {
        'start_time': time.time(),
        'results': [],
    }

    # Sweep all even N from 4 to 400
    N_values = range(4, 402, 2)  # Even values only (3N/2 must be integer)
    total_count = len(N_values)

    for idx, N in enumerate(N_values, 1):
        # Check if already computed
        result_file = output_dir / f'planar_n{N:03d}_result.json'
        if result_file.exists():
            print(f'[{idx}/{total_count}] N={N:3d}: Already computed, skipping...')
            with open(result_file) as f:
                result = json.load(f)
                summary['results'].append({
                    'N': N,
                    'ln_trees': result['ln_trees'],
                    'restarts': result['optimization_stats']['restarts'],
                    'time': result['optimization_stats'].get('total_time', 0),
                    'status': 'cached',
                })
            continue

        # Determine parameters
        n_points = (N + 4) // 2
        restarts = determine_restarts(N)

        print(f'[{idx}/{total_count}] N={N:3d} (n_points={n_points:2d}): ', end='', flush=True)
        print(f'Running {restarts} restarts... ', end='', flush=True)

        start_time = time.time()

        try:
            # Run optimization
            G_tri_best, G_dual_best, ln_trees, stats = optimize_triangulation_multi_restart_parallel(
                n_points=n_points,
                objective_func=objective_trees,
                maximize=True,
                restarts=restarts,
                max_iterations=200,
                n_jobs=-1,  # Use all cores
                verbose=False,
                seed=None,
            )

            elapsed = time.time() - start_time

            # Save result
            filename = save_result(N, G_dual_best, ln_trees, stats, output_dir)

            print(f'ln(trees)={ln_trees:.3f} in {elapsed:.1f}s')

            # Track in summary
            summary['results'].append({
                'N': N,
                'ln_trees': float(ln_trees),
                'restarts': restarts,
                'time': elapsed,
                'status': 'computed',
            })

            # Save summary after each computation (for crash recovery)
            summary_file = output_dir / 'sweep_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            print(f'ERROR: {e}')
            summary['results'].append({
                'N': N,
                'error': str(e),
                'status': 'failed',
            })

    # Final summary
    summary['end_time'] = time.time()
    summary['total_time'] = summary['end_time'] - summary['start_time']

    summary_file = output_dir / 'sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('='*80)
    print('SWEEP COMPLETE')
    print('='*80)
    print(f'Total time: {summary["total_time"]/3600:.2f} hours')
    print(f'Results saved to: {output_dir}')
    print(f'Summary: {summary_file}')
    print()

    # Print some statistics
    computed = [r for r in summary['results'] if r['status'] == 'computed']
    if computed:
        total_time = sum(r['time'] for r in computed)
        avg_time = total_time / len(computed)
        print(f'Computed {len(computed)} graphs in {total_time/3600:.2f} hours')
        print(f'Average time per graph: {avg_time:.1f}s')


if __name__ == '__main__':
    main()
