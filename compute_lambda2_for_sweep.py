#!/usr/bin/env python3
"""
Compute λ₂ (expansion) for all planar sweep results.

The sweep optimized for spanning trees, but let's see what λ₂ values we got!
This will tell us if trees optimization also gives good expansion.
"""

import json
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from cubic_graph_optimizer.core.spanning_trees import get_second_eigenvalue


def load_graph_from_adjacency(adj_list):
    """Reconstruct graph from adjacency list."""
    G = nx.Graph()
    G.add_nodes_from(range(len(adj_list)))

    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            if node < neighbor:  # Add each edge only once
                G.add_edge(node, neighbor)

    return G


def compute_lambda2_for_all_sweeps():
    """Load all sweep results and compute λ₂ for each."""

    results_dir = Path('planar_sweep_results')
    result_files = sorted(results_dir.glob('planar_n*_result.json'))

    print(f"Found {len(result_files)} sweep results")
    print("Computing λ₂ for each graph...\n")

    augmented_results = []

    for result_file in tqdm(result_files, desc="Processing"):
        # Load existing result
        with open(result_file) as f:
            data = json.load(f)

        N = data['N']

        # Reconstruct graph
        G = load_graph_from_adjacency(data['adjacency_list'])

        # Verify it's cubic
        if not all(d == 3 for _, d in G.degree()):
            print(f"WARNING: N={N} is not cubic!")
            continue

        # Compute λ₂
        lambda2 = get_second_eigenvalue(G)

        # Store augmented result
        augmented_results.append({
            'N': N,
            'n_points': data['n_points'],
            'ln_trees': data['ln_trees'],
            'lambda2_original': data.get('lambda2', None),  # From original sweep
            'lambda2_computed': lambda2,
            'file': result_file.name
        })

    return augmented_results


def analyze_results(results):
    """Analyze the λ₂ values."""
    print("\n" + "="*70)
    print("ANALYSIS: Spanning Trees Optimization → Expansion")
    print("="*70)

    N_values = [r['N'] for r in results]
    lambda2_values = [r['lambda2_computed'] for r in results]
    ln_trees_values = [r['ln_trees'] for r in results]

    print(f"\nSample results:")
    print(f"{'N':<6} {'ln(trees)':<12} {'λ₂':<10}")
    print("-" * 30)

    # Show every 10th result
    for i in range(0, len(results), 10):
        r = results[i]
        print(f"{r['N']:<6} {r['ln_trees']:<12.3f} {r['lambda2_computed']:<10.6f}")

    print(f"\nλ₂ Statistics:")
    print(f"  Min: {min(lambda2_values):.6f} (N={N_values[lambda2_values.index(min(lambda2_values))]})")
    print(f"  Max: {max(lambda2_values):.6f} (N={N_values[lambda2_values.index(max(lambda2_values))]})")
    print(f"  Mean: {np.mean(lambda2_values):.6f}")

    # Check Ramanujan bound (λ₂ ≤ 2√2 ≈ 2.828)
    ramanujan_bound = 2 * np.sqrt(2)
    num_ramanujan = sum(1 for lam in lambda2_values if lam <= ramanujan_bound)
    print(f"\nRamanujan graphs (λ₂ ≤ 2√2 ≈ 2.828): {num_ramanujan}/{len(results)} ({100*num_ramanujan/len(results):.1f}%)")

    return N_values, lambda2_values, ln_trees_values


def create_plots(N_values, lambda2_values, ln_trees_values):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: λ₂ vs N
    ax = axes[0, 0]
    ax.scatter(N_values, lambda2_values, alpha=0.6, s=20)
    ax.axhline(2*np.sqrt(2), color='r', linestyle='--', label='Ramanujan bound (2√2)', linewidth=2)
    ax.set_xlabel('N (number of vertices)')
    ax.set_ylabel('λ₂ (second eigenvalue)')
    ax.set_title('Expansion vs Graph Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: ln(trees) vs N
    ax = axes[0, 1]
    ax.scatter(N_values, ln_trees_values, alpha=0.6, s=20, color='green')

    # Fit linear trend (expected: ln(trees) ≈ c * N)
    coeffs = np.polyfit(N_values, ln_trees_values, 1)
    fit_line = np.poly1d(coeffs)
    ax.plot(N_values, fit_line(N_values), 'r--', label=f'Linear fit: {coeffs[0]:.3f}*N + {coeffs[1]:.1f}')

    ax.set_xlabel('N (number of vertices)')
    ax.set_ylabel('ln(spanning trees)')
    ax.set_title('Spanning Trees vs Graph Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: λ₂ vs ln(trees) (is there a correlation?)
    ax = axes[1, 0]
    ax.scatter(ln_trees_values, lambda2_values, alpha=0.6, s=20, color='purple')
    ax.axhline(2*np.sqrt(2), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('ln(spanning trees)')
    ax.set_ylabel('λ₂')
    ax.set_title('Expansion vs Spanning Trees\n(Trees optimizer - does it also optimize expansion?)')
    ax.grid(True, alpha=0.3)

    # Plot 4: λ₂ distribution histogram
    ax = axes[1, 1]
    ax.hist(lambda2_values, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(2*np.sqrt(2), color='r', linestyle='--', label='Ramanujan bound', linewidth=2)
    ax.set_xlabel('λ₂')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of λ₂ Values')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('planar_sweep_lambda2_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: planar_sweep_lambda2_analysis.png")

    return fig


def save_augmented_results(results):
    """Save results with λ₂ values."""
    output_file = 'planar_sweep_with_lambda2.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved augmented results to: {output_file}")

    # Also create a CSV for easy viewing
    import csv
    csv_file = 'planar_sweep_with_lambda2.csv'

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N', 'n_points', 'ln_trees', 'lambda2_computed'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'N': r['N'],
                'n_points': r['n_points'],
                'ln_trees': r['ln_trees'],
                'lambda2_computed': r['lambda2_computed']
            })

    print(f"✓ Saved CSV to: {csv_file}")


def main():
    print("="*70)
    print("Computing λ₂ for Planar Graph Sweep Results")
    print("="*70)
    print()
    print("The original sweep optimized for SPANNING TREES.")
    print("Now we're computing λ₂ to see if it also gave good EXPANSION!")
    print()

    # Compute λ₂ for all results
    results = compute_lambda2_for_all_sweeps()

    # Analyze
    N_values, lambda2_values, ln_trees_values = analyze_results(results)

    # Create plots
    create_plots(N_values, lambda2_values, ln_trees_values)

    # Save augmented results
    save_augmented_results(results)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
