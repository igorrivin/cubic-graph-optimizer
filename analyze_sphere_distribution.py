#!/usr/bin/env python3
"""
Analyze the distribution of spanning trees and λ₂ for sphere-based
planar cubic graph construction.

Tests whether ln(trees) and λ₂(L) are concentrated for the special
geometric family of graphs arising as duals to Delaunay triangulations
on the sphere.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.planar.planar_ops import random_planar_cubic_from_sphere
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue

def sample_distribution(n_points=77, n_samples=500, seed_start=1000):
    """Generate many random sphere-based graphs and measure properties."""

    results = []

    print(f"Generating {n_samples} random sphere-based graphs...")
    print(f"Target: N ≈ 150 vertices (n_points={n_points})")
    print()

    for i in tqdm(range(n_samples)):
        seed = seed_start + i

        try:
            G = random_planar_cubic_from_sphere(n_points, seed=seed)

            n = G.number_of_nodes()
            ln_trees = count_spanning_trees(G)
            lambda2_A = get_second_eigenvalue(G)
            lambda2_L = 3.0 - lambda2_A

            results.append({
                'seed': seed,
                'n': n,
                'ln_trees': ln_trees,
                'lambda2_A': lambda2_A,
                'lambda2_L': lambda2_L
            })

        except Exception as e:
            print(f"\nError with seed {seed}: {e}")
            continue

    return results


def analyze_and_plot(results):
    """Compute statistics and create visualizations."""

    # Extract data
    ln_trees = np.array([r['ln_trees'] for r in results])
    lambda2_L = np.array([r['lambda2_L'] for r in results])
    n_values = np.array([r['n'] for r in results])

    # Filter to exactly N=150 (if we have enough)
    mask_150 = (n_values == 150)
    n_150 = np.sum(mask_150)

    if n_150 > 10:
        ln_trees_150 = ln_trees[mask_150]
        lambda2_L_150 = lambda2_L[mask_150]
        print(f"\n{n_150} graphs with exactly N=150")
    else:
        # Use all if we don't have enough exactly at 150
        ln_trees_150 = ln_trees
        lambda2_L_150 = lambda2_L
        print(f"\nUsing all {len(results)} graphs (N range: {n_values.min()}-{n_values.max()})")

    # Statistics
    print()
    print("="*70)
    print("DISTRIBUTION STATISTICS FOR SPHERE-BASED CONSTRUCTION")
    print("="*70)
    print()

    print("ln(trees):")
    print(f"  Mean: {ln_trees_150.mean():.6f}")
    print(f"  Std:  {ln_trees_150.std():.6f}")
    print(f"  CV:   {ln_trees_150.std() / ln_trees_150.mean() * 100:.3f}%  (coefficient of variation)")
    print(f"  Min:  {ln_trees_150.min():.6f}")
    print(f"  Q1:   {np.percentile(ln_trees_150, 25):.6f}")
    print(f"  Median: {np.median(ln_trees_150):.6f}")
    print(f"  Q3:   {np.percentile(ln_trees_150, 75):.6f}")
    print(f"  Max:  {ln_trees_150.max():.6f}")
    print()

    print("λ₂(L) - Algebraic connectivity:")
    print(f"  Mean: {lambda2_L_150.mean():.6f}")
    print(f"  Std:  {lambda2_L_150.std():.6f}")
    print(f"  CV:   {lambda2_L_150.std() / lambda2_L_150.mean() * 100:.3f}%")
    print(f"  Min:  {lambda2_L_150.min():.6f}")
    print(f"  Q1:   {np.percentile(lambda2_L_150, 25):.6f}")
    print(f"  Median: {np.median(lambda2_L_150):.6f}")
    print(f"  Q3:   {np.percentile(lambda2_L_150, 75):.6f}")
    print(f"  Max:  {lambda2_L_150.max():.6f}")
    print()

    # Reference values
    print("="*70)
    print("COMPARISON WITH OPTIMIZED GRAPHS")
    print("="*70)
    print()

    # Our optimized values
    trees_opt_ln = 119.713
    trees_opt_lambda = 0.0840
    expansion_opt_ln = 117.206
    expansion_opt_lambda = 0.0845
    sweep_champion_ln = 119.716
    sweep_champion_lambda = 0.0884

    # Z-scores
    z_trees_ln = (trees_opt_ln - ln_trees_150.mean()) / ln_trees_150.std()
    z_trees_lambda = (trees_opt_lambda - lambda2_L_150.mean()) / lambda2_L_150.std()
    z_expansion_ln = (expansion_opt_ln - ln_trees_150.mean()) / ln_trees_150.std()
    z_expansion_lambda = (expansion_opt_lambda - lambda2_L_150.mean()) / lambda2_L_150.std()
    z_sweep_ln = (sweep_champion_ln - ln_trees_150.mean()) / ln_trees_150.std()
    z_sweep_lambda = (sweep_champion_lambda - lambda2_L_150.mean()) / lambda2_L_150.std()

    print(f"Trees optimizer (ln={trees_opt_ln:.3f}, λ₂={trees_opt_lambda:.4f}):")
    print(f"  Z-score(ln): {z_trees_ln:+.2f}σ")
    print(f"  Z-score(λ₂): {z_trees_lambda:+.2f}σ")
    print()

    print(f"Expansion optimizer (ln={expansion_opt_ln:.3f}, λ₂={expansion_opt_lambda:.4f}):")
    print(f"  Z-score(ln): {z_expansion_ln:+.2f}σ")
    print(f"  Z-score(λ₂): {z_expansion_lambda:+.2f}σ")
    print()

    print(f"Sweep champion (ln={sweep_champion_ln:.3f}, λ₂={sweep_champion_lambda:.4f}):")
    print(f"  Z-score(ln): {z_sweep_ln:+.2f}σ")
    print(f"  Z-score(λ₂): {z_sweep_lambda:+.2f}σ")
    print()

    # Percentiles
    pct_sweep_ln = np.sum(ln_trees_150 <= sweep_champion_ln) / len(ln_trees_150) * 100
    pct_sweep_lambda = np.sum(lambda2_L_150 >= sweep_champion_lambda) / len(lambda2_L_150) * 100

    print(f"Sweep champion percentiles:")
    print(f"  ln(trees): {pct_sweep_ln:.1f}th percentile (top {100-pct_sweep_ln:.1f}%)")
    print(f"  λ₂(L): {100-pct_sweep_lambda:.1f}th percentile (top {pct_sweep_lambda:.1f}%)")
    print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ln(trees) histogram
    ax = axes[0, 0]
    ax.hist(ln_trees_150, bins=30, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(ln_trees_150.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(trees_opt_ln, color='green', linestyle='-', linewidth=2, label='Trees opt')
    ax.axvline(expansion_opt_ln, color='orange', linestyle='-', linewidth=2, label='Expansion opt')
    ax.axvline(sweep_champion_ln, color='red', linestyle='-', linewidth=2, label='Sweep champion')
    ax.set_xlabel('ln(trees)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of ln(spanning trees)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # λ₂(L) histogram
    ax = axes[0, 1]
    ax.hist(lambda2_L_150, bins=30, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(lambda2_L_150.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(trees_opt_lambda, color='green', linestyle='-', linewidth=2, label='Trees opt')
    ax.axvline(expansion_opt_lambda, color='orange', linestyle='-', linewidth=2, label='Expansion opt')
    ax.axvline(sweep_champion_lambda, color='red', linestyle='-', linewidth=2, label='Sweep champion')
    ax.set_xlabel('λ₂(L) - Algebraic connectivity')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of λ₂(Laplacian)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[1, 0]
    ax.scatter(ln_trees_150, lambda2_L_150, alpha=0.5, s=20)
    ax.scatter([trees_opt_ln], [trees_opt_lambda], color='green', s=200, marker='*',
               label='Trees opt', zorder=10, edgecolors='black', linewidths=1.5)
    ax.scatter([expansion_opt_ln], [expansion_opt_lambda], color='orange', s=200, marker='*',
               label='Expansion opt', zorder=10, edgecolors='black', linewidths=1.5)
    ax.scatter([sweep_champion_ln], [sweep_champion_lambda], color='red', s=300, marker='*',
               label='Sweep champion', zorder=10, edgecolors='black', linewidths=2)
    ax.set_xlabel('ln(trees)')
    ax.set_ylabel('λ₂(L)')
    ax.set_title('Correlation: ln(trees) vs λ₂(L)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation
    corr = np.corrcoef(ln_trees_150, lambda2_L_150)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Q-Q plots for normality check
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(lambda2_L_150, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: λ₂(L) vs Normal Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sphere_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: sphere_distribution_analysis.png")

    return {
        'ln_trees_mean': float(ln_trees_150.mean()),
        'ln_trees_std': float(ln_trees_150.std()),
        'ln_trees_cv': float(ln_trees_150.std() / ln_trees_150.mean() * 100),
        'lambda2_L_mean': float(lambda2_L_150.mean()),
        'lambda2_L_std': float(lambda2_L_150.std()),
        'lambda2_L_cv': float(lambda2_L_150.std() / lambda2_L_150.mean() * 100),
        'correlation': float(corr),
        'n_samples': len(ln_trees_150)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sphere construction distribution')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of random samples (default: 500)')
    parser.add_argument('--n-points', type=int, default=77,
                       help='Number of sphere points (default: 77 → N≈150)')
    parser.add_argument('--seed-start', type=int, default=1000,
                       help='Starting seed (default: 1000)')

    args = parser.parse_args()

    # Sample distribution
    results = sample_distribution(
        n_points=args.n_points,
        n_samples=args.n_samples,
        seed_start=args.seed_start
    )

    # Save raw data
    with open('sphere_distribution_samples.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw data saved to: sphere_distribution_samples.json")

    # Analyze
    stats_summary = analyze_and_plot(results)

    with open('sphere_distribution_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Statistics saved to: sphere_distribution_stats.json")

    print()
    print("="*70)
    print("CONCLUSION:")
    print("="*70)
    if stats_summary['ln_trees_cv'] < 1.0 and stats_summary['lambda2_L_cv'] < 10.0:
        print("✓ CONCENTRATED: Both ln(trees) and λ₂(L) show low variance")
        print("  The sphere construction produces a geometrically privileged family")
        print("  with consistent spectral properties.")
    else:
        print("✗ NOT CONCENTRATED: High variance observed")


if __name__ == '__main__':
    main()
