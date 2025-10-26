#!/usr/bin/env python3
"""
Cubic Graph Optimizer

Main script for optimizing spanning trees in cubic graphs using Whitehead flips.
"""

import networkx as nx
import numpy as np

from core.spanning_trees import count_spanning_trees
from optimization.methods import (
    gradient_ascent_greedy,
    gradient_ascent_first_improvement,
    simulated_annealing,
    optimize_with_restarts,
    run_optimization_sweep
)
from analysis.properties import analyze_graph_properties, compare_graph_structures
from analysis.spectral import analyze_eigenvalue_distribution
from utils.io import save_graph, load_graph
from utils.special_graphs import check_known_cages


def main():
    """Main function with example usage."""
    
    # Example 1: Single k value optimization
    print("=" * 70)
    print("EXAMPLE 1: Single k value optimization")
    print("=" * 70)
    
    k = 5  # For 2k = 10 vertices (Petersen graph)
    n = 2 * k
    
    print(f"\nOptimizing cubic graph with {n} vertices\n")
    
    # For large graphs (k >= 30), use lightweight mode
    lightweight_mode = (k >= 30)
    
    # Generate random starting graph
    G = nx.random_regular_graph(3, n, seed=42)
    
    print("=" * 50)
    print("Method 1: Greedy (best improvement each step)")
    print("=" * 50)
    G_greedy = G.copy()
    G_optimized_greedy, final_value_greedy = gradient_ascent_greedy(G_greedy, max_iterations=100)
    
    print("\n" + "=" * 50)
    print("Method 2: First improvement (faster)")
    print("=" * 50)
    G_first = G.copy()
    G_optimized_first, final_value_first = gradient_ascent_first_improvement(G_first, max_iterations=100)
    
    print("\n" + "=" * 50)
    print("Method 3: Simulated Annealing")
    print("=" * 50)
    G_sa = G.copy()
    G_optimized_sa, final_value_sa = simulated_annealing(
        G_sa, 
        max_iterations=500,
        T0=None,  # Auto-calibrate
        cooling_rate=0.97,
        adaptive=True
    )
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Initial T: {count_spanning_trees(G):.6f}")
    print(f"Greedy final T: {final_value_greedy:.6f}")
    print(f"First-improvement final T: {final_value_first:.6f}")
    print(f"Simulated annealing final T: {final_value_sa:.6f}")
    print(f"\nNote: T(n,d) = ln(number of spanning trees)")
    
    # Analyze structural properties of best solution
    best_method = max([("Greedy", G_optimized_greedy, final_value_greedy),
                       ("First", G_optimized_first, final_value_first),
                       ("SA", G_optimized_sa, final_value_sa)],
                      key=lambda x: x[2])
    
    print(f"\n\nBest method: {best_method[0]}")
    
    # For large graphs, skip expensive computations
    if lightweight_mode:
        print("\n(Using lightweight mode for large graph - skipping girth/cycle stats)")
    
    compare_graph_structures(G, best_method[1], compute_expensive=not lightweight_mode)
    
    # Save the best graph
    save_graph(best_method[1], f"optimal_cubic_k{k}.pkl", best_method[2])
    
    print(f"\n{'='*60}")
    print(f"RESULT FOR T(2k, 3) with k={k}")
    print(f"{'='*60}")
    print(f"ln(spanning trees) = {best_method[2]:.6f}")
    print(f"Number of spanning trees ≈ {np.exp(best_method[2]):.6e}")
    
    # Check against known cages
    is_known, cage_name = check_known_cages(best_method[1], k)
    if is_known:
        print(f"\nGraph is isomorphic to: {cage_name}")
    
    # Example 2: Sweep over multiple k values with restarts
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Multiple k values with restarts")
    print("=" * 70)
    print("Uncomment the code below to run a sweep over multiple k values")
    
    # Uncomment to run:
    # results = run_optimization_sweep(
    #     k_values=range(5, 16),  # k from 5 to 15
    #     methods=['greedy', 'first'],  # Skip SA for speed
    #     restarts=10,  # Try 10 different random starting graphs per method
    #     parallel=True,  # Use all CPU cores
    #     save_graphs=True,
    #     verbose=True
    # )
    
    # Example 3: Auto restarts (runs until no improvement)
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Optimization with auto restarts")
    print("=" * 70)
    print("Uncomment the code below to run with automatic stopping")
    
    # results = run_optimization_sweep(
    #     k_values=[20, 25, 30],
    #     methods=['greedy'],
    #     restarts='auto',  # Automatic stopping
    #     parallel=True,
    #     save_graphs=True
    # )
    
    # Example 4: Single optimization with restarts
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Single optimization with multiple restarts")
    print("=" * 70)
    print("Uncomment the code below to optimize a single graph with restarts")
    
    # G_opt, T_opt, num_trials = optimize_with_restarts(
    #     n=26,  # k=13
    #     method='greedy',
    #     restarts=20,
    #     parallel=True,
    #     verbose=True
    # )
    # print(f"Best of {num_trials} trials: T = {T_opt:.6f}")
    
    # Example 5: Analyze eigenvalue distributions
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Eigenvalue distribution analysis")
    print("=" * 70)
    print("Uncomment the code below to analyze eigenvalue distributions")
    
    # eigenvalue_results = analyze_eigenvalue_distribution(
    #     k_values=[5, 7, 10, 12, 15, 20],
    #     samples_per_k=100,  # Sample 100 random graphs per k
    #     include_optimized=True,
    #     optimization_method='greedy',
    #     plot_histograms=True  # Requires matplotlib
    # )


if __name__ == "__main__":
    main()