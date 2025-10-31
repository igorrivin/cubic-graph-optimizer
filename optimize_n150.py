#!/usr/bin/env python3
"""
Optimize N=150 cubic graph using alternating optimization.
"""

import sys
import networkx as nx
import json
from datetime import datetime

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.optimization.alternating import alternating_optimization
from cubic_graph_optimizer.utils.io import save_graph

def optimize_n150():
    """Run alternating optimization for N=150."""
    n = 150

    print(f"Starting alternating optimization for N={n}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Start with a random graph
    print("Generating random 3-regular graph...")
    G = nx.random_regular_graph(3, n, seed=42)
    print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Run alternating optimization (single run, fewer cycles)
    print("\nRunning alternating optimization (single run)...")
    result = alternating_optimization(
        G,
        objectives=['spanning_trees', 'expansion'],
        max_cycles=10,  # Reduced from 20
        max_iterations_per_phase=50,  # Reduced from 100
        verbose=True
    )

    # Save the optimized graph
    final_graph = result['final_graph']
    trajectory = result['trajectory']

    # Get final metrics
    final_state = trajectory[-1]
    ln_trees = final_state['ln_trees']
    lambda2 = final_state['lambda2']

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Final ln(trees): {ln_trees:.6f}")
    print(f"Final λ₂:        {lambda2:.6f}")
    print(f"Converged:       {result['converged']}")
    print(f"Cycled:          {result['cycled']}")
    print(f"Num cycles:      {result['num_cycles']}")

    # Save the graph
    output_file = f"optimized_graphs/n{n}_alternating.pkl"
    print(f"\nSaving graph to: {output_file}")
    save_graph(final_graph, output_file, ln_trees)

    # Save trajectory data
    trajectory_file = f"optimized_graphs/n{n}_alternating_trajectory.json"
    print(f"Saving trajectory to: {trajectory_file}")

    trajectory_data = {
        'n': n,
        'converged': result['converged'],
        'cycled': result['cycled'],
        'num_cycles': result['num_cycles'],
        'trajectory': trajectory
    }

    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result

if __name__ == "__main__":
    result = optimize_n150()
