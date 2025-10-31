#!/usr/bin/env python3
"""
Generate the planar expansion challenge dataset.

Creates 10 graphs:
- Graph 0: Our champion (optimized for expansion)
- Graphs 1-9: Random planar cubic graphs with N=150

Output: challenge_dataset.json
"""

import json
import sys
import numpy as np

sys.path.insert(0, '/home/igor/devel/cubic-graph-optimizer')

from cubic_graph_optimizer.planar.planar_ops import random_planar_cubic_from_sphere
from cubic_graph_optimizer.core.spanning_trees import get_second_eigenvalue
import networkx as nx


def graph_to_dict(G, label, metadata=None):
    """Convert graph to dictionary format."""
    # Adjacency list
    adj_list = [sorted(list(G.neighbors(i))) for i in range(G.number_of_nodes())]

    # Compute properties
    lambda2_A = get_second_eigenvalue(G)
    lambda2_L = 3.0 - lambda2_A  # For cubic graphs

    # Verify planarity
    is_planar = nx.is_planar(G)
    is_cubic = all(d == 3 for _, d in G.degree())

    graph_dict = {
        'label': label,
        'n_vertices': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'adjacency_list': adj_list,
        'properties': {
            'is_planar': is_planar,
            'is_cubic': is_cubic,
            'is_connected': nx.is_connected(G),
            'lambda2_adjacency': float(lambda2_A),
            'lambda2_laplacian': float(lambda2_L),
            'algebraic_connectivity': float(lambda2_L)
        }
    }

    if metadata:
        graph_dict['metadata'] = metadata

    return graph_dict


def load_champion():
    """Load our optimized champion graph."""
    # Load from the expansion optimization result
    import pickle

    # Try multiple possible file names (from different runs)
    possible_files = [
        'optimized_graphs/planar_n150_champion.pkl',  # Our new champion
        'optimized_graphs/planar_n150_best_expansion.pkl',
        'optimized_graphs/planar_n150_expansion.pkl',
        'optimized_graphs/planar_n150_expansion_triangulation.pkl',
    ]

    for pkl_file in possible_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Handle both direct Graph objects and dictionaries
            if isinstance(data, nx.Graph):
                G = data
            elif isinstance(data, dict):
                if 'graph' in data:
                    G = data['graph']
                elif 'dual_graph' in data:
                    G = data['dual_graph']
                else:
                    print(f"Warning: Could not extract graph from {pkl_file}")
                    continue
            else:
                print(f"Warning: Unknown data type in {pkl_file}: {type(data)}")
                continue

            print(f"✓ Loaded champion from {pkl_file}")
            print(f"  N = {G.number_of_nodes()}")
            print(f"  λ₂(L) = {3.0 - get_second_eigenvalue(G):.6f}")

            return G

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Warning: Error loading {pkl_file}: {e}")
            continue

    print(f"Warning: Could not load from any pickle file, will try backup")
    return None


def load_champion_from_json():
    """Backup: load champion from JSON result."""
    # Try multiple possible JSON files
    possible_files = [
        'optimized_graphs/planar_n150_expansion_triangulation_full.json',
        'optimized_graphs/planar_n150_trees_triangulation_full.json',
    ]

    for json_file in possible_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Try different adjacency list field names
            if 'dual_adjacency' in data:
                adj_list = data['dual_adjacency']
            elif 'adjacency_list' in data:
                adj_list = data['adjacency_list']
            elif 'adjacency' in data:
                adj_list = data['adjacency']
            else:
                print(f"Warning: Could not find adjacency list in {json_file}")
                continue

            G = nx.Graph()
            G.add_nodes_from(range(len(adj_list)))

            for node, neighbors in enumerate(adj_list):
                for neighbor in neighbors:
                    if node < neighbor:
                        G.add_edge(node, neighbor)

            print(f"✓ Loaded champion from {json_file}")
            print(f"  N = {G.number_of_nodes()}")
            print(f"  λ₂(L) = {3.0 - get_second_eigenvalue(G):.6f}")

            return G

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Warning: Error loading {json_file}: {e}")
            continue

    print(f"Error: Could not find any champion graph files!")
    return None


def generate_random_graphs(n_points=77, count=9, seed_start=100):
    """Generate random planar cubic graphs."""
    graphs = []

    print(f"\nGenerating {count} random planar cubic graphs...")

    for i in range(count):
        seed = seed_start + i
        print(f"  Graph {i+1}/{count} (seed={seed})...", end=' ', flush=True)

        G = random_planar_cubic_from_sphere(n_points, seed=seed)

        lambda2_L = 3.0 - get_second_eigenvalue(G)
        print(f"N={G.number_of_nodes()}, λ₂(L)={lambda2_L:.6f}")

        graphs.append(G)

    return graphs


def main():
    print("="*70)
    print("PLANAR EXPANSION CHALLENGE DATASET GENERATOR")
    print("="*70)

    # Load champion
    print("\nLoading champion graph...")
    champion = load_champion()

    if champion is None:
        champion = load_champion_from_json()

    if champion is None:
        print("ERROR: Could not load champion graph!")
        sys.exit(1)

    # Generate random graphs
    random_graphs = generate_random_graphs(n_points=77, count=9, seed_start=200)

    # Create dataset
    dataset = {
        'challenge': 'Planar Cubic Graph Expansion Competition',
        'task': 'Find planar cubic graph on 150 vertices with maximum algebraic connectivity λ₂(L)',
        'rules': {
            'n_vertices': 150,
            'must_be_planar': True,
            'must_be_cubic': True,
            'must_be_connected': True,
            'objective': 'Maximize λ₂(Laplacian) = algebraic connectivity',
            'equivalently': 'Minimize λ₂(Adjacency) for 3-regular graphs',
            'relationship': 'λ₂(L) = 3 - λ₂(A) for cubic graphs'
        },
        'theoretical_bounds': {
            'spielman_teng_upper_bound': 0.16,
            'explanation': 'For cubic planar graphs: λ₂(L) ≤ 8Δ/n = 24/150 = 0.16',
            'note': 'This bound may not be tight; optimal value likely between 0.08 and 0.16'
        },
        'champion_threshold': 0.085,
        'graphs': []
    }

    # Add champion (index 0)
    champion_dict = graph_to_dict(
        champion,
        label='champion_optimized',
        metadata={
            'method': 'Triangulation-based diagonal flip optimization',
            'objective': 'expansion (minimize λ₂ of dual cubic graph)',
            'algorithm': 'Multi-restart parallel optimization with 60 restarts, 150 iterations',
            'computation_time': '~5-7 seconds (with fast eigenvalue computation)'
        }
    )
    dataset['graphs'].append(champion_dict)

    # Add random graphs (indices 1-9)
    for i, G in enumerate(random_graphs):
        random_dict = graph_to_dict(
            G,
            label=f'random_{i+1}',
            metadata={
                'method': 'Random sphere construction',
                'seed': 200 + i
            }
        )
        dataset['graphs'].append(random_dict)

    # Save dataset
    output_file = 'planar_expansion_challenge_dataset.json'

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DATASET CREATED")
    print(f"{'='*70}")
    print(f"Output: {output_file}")
    print(f"Total graphs: {len(dataset['graphs'])}")
    print()

    # Summary statistics
    print("Graph Summary:")
    print(f"{'Index':<7} {'Label':<25} {'λ₂(L)':<12} {'λ₂(A)':<12}")
    print("-"*70)

    for i, g in enumerate(dataset['graphs']):
        props = g['properties']
        print(f"{i:<7} {g['label']:<25} {props['lambda2_laplacian']:<12.6f} {props['lambda2_adjacency']:<12.6f}")

    # Find best
    best_idx = max(range(len(dataset['graphs'])),
                   key=lambda i: dataset['graphs'][i]['properties']['lambda2_laplacian'])

    best_lambda2_L = dataset['graphs'][best_idx]['properties']['lambda2_laplacian']
    best_lambda2_A = dataset['graphs'][best_idx]['properties']['lambda2_adjacency']

    print()
    print(f"Best graph: Index {best_idx} ({dataset['graphs'][best_idx]['label']})")
    print(f"  λ₂(L) = {best_lambda2_L:.6f} (algebraic connectivity)")
    print(f"  λ₂(A) = {best_lambda2_A:.6f} (adjacency eigenvalue)")
    print(f"  Fraction of Spielman-Teng bound: {best_lambda2_L/0.16*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"To verify a graph:")
    print(f"  python planar_expansion_checker.py {output_file} --graph-index 0")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
