#!/usr/bin/env python3
"""
Standalone checker for planar cubic graph expansion challenge.

Verifies:
1. Graph is planar
2. Graph is cubic (3-regular)
3. Graph has exactly 150 vertices
4. Computes λ₂(A) (adjacency) and λ₂(L) (Laplacian)
5. Checks if λ₂(L) > threshold (better expansion = higher algebraic connectivity)

Usage:
    python planar_expansion_checker.py <graph.json> [--threshold 0.085]
"""

import json
import sys
import argparse
import networkx as nx
import numpy as np


def load_graph_from_json(filename, graph_index=None):
    """Load graph from JSON adjacency list format.

    Args:
        filename: Path to JSON file
        graph_index: If provided, load from dataset['graphs'][graph_index]
                    Otherwise load from root-level adjacency list
    """
    with open(filename) as f:
        data = json.load(f)

    # Check if this is a dataset with multiple graphs
    if 'graphs' in data:
        if graph_index is None:
            raise ValueError("File contains multiple graphs. Use --graph-index to select one.")
        if graph_index < 0 or graph_index >= len(data['graphs']):
            raise ValueError(f"Graph index {graph_index} out of range (0-{len(data['graphs'])-1})")
        graph_data = data['graphs'][graph_index]
        adj_list = graph_data['adjacency_list']
    elif 'adjacency_list' in data:
        adj_list = data['adjacency_list']
    elif 'adjacency' in data:
        adj_list = data['adjacency']
    else:
        raise ValueError("JSON must contain 'adjacency_list', 'adjacency', or 'graphs' field")

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(adj_list)))

    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            if node < neighbor:  # Add each edge once
                G.add_edge(node, neighbor)

    return G


def verify_graph(G, verbose=True):
    """
    Verify graph properties.

    Returns:
        dict: Verification results
    """
    results = {}

    # Check number of vertices
    n = G.number_of_nodes()
    results['n_vertices'] = n
    results['correct_size'] = (n == 150)

    if verbose:
        print(f"Vertices: {n} {'✓' if n == 150 else '✗ (expected 150)'}")

    # Check if cubic (3-regular)
    degrees = dict(G.degree())
    is_cubic = all(d == 3 for d in degrees.values())
    results['is_cubic'] = is_cubic
    results['degrees'] = list(degrees.values())

    if verbose:
        if is_cubic:
            print(f"Cubic (3-regular): ✓")
        else:
            non_cubic = [(v, d) for v, d in degrees.items() if d != 3]
            print(f"Cubic (3-regular): ✗")
            print(f"  Non-cubic vertices: {non_cubic[:5]}...")

    # Check if planar
    is_planar = nx.is_planar(G)
    results['is_planar'] = is_planar

    if verbose:
        print(f"Planar: {'✓' if is_planar else '✗'}")

    # Check if connected
    is_connected = nx.is_connected(G)
    results['is_connected'] = is_connected

    if verbose:
        print(f"Connected: {'✓' if is_connected else '✗'}")

    return results


def compute_eigenvalues(G, verbose=True):
    """
    Compute spectral properties.

    Returns:
        dict: Eigenvalue results
    """
    results = {}

    # Adjacency matrix eigenvalues
    A = nx.adjacency_matrix(G).toarray()
    evals_A = np.linalg.eigvalsh(A)  # Sorted ascending
    lambda1_A = evals_A[-1]  # Largest
    lambda2_A = evals_A[-2]  # Second largest

    results['lambda1_adjacency'] = float(lambda1_A)
    results['lambda2_adjacency'] = float(lambda2_A)
    results['spectral_gap_adjacency'] = float(lambda1_A - lambda2_A)

    # Laplacian eigenvalues
    L = nx.laplacian_matrix(G).toarray()
    evals_L = np.linalg.eigvalsh(L)  # Sorted ascending
    lambda1_L = evals_L[0]  # Smallest (always 0 for connected)
    lambda2_L = evals_L[1]  # Algebraic connectivity

    results['lambda1_laplacian'] = float(lambda1_L)
    results['lambda2_laplacian'] = float(lambda2_L)
    results['algebraic_connectivity'] = float(lambda2_L)

    # Verify relationship for cubic graphs: λ₂(L) = 3 - λ₂(A)
    expected_lambda2_L = 3 - lambda2_A
    results['laplacian_adjacency_relationship_error'] = float(abs(lambda2_L - expected_lambda2_L))

    if verbose:
        print(f"\nSpectral Properties:")
        print(f"  Adjacency matrix:")
        print(f"    λ₁(A) = {lambda1_A:.6f} (max eigenvalue)")
        print(f"    λ₂(A) = {lambda2_A:.6f} (second eigenvalue)")
        print(f"    Spectral gap = {lambda1_A - lambda2_A:.6f}")
        print(f"  Laplacian matrix:")
        print(f"    λ₁(L) = {lambda1_L:.6f} (should be ~0)")
        print(f"    λ₂(L) = {lambda2_L:.6f} (algebraic connectivity)")
        print(f"  Verification: λ₂(L) = 3 - λ₂(A) = {expected_lambda2_L:.6f}")
        print(f"    Error: {abs(lambda2_L - expected_lambda2_L):.9f}")

    return results


def check_threshold(results, threshold, verbose=True):
    """Check if graph beats the threshold."""
    lambda2_L = results['lambda2_laplacian']
    passes = lambda2_L >= threshold

    # Spielman-Teng bound for cubic planar: λ₂(L) ≤ 8Δ/n = 24/150 = 0.16
    st_bound = 24.0 / 150.0
    fraction_of_bound = lambda2_L / st_bound

    if verbose:
        print(f"\nThreshold Check:")
        print(f"  Target: λ₂(L) ≥ {threshold}")
        print(f"  Actual: λ₂(L) = {lambda2_L:.6f}")
        print(f"  Result: {'✓ PASS' if passes else '✗ FAIL'}")
        print(f"\nSpielman-Teng Bound:")
        print(f"  Upper bound: λ₂(L) ≤ {st_bound:.6f}")
        print(f"  Our result: {fraction_of_bound*100:.1f}% of bound")

    results['threshold'] = threshold
    results['passes_threshold'] = passes
    results['spielman_teng_bound'] = st_bound
    results['fraction_of_bound'] = fraction_of_bound

    return passes


def main():
    parser = argparse.ArgumentParser(
        description='Verify planar cubic graph expansion properties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python planar_expansion_checker.py my_graph.json
  python planar_expansion_checker.py my_graph.json --threshold 0.085
  python planar_expansion_checker.py my_graph.json --json-output results.json

The graph JSON should contain an "adjacency_list" field with a list of neighbor lists.
        """
    )
    parser.add_argument('graph_file', help='JSON file containing graph adjacency list')
    parser.add_argument('--graph-index', type=int, default=None,
                       help='Index of graph to check (for dataset files with multiple graphs)')
    parser.add_argument('--threshold', type=float, default=0.085,
                       help='Minimum λ₂(L) threshold (default: 0.085)')
    parser.add_argument('--json-output', help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("PLANAR CUBIC GRAPH EXPANSION CHECKER")
        print("="*70)
        print(f"Graph file: {args.graph_file}")
        print()

    # Load graph
    try:
        G = load_graph_from_json(args.graph_file, args.graph_index)
    except Exception as e:
        print(f"ERROR loading graph: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify structure
    verification = verify_graph(G, verbose=verbose)

    # Check all basic properties
    if not verification['correct_size']:
        print(f"\n✗ INVALID: Graph must have exactly 150 vertices", file=sys.stderr)
        sys.exit(1)

    if not verification['is_cubic']:
        print(f"\n✗ INVALID: Graph must be cubic (3-regular)", file=sys.stderr)
        sys.exit(1)

    if not verification['is_planar']:
        print(f"\n✗ INVALID: Graph must be planar", file=sys.stderr)
        sys.exit(1)

    if not verification['is_connected']:
        print(f"\n✗ INVALID: Graph must be connected", file=sys.stderr)
        sys.exit(1)

    # Compute eigenvalues
    spectral = compute_eigenvalues(G, verbose=verbose)

    # Check threshold
    passes = check_threshold(spectral, args.threshold, verbose=verbose)

    # Combine results
    full_results = {
        'graph_file': args.graph_file,
        'verification': verification,
        'spectral': spectral,
        'valid': verification['correct_size'] and verification['is_cubic'] and
                 verification['is_planar'] and verification['is_connected'],
        'passes_threshold': passes
    }

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(full_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {args.json_output}")

    # Final summary
    if verbose:
        print("\n" + "="*70)
        if full_results['valid'] and passes:
            print("✓ VALID GRAPH - PASSES THRESHOLD")
        elif full_results['valid']:
            print("✓ VALID GRAPH - FAILS THRESHOLD")
        else:
            print("✗ INVALID GRAPH")
        print("="*70)

    # Exit code
    sys.exit(0 if (full_results['valid'] and passes) else 1)


if __name__ == "__main__":
    main()
