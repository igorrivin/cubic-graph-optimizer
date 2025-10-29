#!/usr/bin/env python3
"""
Standalone Planar Cubic Graph Checker

This script verifies planar cubic graphs and checks if they exceed a spanning tree threshold.

Requirements:
    pip install networkx numpy scipy

Usage:
    python planar_graph_checker.py test_file.json

Input JSON format:
{
    "threshold": 31.28,
    "expected_results": [true, false, ...],
    "graphs": [
        {
            "name": "graph_name",
            "adjacency_list": [[1, 2, 3], [0, 4, 5], ...]
        },
        ...
    ]
}

Returns:
    - Checks each graph is cubic, planar, and connected
    - Computes spanning tree count via Matrix-Tree Theorem
    - Compares to threshold
    - Validates against expected results
"""

import json
import sys
import numpy as np
import networkx as nx
from typing import List, Tuple


def count_spanning_trees(G: nx.Graph) -> float:
    """
    Count spanning trees using Kirchhoff's Matrix-Tree Theorem.

    Returns ln(tau) where tau is the number of spanning trees.
    """
    if not nx.is_connected(G):
        return -np.inf

    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Remove one row and column (doesn't matter which)
    L_reduced = L[:-1, :-1]

    # Compute determinant via log to avoid overflow
    # det(L_reduced) = tau (number of spanning trees)
    sign, logdet = np.linalg.slogdet(L_reduced)

    if sign <= 0:
        return -np.inf

    return logdet


def check_graph(adjacency_list: List[List[int]]) -> Tuple[bool, dict]:
    """
    Check if graph is valid (cubic, planar, connected) and compute ln(trees).

    Returns:
        (is_valid, info_dict)
    """
    info = {}

    # Build graph
    G = nx.Graph()
    n = len(adjacency_list)

    for v, neighbors in enumerate(adjacency_list):
        for u in neighbors:
            G.add_edge(v, u)

    # Check basic properties
    info['vertices'] = G.number_of_nodes()
    info['edges'] = G.number_of_edges()

    # Check cubic (all vertices degree 3)
    degrees = [G.degree(v) for v in range(n)]
    info['min_degree'] = min(degrees) if degrees else 0
    info['max_degree'] = max(degrees) if degrees else 0
    info['is_cubic'] = all(d == 3 for d in degrees)

    # Check connected
    info['is_connected'] = nx.is_connected(G)

    # Check planar
    info['is_planar'] = nx.check_planarity(G)[0]

    # All checks must pass
    is_valid = info['is_cubic'] and info['is_connected'] and info['is_planar']

    if is_valid:
        # Compute spanning trees
        info['ln_trees'] = count_spanning_trees(G)
        info['trees'] = np.exp(info['ln_trees'])
    else:
        info['ln_trees'] = None
        info['trees'] = None

    return is_valid, info


def run_test_suite(test_file: str, threshold: float = None, verbose: bool = True) -> dict:
    """
    Run test suite from JSON file.

    Returns summary of results.
    """
    # Load test file
    with open(test_file) as f:
        data = json.load(f)

    # Handle two formats: {threshold, graphs} or just [graphs]
    if isinstance(data, dict):
        threshold = data.get('threshold', threshold)
        expected_results = data.get('expected_results', [])
        graphs = data['graphs']
    else:
        # data is a list of graphs
        expected_results = []
        graphs = data

    if verbose:
        print('='*70)
        print('PLANAR CUBIC GRAPH CHECKER')
        print('='*70)
        print(f'Test file: {test_file}')
        if threshold is not None:
            print(f'Threshold: ln(trees) > {threshold}')
        else:
            print('Threshold: None (will just verify graphs)')
        print(f'Graphs: {len(graphs)}')
        print()

    results = []
    passed = 0
    failed = 0

    for i, graph_data in enumerate(graphs):
        name = graph_data.get('name', f'graph_{i}')
        adj_list = graph_data['adjacency_list']

        if verbose:
            print(f'[{i+1}/{len(graphs)}] {name}:')

        # Check validity
        is_valid, info = check_graph(adj_list)

        if not is_valid:
            if verbose:
                print(f'  ❌ INVALID GRAPH')
                if not info['is_cubic']:
                    print(f'     Not cubic: degree range [{info["min_degree"]}, {info["max_degree"]}]')
                if not info['is_planar']:
                    print(f'     Not planar')
                if not info['is_connected']:
                    print(f'     Not connected')
                print()

            results.append({
                'name': name,
                'valid': False,
                'passes_threshold': False,
                'info': info,
            })
            failed += 1
            continue

        # Check threshold
        ln_trees = info['ln_trees']
        passes = ln_trees > threshold if threshold is not None else True

        if verbose:
            if threshold is not None:
                status = '✓' if passes else '✗'
                print(f'  {status} ln(trees) = {ln_trees:.6f} ({"PASS" if passes else "FAIL"})')
            else:
                print(f'  ✓ ln(trees) = {ln_trees:.6f}')
            print(f'     trees = {info["trees"]:.3e}')
            print(f'     vertices = {info["vertices"]}, edges = {info["edges"]}')
            print()

        results.append({
            'name': name,
            'valid': True,
            'ln_trees': ln_trees,
            'passes_threshold': passes,
            'info': info,
        })

        if passes:
            passed += 1
        else:
            failed += 1

    # Check against expected results
    if expected_results:
        matches = sum(1 for i, r in enumerate(results)
                     if r['passes_threshold'] == expected_results[i])

        if verbose:
            print('='*70)
            print('VALIDATION AGAINST EXPECTED RESULTS')
            print('='*70)
            print(f'Matched: {matches}/{len(expected_results)}')

            if matches == len(expected_results):
                print('✓ All results match expectations!')
            else:
                print('✗ Some results do not match:')
                for i, (result, expected) in enumerate(zip(results, expected_results)):
                    if result['passes_threshold'] != expected:
                        print(f'  Graph {i} ({result["name"]}): '
                              f'got {result["passes_threshold"]}, expected {expected}')
            print()

    # Summary
    if verbose:
        print('='*70)
        print('SUMMARY')
        print('='*70)
        print(f'Total graphs: {len(graphs)}')
        print(f'Passed threshold: {passed}')
        print(f'Failed threshold: {failed}')
        print(f'Invalid graphs: {sum(1 for r in results if not r["valid"])}')
        print()

    return {
        'threshold': threshold,
        'total': len(graphs),
        'passed': passed,
        'failed': failed,
        'results': results,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python planar_graph_checker.py <test_file.json> [threshold]')
        print()
        print('Examples:')
        print('  python planar_graph_checker.py test_planar_n40_1v9.json')
        print('  python planar_graph_checker.py test_planar_n40_1v9.json 31.28')
        sys.exit(1)

    test_file = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        summary = run_test_suite(test_file, threshold=threshold, verbose=True)

        # Exit with success if all tests passed
        if summary['passed'] == summary['total']:
            sys.exit(0)
        else:
            sys.exit(1)

    except FileNotFoundError:
        print(f'Error: Test file not found: {test_file}')
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in test file: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
