"""
Core functions for computing spanning trees.
"""

import networkx as nx
import numpy as np


def count_spanning_trees(G):
    """
    Compute ln(number of spanning trees) using Kirchhoff's theorem.
    Returns the natural log to avoid overflow for large graphs.

    Args:
        G: NetworkX graph

    Returns:
        float: Natural log of the number of spanning trees
               Returns -inf if graph is not connected
    """
    if not nx.is_connected(G):
        return float('-inf')

    # Get Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Remove last row and column (any row/column works)
    L_reduced = L[:-1, :-1]

    # Compute determinant
    # Use sign and logdet to avoid overflow
    sign, logdet = np.linalg.slogdet(L_reduced)

    if sign <= 0:
        return float('-inf')

    return logdet


def get_second_eigenvalue(G):
    """
    Compute the second-largest eigenvalue of the adjacency matrix.
    For cubic graphs, λ₁ = 3 always, so this returns λ₂.

    Minimizing λ₂ maximizes the spectral gap and graph expansion.

    Args:
        G: NetworkX graph

    Returns:
        float: The second-largest eigenvalue (λ₂)
               Returns +inf if graph is not connected
    """
    if not nx.is_connected(G):
        return float('inf')

    # Compute eigenvalues of adjacency matrix
    eigenvalues = nx.adjacency_spectrum(G).real

    # Sort in descending order
    eigenvalues_sorted = sorted(eigenvalues, reverse=True)

    # Return second-largest (index 1)
    return eigenvalues_sorted[1]