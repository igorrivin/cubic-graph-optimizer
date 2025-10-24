"""
Validation and checking utilities for cubic graphs.
"""

import networkx as nx
import numpy as np


def is_cubic_graph(G):
    """
    Check if graph is cubic (3-regular).
    
    Args:
        G: NetworkX graph
        
    Returns:
        bool: True if graph is cubic
    """
    degrees = dict(G.degree())
    return all(d == 3 for d in degrees.values())


def has_more_spanning_trees_than(G, ln_threshold):
    """
    Check if a graph has more than exp(ln_threshold) spanning trees.
    
    Args:
        G: NetworkX graph
        ln_threshold: natural log of the threshold (checking if count > exp(ln_threshold))
    
    Returns:
        bool: True if ln(spanning trees) > ln_threshold, False otherwise
    
    Example:
        >>> G = nx.petersen_graph()
        >>> has_more_spanning_trees_than(G, 7.5)  # Check if more than e^7.5 ≈ 1808 trees
        True
        >>> has_more_spanning_trees_than(G, 7.7)  # Check if more than e^7.7 ≈ 2208 trees
        False
    """
    if not nx.is_connected(G):
        return False
    
    # Get Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()
    
    # Remove last row and column (any row/column works)
    L_reduced = L[:-1, :-1]
    
    # Compute determinant using sign and logdet to avoid overflow
    sign, logdet = np.linalg.slogdet(L_reduced)
    
    if sign <= 0:
        return False
    
    return logdet > ln_threshold


def check_ramanujan_property(G):
    """
    Check if a cubic graph satisfies the Ramanujan property.
    
    A d-regular graph is Ramanujan if all non-trivial eigenvalues λ satisfy |λ| ≤ 2√(d-1).
    For cubic graphs (d=3): |λ| ≤ 2√2 ≈ 2.828
    
    Args:
        G: NetworkX graph
        
    Returns:
        tuple: (is_ramanujan, max_nontrivial_eigenvalue, margin)
            - is_ramanujan: bool, whether graph is Ramanujan
            - max_nontrivial_eigenvalue: float, max |λ| among non-trivial eigenvalues
            - margin: float, distance from Ramanujan bound (positive if satisfied)
    """
    d = 3  # cubic
    ramanujan_bound = 2 * np.sqrt(d - 1)
    
    eigenvalues = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    lambda_2 = eigenvalues[1]  # Second largest
    lambda_n = eigenvalues[-1]  # Most negative
    
    max_nontrivial = max(abs(lambda_2), abs(lambda_n))
    is_ramanujan = max_nontrivial <= ramanujan_bound
    margin = ramanujan_bound - max_nontrivial
    
    return is_ramanujan, max_nontrivial, margin


def validate_whitehead_flip(G, edge1, edge2, flip_type):
    """
    Validate that a Whitehead flip doesn't create multi-edges or self-loops.
    
    Args:
        G: NetworkX graph
        edge1: tuple (a, b) representing first edge
        edge2: tuple (c, d) representing second edge
        flip_type: 0 for (a,c)(b,d), 1 for (a,d)(b,c)
        
    Returns:
        bool: True if flip is valid
    """
    a, b = edge1
    c, d = edge2
    
    # Check edges exist
    if not G.has_edge(a, b) or not G.has_edge(c, d):
        return False
    
    # Check edges are disjoint
    if len({a, b, c, d}) != 4:
        return False
    
    # Check flip won't create multi-edges
    if flip_type == 0:
        return not G.has_edge(a, c) and not G.has_edge(b, d)
    else:  # flip_type == 1
        return not G.has_edge(a, d) and not G.has_edge(b, c)