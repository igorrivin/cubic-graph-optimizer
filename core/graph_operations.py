"""
Graph operations including Whitehead flips.
"""

import networkx as nx


def get_valid_whitehead_flips(G):
    """
    Generate all valid Whitehead flips that don't create multi-edges or self-loops.
    
    Args:
        G: NetworkX graph
        
    Returns:
        list: List of tuples ((a,b), (c,d), flip_type) where flip_type is 0 or 1
    """
    edges = list(G.edges())
    valid_flips = []
    
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i:  # Avoid duplicates and same edge
                continue
            
            # Check if edges are disjoint
            if len({a, b, c, d}) != 4:
                continue
            
            # Flip type 0: (a,b), (c,d) -> (a,c), (b,d)
            if not G.has_edge(a, c) and not G.has_edge(b, d):
                valid_flips.append(((a, b), (c, d), 0))
            
            # Flip type 1: (a,b), (c,d) -> (a,d), (b,c)
            if not G.has_edge(a, d) and not G.has_edge(b, c):
                valid_flips.append(((a, b), (c, d), 1))
    
    return valid_flips


def apply_whitehead_flip(G, edge1, edge2, flip_type):
    """
    Apply a Whitehead flip to graph G (in-place).
    
    Args:
        G: NetworkX graph (modified in-place)
        edge1: tuple (a, b) representing first edge
        edge2: tuple (c, d) representing second edge
        flip_type: 0 for (a,c)(b,d), 1 for (a,d)(b,c)
    """
    a, b = edge1
    c, d = edge2
    
    G.remove_edge(a, b)
    G.remove_edge(c, d)
    
    if flip_type == 0:
        G.add_edge(a, c)
        G.add_edge(b, d)
    else:  # flip_type == 1
        G.add_edge(a, d)
        G.add_edge(b, c)