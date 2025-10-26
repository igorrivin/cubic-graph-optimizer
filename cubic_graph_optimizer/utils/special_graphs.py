"""
Utilities for checking against known special graphs.
"""

import networkx as nx


def check_known_cages(G, k):
    """
    Check if graph is isomorphic to known cubic cages.
    
    Args:
        G: NetworkX graph to check
        k: Parameter k (where n = 2k)
        
    Returns:
        tuple: (is_known_cage, cage_name) or (False, None)
    """
    if k == 5:
        petersen = nx.petersen_graph()
        if nx.is_isomorphic(G, petersen):
            return True, "Petersen graph"
    
    elif k == 7:
        heawood = nx.LCF_graph(14, [5, -5], 7)
        if nx.is_isomorphic(G, heawood):
            return True, "Heawood graph"
    
    elif k == 12:
        mcgee = nx.LCF_graph(24, [12, 7, -7], 8)
        if nx.is_isomorphic(G, mcgee):
            return True, "McGee graph"
    
    elif k == 15:
        tutte = nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5)
        if nx.is_isomorphic(G, tutte):
            return True, "Tutte-Coxeter graph"
    
    return False, None