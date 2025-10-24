"""
I/O utilities for saving and loading graphs.
"""

import pickle
import networkx as nx
import numpy as np
from ..core.spanning_trees import count_spanning_trees
from ..utils.validation import check_ramanujan_property
from ..analysis.automorphisms import HAS_PYNAUTY, get_automorphism_group_size


def save_graph(G, filename, T_value=None, compute_all_properties=True):
    """
    Save graph to file with comprehensive metadata.
    Saves as pickle format which preserves all properties.
    
    Args:
        G: NetworkX graph
        filename: Path to save file
        T_value: Pre-computed ln(spanning trees) value (optional)
        compute_all_properties: Whether to compute and save all properties
    """
    # Add basic metadata
    G.graph['n'] = G.number_of_nodes()
    G.graph['k'] = G.number_of_nodes() // 2
    if T_value is not None:
        G.graph['ln_spanning_trees'] = T_value
        G.graph['spanning_trees'] = float(np.exp(T_value))
    
    # Compute and save additional properties if requested
    if compute_all_properties:
        try:
            # Girth (may fail for very large graphs)
            try:
                G.graph['girth'] = nx.girth(G)
            except:
                G.graph['girth'] = None
            
            # Diameter and radius
            if nx.is_connected(G):
                G.graph['diameter'] = nx.diameter(G)
                G.graph['radius'] = nx.radius(G)
            
            # Clustering
            G.graph['avg_clustering'] = nx.average_clustering(G)
            
            # Spectral properties
            eigs = sorted(nx.adjacency_spectrum(G).real, reverse=True)
            G.graph['lambda_1'] = float(eigs[0])
            G.graph['lambda_2'] = float(eigs[1])
            G.graph['lambda_n'] = float(eigs[-1])
            G.graph['spectral_gap'] = float(eigs[0] - eigs[1])
            
            # Ramanujan check
            is_ramanujan, max_nontrivial, margin = check_ramanujan_property(G)
            G.graph['is_ramanujan'] = bool(is_ramanujan)
            G.graph['ramanujan_margin'] = float(margin)
            
            # Algebraic connectivity
            if nx.is_connected(G):
                lap_spec = nx.laplacian_spectrum(G)
                G.graph['algebraic_connectivity'] = float(sorted(lap_spec)[1])
            
            # Automorphism group
            if HAS_PYNAUTY:
                aut_result = get_automorphism_group_size(G)
                if aut_result:
                    G.graph['aut_group_size'] = aut_result[0]
                    G.graph['num_orbits'] = aut_result[1]
                    G.graph['is_vertex_transitive'] = aut_result[2]
        
        except Exception as e:
            print(f"Warning: Could not compute some properties: {e}")
    
    # Save as pickle
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Graph saved to {filename}")
    if compute_all_properties:
        print(f"  Saved with comprehensive metadata")


def load_graph(filename):
    """
    Load graph from file and display metadata.
    
    Args:
        filename: Path to graph file
        
    Returns:
        NetworkX graph with metadata
    """
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Graph loaded from {filename}")
    print(f"Metadata:")
    for key, value in sorted(G.graph.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    return G