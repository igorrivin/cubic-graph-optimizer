"""
Automorphism group analysis for graphs.
"""

import networkx as nx

# Try to import pynauty for automorphism group computation
# Install with: pip install pynauty
try:
    import pynauty
    HAS_PYNAUTY = True
except ImportError:
    HAS_PYNAUTY = False
    print("Note: pynauty not available. Install with 'pip install pynauty' for automorphism group computation.")


def get_automorphism_group_size(G):
    """
    Compute the size of the automorphism group using pynauty.
    
    Args:
        G: NetworkX graph
        
    Returns:
        tuple or None: (group_size, num_orbits, is_vertex_transitive) or None if pynauty unavailable
        
    Note:
        pynauty.autgrp returns: (generators, group_size, flag, orbit_partition, num_orbits)
    """
    if not HAS_PYNAUTY:
        return None
    
    try:
        # Convert NetworkX graph to pynauty format
        node_mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
        n = len(node_mapping)
        
        # Build adjacency dict
        adjacency_dict = {}
        for node in G.nodes():
            i = node_mapping[node]
            neighbor_list = [node_mapping[neighbor] for neighbor in G.neighbors(node)]
            adjacency_dict[i] = sorted(neighbor_list)
        
        # Create pynauty graph
        g = pynauty.Graph(
            number_of_vertices=n,
            directed=False,
            adjacency_dict=adjacency_dict
        )
        
        # Compute automorphism group
        # Returns: (generators, group_size, flag, orbit_partition, num_orbits)
        aut_result = pynauty.autgrp(g)
        
        generators = aut_result[0]
        group_size = aut_result[1]
        # aut_result[2] is some flag (always 0?)
        # aut_result[3] is orbit partition (list assigning each vertex to an orbit)
        num_orbits = aut_result[4]  # Number of orbits directly from pynauty
        
        # Vertex-transitive means all vertices are in the same orbit
        is_vertex_transitive = (num_orbits == 1)
        
        # Sanity check: if vertex-transitive, group size should be divisible by n
        if is_vertex_transitive and n > 0 and group_size > 0:
            remainder = group_size % n
            if abs(remainder) > 0.001:  # Allow small floating point error
                print(f"  [WARNING] Vertex-transitive but |Aut|={group_size} not divisible by n={n}!")
                print(f"  [WARNING] Remainder: {remainder}")
        
        return group_size, num_orbits, is_vertex_transitive
    
    except Exception as e:
        print(f"Error computing automorphism group: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_vertex_transitivity_simple(G):
    """
    Simple heuristic to check if graph appears vertex-transitive.
    Checks if all vertices have identical distance profiles.
    
    Args:
        G: NetworkX graph
        
    Returns:
        tuple: (is_likely_vertex_transitive, num_unique_profiles)
    """
    distance_profiles = []
    for v in G.nodes():
        distances = nx.single_source_shortest_path_length(G, v)
        # Profile is sorted list of distances to all other vertices
        profile = tuple(sorted(distances.values()))
        distance_profiles.append(profile)
    
    # If all profiles are identical, likely vertex-transitive
    unique_profiles = len(set(distance_profiles))
    return unique_profiles == 1, unique_profiles


def test_pynauty():
    """Simple test function to debug pynauty issues."""
    if not HAS_PYNAUTY:
        print("pynauty not available")
        return False
    
    print("Testing pynauty with simple graph (K4)...")
    try:
        # Create simple complete graph K4
        g = pynauty.Graph(number_of_vertices=4, directed=False,
                         adjacency_dict={0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]})
        
        result = pynauty.autgrp(g)
        print(f"K4 autgrp result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        for i, item in enumerate(result):
            print(f"  Item {i}: {item}, type: {type(item)}")
        
        print("pynauty test successful!")
        return True
    except Exception as e:
        print(f"pynauty test failed: {e}")
        import traceback
        traceback.print_exc()
        return False