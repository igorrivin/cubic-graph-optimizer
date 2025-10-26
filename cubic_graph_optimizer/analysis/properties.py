"""
Graph property analysis utilities.
"""

import networkx as nx
import numpy as np
from ..core.spanning_trees import count_spanning_trees
from ..utils.validation import check_ramanujan_property
from .automorphisms import HAS_PYNAUTY, get_automorphism_group_size, check_vertex_transitivity_simple


def analyze_graph_properties(G, label="Graph", compute_girth=True, compute_cycles=True, compute_automorphisms=True):
    """
    Compute and display various structural properties of the graph.
    
    Args:
        G: NetworkX graph
        label: Label for the graph in output
        compute_girth: Whether to compute girth (can be slow for large graphs)
        compute_cycles: Whether to compute cycle statistics (slow for large graphs)
        compute_automorphisms: Whether to compute automorphism group
    
    Note:
        Set compute_girth=False and compute_cycles=False for large graphs (k>=30).
    """
    print(f"\n{label} Properties:")
    print("-" * 40)
    
    # Basic properties
    print(f"Vertices: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected: {nx.is_connected(G)}")
    
    # Girth (length of shortest cycle) - can be slow/memory-intensive for large graphs
    if compute_girth:
        try:
            girth = nx.girth(G)
            print(f"Girth: {girth}")
        except Exception as e:
            print(f"Girth: Could not compute (error: {str(e)[:50]}...)")
    else:
        print(f"Girth: Skipped (set compute_girth=True to enable)")
    
    # Diameter and radius
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        radius = nx.radius(G)
        print(f"Diameter: {diameter}")
        print(f"Radius: {radius}")
    
    # Average clustering
    clustering = nx.average_clustering(G)
    print(f"Average clustering: {clustering:.4f}")
    
    # Spectral properties
    adj_eigenvalues = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    lambda_1 = adj_eigenvalues[0]  # Should be 3 for cubic graphs
    lambda_2 = adj_eigenvalues[1]  # Second largest
    lambda_n = adj_eigenvalues[-1]  # Most negative
    
    spectral_gap = lambda_1 - lambda_2
    print(f"Spectral gap (λ₁ - λ₂): {spectral_gap:.4f}")
    print(f"λ₂ (second eigenvalue): {lambda_2:.4f}")
    print(f"λₙ (smallest eigenvalue): {lambda_n:.4f}")
    
    # Ramanujan property check
    is_ramanujan, max_nontrivial_eigenvalue, margin = check_ramanujan_property(G)
    
    print(f"Max |λ| (non-trivial): {max_nontrivial_eigenvalue:.4f}")
    print(f"Ramanujan bound (2√2): {2 * np.sqrt(2):.4f}")
    print(f"Is Ramanujan? {is_ramanujan} {'✓' if is_ramanujan else '✗'}")
    if is_ramanujan:
        print(f"Margin below bound: {margin:.4f}")
    
    # Algebraic connectivity (Fiedler value) - for comparison
    if nx.is_connected(G):
        laplacian_spectrum = nx.laplacian_spectrum(G)
        algebraic_conn = sorted(laplacian_spectrum)[1]
        print(f"Algebraic connectivity: {algebraic_conn:.4f}")
    
    # Automorphism group and symmetry properties
    if compute_automorphisms:
        if HAS_PYNAUTY:
            aut_result = get_automorphism_group_size(G)
            if aut_result:
                group_size, num_orbits, is_vt = aut_result
                print(f"Automorphism group size: {group_size}")
                print(f"Number of vertex orbits: {num_orbits}")
                print(f"Vertex-transitive: {is_vt} {'✓' if is_vt else '✗'}")
        else:
            # Fallback to simple transitivity check
            is_vt, num_profiles = check_vertex_transitivity_simple(G)
            print(f"Appears vertex-transitive: {is_vt} (simple check)")
            if not is_vt:
                print(f"  Found {num_profiles} distinct vertex profiles")
    
    # Cycle statistics - also expensive for large graphs
    if compute_cycles and G.number_of_nodes() <= 30:
        cycles = list(nx.simple_cycles(G.to_directed()))
        if cycles:
            cycle_lengths = [len(c) for c in cycles]
            print(f"Total cycles: {len(cycles)}")
            print(f"Cycle length distribution: min={min(cycle_lengths)}, "
                  f"mean={np.mean(cycle_lengths):.2f}, max={max(cycle_lengths)}")
    else:
        print(f"Cycle statistics: Skipped for large graph")
    
    # Spanning trees - the key metric!
    T_value = count_spanning_trees(G)
    print(f"ln(spanning trees): {T_value:.6f}")
    print(f"Spanning trees: ≈ {np.exp(T_value):.2e}")
    print("-" * 40)


def compare_graph_structures(G_initial, G_optimized, compute_expensive=True, compute_automorphisms=True):
    """
    Compare structural properties between initial and optimized graphs.
    
    Args:
        G_initial: Initial NetworkX graph
        G_optimized: Optimized NetworkX graph
        compute_expensive: Whether to compute expensive properties (girth, cycles)
        compute_automorphisms: Whether to compute automorphism groups
    
    Note:
        Set compute_expensive=False for large graphs (k>=30).
    """
    print("\n" + "=" * 60)
    print("STRUCTURAL COMPARISON")
    print("=" * 60)
    
    analyze_graph_properties(G_initial, "Initial Graph", 
                            compute_girth=compute_expensive,
                            compute_cycles=compute_expensive,
                            compute_automorphisms=compute_automorphisms)
    analyze_graph_properties(G_optimized, "Optimized Graph",
                            compute_girth=compute_expensive, 
                            compute_cycles=compute_expensive,
                            compute_automorphisms=compute_automorphisms)
    
    # Compute changes
    print("\nKey Changes:")
    print("-" * 40)
    
    if compute_expensive:
        try:
            girth_initial = nx.girth(G_initial)
            girth_optimized = nx.girth(G_optimized)
            print(f"Girth: {girth_initial} → {girth_optimized} "
                  f"({'↑' if girth_optimized > girth_initial else '↓' if girth_optimized < girth_initial else '='}"
                  f" {abs(girth_optimized - girth_initial)})")
        except:
            print("Girth: Could not compare")
    
    if nx.is_connected(G_initial) and nx.is_connected(G_optimized):
        diam_initial = nx.diameter(G_initial)
        diam_optimized = nx.diameter(G_optimized)
        print(f"Diameter: {diam_initial} → {diam_optimized} "
              f"({'↑' if diam_optimized > diam_initial else '↓' if diam_optimized < diam_initial else '='}"
              f" {abs(diam_optimized - diam_initial)})")
    
    clust_initial = nx.average_clustering(G_initial)
    clust_optimized = nx.average_clustering(G_optimized)
    print(f"Clustering: {clust_initial:.4f} → {clust_optimized:.4f} "
          f"({'↑' if clust_optimized > clust_initial else '↓'})")
    
    # Spectral gap changes
    eigs_initial = sorted(nx.adjacency_spectrum(G_initial).real, reverse=True)
    eigs_optimized = sorted(nx.adjacency_spectrum(G_optimized).real, reverse=True)
    gap_initial = eigs_initial[0] - eigs_initial[1]
    gap_optimized = eigs_optimized[0] - eigs_optimized[1]
    print(f"Spectral gap: {gap_initial:.4f} → {gap_optimized:.4f} "
          f"({'↑' if gap_optimized > gap_initial else '↓'})")
    
    # Check Ramanujan property change
    is_ram_initial, _, _ = check_ramanujan_property(G_initial)
    is_ram_optimized, _, _ = check_ramanujan_property(G_optimized)
    
    if is_ram_initial and is_ram_optimized:
        print(f"Ramanujan: Yes → Yes (both satisfy bound)")
    elif not is_ram_initial and is_ram_optimized:
        print(f"Ramanujan: No → Yes ✓ (optimization achieved Ramanujan!)")
    elif is_ram_initial and not is_ram_optimized:
        print(f"Ramanujan: Yes → No (lost Ramanujan property)")
    else:
        print(f"Ramanujan: No → No (neither satisfies bound)")
    
    # Automorphism group changes
    if compute_automorphisms and HAS_PYNAUTY:
        aut_initial = get_automorphism_group_size(G_initial)
        aut_optimized = get_automorphism_group_size(G_optimized)
        if aut_initial and aut_optimized:
            print(f"Aut. group size: {aut_initial[0]} → {aut_optimized[0]} "
                  f"({'↑' if aut_optimized[0] > aut_initial[0] else '↓' if aut_optimized[0] < aut_initial[0] else '='})")
            if aut_optimized[2]:
                print(f"  → Optimized graph is vertex-transitive! ✓")
    
    T_initial = count_spanning_trees(G_initial)
    T_optimized = count_spanning_trees(G_optimized)
    improvement = T_optimized - T_initial
    print(f"\nln(spanning trees): {T_initial:.6f} → {T_optimized:.6f}")
    print(f"Improvement: {improvement:.6f} ({100*improvement/T_initial:.2f}%)")
    print("-" * 40)