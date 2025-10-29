"""
Operations for planar cubic graphs.

Planar cubic graphs have a beautiful duality with triangulations:
- The dual of a planar cubic graph is a triangulation (3-connected planar graph with triangular faces)
- A Whitehead flip in the cubic graph corresponds to a diagonal flip in the dual triangulation
- Only Whitehead flips that preserve planarity are valid

This module implements planarity-preserving Whitehead flips and generators for planar cubic graphs.
"""

import networkx as nx
import numpy as np
import random
from itertools import combinations
from typing import Tuple, List, Optional
from scipy.spatial import ConvexHull


def is_planar_cubic(G: nx.Graph) -> bool:
    """Check if a graph is both planar and cubic (3-regular)."""
    if not all(d == 3 for _, d in G.degree()):
        return False
    return nx.is_planar(G)


def check_flip_preserves_planarity(G: nx.Graph, e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
    """
    Check if a Whitehead flip preserves planarity.

    A Whitehead flip swaps edges e1=(a,b) and e2=(c,d) to (a,c) and (b,d),
    where {a,b,c,d} form a 4-cycle.

    Args:
        G: The graph
        e1: First edge tuple (a, b)
        e2: Second edge tuple (c, d)

    Returns:
        True if the flip preserves planarity
    """
    (a, b), (c, d) = e1, e2

    # Check if the new edges would create duplicates
    # The new edges are (a,c) and (b,d)
    if G.has_edge(a, c) or G.has_edge(b, d):
        return False

    # Create a copy and perform the flip
    G_test = G.copy()

    # Remove old edges and add new ones
    G_test.remove_edges_from([e1, e2])
    G_test.add_edges_from([(a, c), (b, d)])

    # Check if still planar and connected
    if not nx.is_connected(G_test):
        return False

    return nx.is_planar(G_test)


def find_planar_whitehead_flips(G: nx.Graph, max_flips: Optional[int] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Find all planarity-preserving Whitehead flips in a planar cubic graph.

    A Whitehead flip candidate requires:
    1. Two edges that share no vertices
    2. The four vertices form a 4-cycle (exactly 4 edges among them)
    3. The flip preserves planarity

    Args:
        G: Planar cubic graph
        max_flips: Maximum number of flips to return (None = all)

    Returns:
        List of valid flip pairs: [(e1, e2), ...]
    """
    planar_flips = []

    # Find all Whitehead flip candidates
    for e1, e2 in combinations(G.edges(), 2):
        a, b = e1
        c, d = e2

        # Check if they share no vertices
        vertices = {a, b, c, d}
        if len(vertices) != 4:
            continue

        # Check if they form a 4-cycle (subgraph has exactly 4 edges)
        subgraph = G.subgraph(vertices)
        if subgraph.number_of_edges() != 4:
            continue

        # Check if flip preserves planarity
        if check_flip_preserves_planarity(G, e1, e2):
            planar_flips.append((e1, e2))

            if max_flips and len(planar_flips) >= max_flips:
                break

    return planar_flips


def perform_planar_flip(G: nx.Graph, e1: Tuple[int, int], e2: Tuple[int, int]) -> nx.Graph:
    """
    Perform a planarity-preserving Whitehead flip.

    WARNING: This does NOT check if the flip preserves planarity!
    Use check_flip_preserves_planarity() first.

    Args:
        G: The graph
        e1: First edge (a, b)
        e2: Second edge (c, d)

    Returns:
        Modified graph (modifies G in place and returns it)
    """
    (a, b), (c, d) = e1, e2

    # Remove old edges
    G.remove_edges_from([e1, e2])

    # Add new edges
    G.add_edges_from([(a, c), (b, d)])

    return G


def random_planar_cubic_graph(n: int, num_randomization_flips: int = 100, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a random planar cubic graph by applying random planarity-preserving flips
    to a starting planar cubic graph.

    Strategy:
    1. Start with a known planar cubic graph (cube, prism, or other)
    2. Apply many random planarity-preserving Whitehead flips
    3. This explores the space of planar cubic graphs

    Args:
        n: Number of vertices (must be even, >= 4)
        num_randomization_flips: Number of random flips to apply
        seed: Random seed

    Returns:
        A random planar cubic graph with n vertices

    Raises:
        ValueError: If n is odd or < 4
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even for cubic graphs (given n={n})")
    if n < 4:
        raise ValueError(f"n must be >= 4 (given n={n})")

    if seed is not None:
        random.seed(seed)

    # Generate starting graph based on size
    if n == 4:
        # K4 (complete graph on 4 vertices)
        G = nx.complete_graph(4)
    elif n == 6:
        # Prism graph (triangular prism)
        G = nx.circular_ladder_graph(3)
    elif n == 8:
        # Cube graph
        G = nx.cubical_graph()
    else:
        # For larger graphs, start with a prism and use a different strategy
        # Use the circular ladder construction and extend it
        m = n // 2
        G = nx.circular_ladder_graph(m)

        # If not cubic yet, need to construct differently
        if not all(d == 3 for _, d in G.degree()):
            # Fall back to constructing from smaller graph with reconnections
            # For now, start with largest available and will need a different approach
            # This is a limitation - we'd need plantri or a proper generator
            raise NotImplementedError(
                f"Random planar cubic graph generation for n={n} not yet implemented. "
                f"Try n in {{4, 6, 8}} or use plantri."
            )

    # Verify starting graph
    if not is_planar_cubic(G):
        raise ValueError(f"Starting graph is not planar cubic!")

    # Apply random planarity-preserving flips
    for _ in range(num_randomization_flips):
        # Find valid flips
        valid_flips = find_planar_whitehead_flips(G, max_flips=50)

        if not valid_flips:
            # No valid flips available (might be a highly symmetric graph)
            break

        # Choose random flip
        e1, e2 = random.choice(valid_flips)

        # Perform flip
        perform_planar_flip(G, e1, e2)

    return G


def random_planar_cubic_from_sphere(n: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a random planar cubic graph via sphere construction.

    Algorithm:
    1. Generate n random points uniformly on the unit sphere
    2. Compute their convex hull (gives a triangulated polyhedron)
    3. Return the dual graph (faces → vertices, adjacent faces → edges)

    By Steinitz's theorem, the dual of a convex polyhedron's face graph is always
    a 3-connected planar graph. Since the faces are triangles (generic position),
    the dual is cubic (3-regular).

    This is a much more general random generator than the flip-based approach!

    Args:
        n: Number of points on sphere (= number of faces = number of vertices in dual)
        seed: Random seed for reproducibility

    Returns:
        A random planar cubic graph with approximately n vertices

    Note:
        The actual number of vertices may be slightly less than n if some points
        are redundant (inside the convex hull), but this is rare for random points.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate n random points on unit sphere
    # Method: generate 3D Gaussian, normalize to unit length
    points = np.random.randn(n, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Compute convex hull
    hull = ConvexHull(points)

    # Build dual graph:
    # - Each face (triangle) becomes a vertex
    # - Two vertices are connected if their faces share an edge

    G = nx.Graph()

    # Add vertices (one per face)
    num_faces = len(hull.simplices)
    G.add_nodes_from(range(num_faces))

    # Build edge adjacency: two faces are adjacent if they share an edge (2 vertices)
    # Create a map from edges to faces containing them
    edge_to_faces = {}

    for face_idx, simplex in enumerate(hull.simplices):
        # Each simplex is a triangle (3 vertices)
        # Its edges are all pairs of vertices
        vertices = sorted(simplex)
        edges = [
            tuple(sorted([vertices[0], vertices[1]])),
            tuple(sorted([vertices[1], vertices[2]])),
            tuple(sorted([vertices[0], vertices[2]])),
        ]

        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Add edges to dual graph: faces sharing an edge are neighbors
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            # Two faces share this edge → connect them in dual
            G.add_edge(faces[0], faces[1])

    return G


def get_dual_triangulation(G: nx.Graph) -> nx.Graph:
    """
    Compute the dual triangulation of a planar cubic graph.

    For a planar cubic graph with a planar embedding:
    - Each face becomes a vertex in the dual
    - Two vertices in the dual are connected if their corresponding faces share an edge
    - The result is a triangulation (all faces are triangles)

    Args:
        G: Planar cubic graph with embedding

    Returns:
        The dual triangulation

    Note:
        This requires a planar embedding. NetworkX can compute this via check_planarity().
    """
    # Get planar embedding
    is_planar, embedding = nx.check_planarity(G)

    if not is_planar:
        raise ValueError("Graph is not planar!")

    # NetworkX doesn't have a built-in dual graph for arbitrary planar graphs
    # We'd need to implement face finding and dual construction
    # This is non-trivial, so leaving as placeholder for now
    raise NotImplementedError("Dual triangulation computation not yet implemented")


# Known small planar cubic graphs
SMALL_PLANAR_CUBIC_GRAPHS = {
    4: "K4 (complete graph)",
    6: "Prism",
    8: "Cube",
    # Could add more: 10, 12, etc.
}


def list_available_sizes():
    """List the sizes of planar cubic graphs we can generate."""
    print("Available planar cubic graph sizes:")
    for n, name in SMALL_PLANAR_CUBIC_GRAPHS.items():
        print(f"  N={n}: {name}")
    print("\nFor other sizes, you'll need plantri or a custom generator.")
