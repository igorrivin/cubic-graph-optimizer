"""
Operations on planar triangulations.

Working directly with triangulations is cleaner than working with their duals:
1. Diagonal flips are well-defined and easy to implement
2. Spanning trees count is the same as in the dual (for planar graphs)
3. Can dualize when needed for other objectives
"""

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional, Set
from itertools import combinations


def random_triangulation_from_sphere(n: int, seed: Optional[int] = None) -> Tuple[nx.Graph, np.ndarray]:
    """
    Generate a random planar triangulation via sphere construction.

    Algorithm:
    1. Generate n random points uniformly on the unit sphere
    2. Compute their convex hull (gives a triangulated polyhedron)
    3. Return the vertex graph

    Args:
        n: Number of points on sphere
        seed: Random seed

    Returns:
        (triangulation_graph, point_coordinates)
        The graph has the convex hull as edges, and we return the 3D coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate n random points on unit sphere
    points = np.random.randn(n, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Compute convex hull
    hull = ConvexHull(points)

    # Build vertex graph (edges of the polyhedron)
    G = nx.Graph()
    G.add_nodes_from(range(len(hull.vertices)))

    # Extract edges from simplices (triangular faces)
    edges = set()
    for simplex in hull.simplices:
        # Each simplex is a triangle
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)

    G.add_edges_from(edges)

    # Store the 3D coordinates as node attributes
    for i, vertex_idx in enumerate(hull.vertices):
        G.nodes[i]['pos'] = points[vertex_idx]

    return G, points[hull.vertices]


def find_flippable_edges(G: nx.Graph) -> List[Tuple[int, int]]:
    """
    Find all edges in a triangulation that can be flipped.

    An edge can be flipped if:
    1. It's an interior edge (not on the boundary)
    2. It has exactly 2 adjacent triangular faces
    3. Flipping it maintains planarity (always true for convex position)

    For a convex hull triangulation, all edges are potentially flippable
    except we need to check the degree-3 property.

    Args:
        G: Triangulation graph

    Returns:
        List of flippable edges
    """
    flippable = []

    for edge in G.edges():
        u, v = edge

        # Find the two triangles adjacent to this edge
        # These are the common neighbors of u and v
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common_neighbors = neighbors_u & neighbors_v

        if len(common_neighbors) == 2:
            # This is an interior edge with exactly 2 adjacent triangles
            # The quadrilateral is formed by u, v, and the two common neighbors
            a, b = common_neighbors

            # Check if the diagonal flip is valid
            # The new edge would be (a, b)
            if not G.has_edge(a, b):
                flippable.append(edge)

    return flippable


def diagonal_flip(G: nx.Graph, edge: Tuple[int, int]) -> nx.Graph:
    """
    Perform a diagonal flip on an edge in a triangulation.

    Given edge (u, v) with adjacent triangles (u, v, a) and (u, v, b),
    removes (u, v) and adds (a, b).

    Args:
        G: Triangulation graph
        edge: Edge to flip (u, v)

    Returns:
        Modified graph (modifies in place and returns)

    Raises:
        ValueError: If edge is not flippable
    """
    u, v = edge

    # Find the quadrilateral
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common_neighbors = neighbors_u & neighbors_v

    if len(common_neighbors) != 2:
        raise ValueError(f"Edge {edge} is not flippable (has {len(common_neighbors)} common neighbors)")

    a, b = common_neighbors

    if G.has_edge(a, b):
        raise ValueError(f"Edge {edge} is not flippable (new edge {(a,b)} already exists)")

    # Perform the flip
    G.remove_edge(u, v)
    G.add_edge(a, b)

    return G


def triangulation_to_dual_cubic(G: nx.Graph) -> nx.Graph:
    """
    Convert a planar triangulation to its dual cubic graph.

    Each triangular face becomes a vertex in the dual.
    Two vertices in the dual are connected if their faces share an edge.

    Args:
        G: Triangulation (planar graph with triangular faces)

    Returns:
        Dual cubic graph

    Note:
        For a triangulation, each internal face is a triangle, so each has degree 3.
        The dual will be cubic (3-regular).
    """
    # Convert to regular int labels (avoid numpy int32 issues)
    G_relabeled = nx.convert_node_labels_to_integers(G, first_label=0)

    # Get planar embedding
    is_planar, embedding = nx.check_planarity(G_relabeled)

    if not is_planar:
        raise ValueError("Graph is not planar!")

    # Find all faces by traversing the embedding
    faces = []
    visited_half_edges = set()

    # Traverse all faces
    for node in embedding.nodes():
        for neighbor in embedding.neighbors_cw_order(node):
            half_edge = (node, neighbor)

            # Skip if already visited this directed edge
            if half_edge in visited_half_edges:
                continue

            # Traverse this face (returns list of vertices)
            face_vertices = embedding.traverse_face(node, neighbor)

            # Mark all half-edges in this face as visited
            for i in range(len(face_vertices)):
                v1 = face_vertices[i]
                v2 = face_vertices[(i + 1) % len(face_vertices)]
                visited_half_edges.add((v1, v2))

            faces.append(face_vertices)

    # Build dual graph
    # Each face becomes a vertex
    dual = nx.Graph()
    dual.add_nodes_from(range(len(faces)))

    # Build edge→faces mapping
    edge_to_faces = {}
    for face_idx, face_vertices in enumerate(faces):
        # Get edges from face vertices
        for i in range(len(face_vertices)):
            u = face_vertices[i]
            v = face_vertices[(i + 1) % len(face_vertices)]
            # Undirected edge
            edge = tuple(sorted([u, v]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Add edges in dual: faces sharing an edge are adjacent
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            dual.add_edge(face_list[0], face_list[1])

    return dual


def optimize_triangulation(
    G: nx.Graph,
    objective_func,
    maximize: bool = True,
    max_iterations: int = 100,
    verbose: bool = False,
) -> Tuple[nx.Graph, float, dict]:
    """
    Optimize a triangulation using diagonal flips.

    Args:
        G: Initial triangulation
        objective_func: Function to optimize (takes graph, returns float)
        maximize: If True, maximize; if False, minimize
        max_iterations: Maximum iterations
        verbose: Print progress

    Returns:
        (optimized_graph, final_value, stats)
    """
    import random

    G = G.copy()
    current_value = objective_func(G)

    stats = {
        'initial_value': current_value,
        'iterations': 0,
        'flips_performed': 0,
    }

    if verbose:
        print(f"Initial value: {current_value:.6f}")

    for iteration in range(max_iterations):
        improved = False

        # Find all flippable edges
        flippable = find_flippable_edges(G)

        if not flippable:
            if verbose:
                print(f"Iteration {iteration}: No flippable edges")
            break

        # Shuffle for randomization
        random.shuffle(flippable)

        # Try each flip
        for edge in flippable:
            # Test flip
            G_test = G.copy()
            diagonal_flip(G_test, edge)

            new_value = objective_func(G_test)

            # Check if improved
            is_better = (new_value > current_value) if maximize else (new_value < current_value)

            if is_better:
                G = G_test
                current_value = new_value
                stats['flips_performed'] += 1
                improved = True

                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: {current_value:.6f}")

                break

        stats['iterations'] += 1

        if not improved:
            if verbose:
                print(f"Local optimum at iteration {iteration}")
            break

    stats['final_value'] = current_value

    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Final value: {current_value:.6f}")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Flips: {stats['flips_performed']}")

    return G, current_value, stats
