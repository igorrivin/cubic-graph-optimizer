#!/usr/bin/env python3
"""
Benchmark eigenvalue computation to find the bottleneck.
"""

import time
import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla

# Load a real graph from the sweep
import json

with open('planar_sweep_results/planar_n150_result.json') as f:
    data = json.load(f)

# Reconstruct graph
G = nx.Graph()
adj_list = data['adjacency_list']
G.add_nodes_from(range(len(adj_list)))
for node, neighbors in enumerate(adj_list):
    for neighbor in neighbors:
        if node < neighbor:
            G.add_edge(node, neighbor)

print(f"Graph: N={G.number_of_nodes()}, E={G.number_of_edges()}")
print()

# Method 1: NetworkX adjacency_spectrum (current method)
print("=" * 60)
print("Method 1: nx.adjacency_spectrum(G)")
print("=" * 60)
start = time.time()
eigenvalues = nx.adjacency_spectrum(G)
eigenvalues_sorted = sorted(eigenvalues.real, reverse=True)
lambda2 = eigenvalues_sorted[1]
elapsed = time.time() - start
print(f"Result: λ₂ = {lambda2:.6f}")
print(f"Time: {elapsed:.4f} seconds")
print()

# Method 2: Dense numpy eigenvalues
print("=" * 60)
print("Method 2: Dense numpy (np.linalg.eigvalsh)")
print("=" * 60)
start = time.time()
A = nx.adjacency_matrix(G).toarray()
eigenvalues = np.linalg.eigvalsh(A)  # Symmetric, faster than eig
eigenvalues_sorted = sorted(eigenvalues, reverse=True)
lambda2 = eigenvalues_sorted[1]
elapsed = time.time() - start
print(f"Result: λ₂ = {lambda2:.6f}")
print(f"Time: {elapsed:.4f} seconds")
print()

# Method 3: Sparse eigenvalues (only top 2)
print("=" * 60)
print("Method 3: Sparse (scipy.sparse.linalg.eigsh, k=2)")
print("=" * 60)
start = time.time()
A_sparse = nx.adjacency_matrix(G)
eigenvalues = spla.eigsh(A_sparse, k=2, which='LA', return_eigenvectors=False)
lambda2 = sorted(eigenvalues, reverse=True)[1]
elapsed = time.time() - start
print(f"Result: λ₂ = {lambda2:.6f}")
print(f"Time: {elapsed:.4f} seconds")
print()

# Method 4: Just converting to adjacency matrix
print("=" * 60)
print("Method 4: Just nx.adjacency_matrix(G) (no eigenvalues)")
print("=" * 60)
start = time.time()
A = nx.adjacency_matrix(G)
elapsed = time.time() - start
print(f"Time: {elapsed:.4f} seconds")
print()

# Method 5: Converting to dense
print("=" * 60)
print("Method 5: nx.adjacency_matrix(G).toarray()")
print("=" * 60)
start = time.time()
A = nx.adjacency_matrix(G).toarray()
elapsed = time.time() - start
print(f"Time: {elapsed:.4f} seconds")
print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("If Method 1 is slow (>1s), the issue is nx.adjacency_spectrum")
print("If Method 2 is fast (<0.1s), we should use dense numpy")
print("If Method 3 is fastest (<0.05s), we should use sparse k=2")
