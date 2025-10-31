# CRITICAL Performance Fix: Eigenvalue Computation

## Problem

The current `get_second_eigenvalue()` function in `cubic_graph_optimizer/core/spanning_trees.py` uses:

```python
eigenvalues = nx.adjacency_spectrum(G).real  # Line 58
```

**This is 7000× slower than it should be!**

## Benchmark Results (N=150)

| Method | Time | Speedup |
|--------|------|---------|
| `nx.adjacency_spectrum(G)` | **10.54 seconds** | 1× (baseline) |
| `np.linalg.eigvalsh(A)` | **0.0015 seconds** | **7027×** |
| `scipy.sparse.linalg.eigsh(A, k=2)` | 0.027 seconds | 391× |

## Root Cause

NetworkX's `adjacency_spectrum()` appears to:
1. Convert graph to dense matrix
2. Use a slow eigenvalue solver (possibly calling LAPACK incorrectly)
3. Compute ALL eigenvalues when we only need λ₂

Using `np.linalg.eigvalsh()` directly is dramatically faster.

## Fix

Replace lines 40-64 in `cubic_graph_optimizer/core/spanning_trees.py`:

```python
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

    # Get adjacency matrix as dense array (fast for small graphs)
    A = nx.adjacency_matrix(G).toarray()

    # Compute eigenvalues (symmetric matrix, so use eigvalsh which is faster)
    eigenvalues = np.linalg.eigvalsh(A)

    # eigvalsh returns sorted in ascending order, so take second-last
    return eigenvalues[-2]
```

## Alternative: Sparse for Large Graphs

For very large graphs (N > 1000), use sparse solver to compute only top 2 eigenvalues:

```python
def get_second_eigenvalue(G):
    """Optimized for large graphs using sparse solver."""
    if not nx.is_connected(G):
        return float('inf')

    from scipy.sparse.linalg import eigsh

    A = nx.adjacency_matrix(G)  # Keep as sparse

    # Compute only the 2 largest eigenvalues
    eigenvalues = eigsh(A, k=2, which='LA', return_eigenvectors=False)

    # Returns sorted in ascending order
    return eigenvalues[-2]  # Second largest
```

## Impact

**For N=150 expansion optimization:**
- **Before**: 150 restarts × 10s/eigenvalue × ~50 iterations = **2+ hours**
- **After**: 150 restarts × 0.0015s/eigenvalue × 50 iterations = **11 seconds**

**~650× speedup for the entire optimization!**

## Implementation

Choose based on graph size:
- **N < 500**: Use dense `np.linalg.eigvalsh()` (simplest, fastest)
- **N ≥ 500**: Use sparse `scipy.sparse.linalg.eigsh(k=2)` (memory efficient)

For this codebase (N ≤ 400), dense is perfect.

## Testing

Run the benchmark:
```bash
python3 benchmark_eigenvalues.py
```

Should show ~0.002s for dense method vs ~10s for NetworkX.
