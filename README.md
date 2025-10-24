# Cubic Graph Optimizer

A Python package for optimizing the number of spanning trees in cubic graphs using Whitehead flips.

## Overview

This package implements several optimization algorithms to maximize the number of spanning trees in cubic (3-regular) graphs. The optimization is performed using Whitehead flips, which are edge-swap operations that preserve the cubic structure of the graph.

## Features

- **Multiple optimization methods**:
  - Greedy gradient ascent
  - First-improvement gradient ascent
  - Simulated annealing
  
- **Analysis tools**:
  - Spectral analysis and eigenvalue distributions
  - Ramanujan property checking
  - Automorphism group computation (with pynauty)
  - Comparison with known special graphs (Petersen, Heawood, etc.)

- **Performance features**:
  - Parallel processing support for multiple restarts
  - Automatic restart detection
  - Lightweight mode for large graphs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cubic_graph_optimizer.git
cd cubic_graph_optimizer

# Install dependencies
pip install -r requirements.txt

# Optional: Install pynauty for automorphism group computation
pip install pynauty
```

## Quick Start

```python
import networkx as nx
from cubic_graph_optimizer import (
    gradient_ascent_greedy,
    count_spanning_trees,
    analyze_graph_properties
)

# Create a random cubic graph
k = 10  # n = 2k vertices
G = nx.random_regular_graph(3, 2*k)

# Optimize using greedy method
G_opt, T_opt = gradient_ascent_greedy(G, max_iterations=100)

# Display results
print(f"Initial ln(spanning trees): {count_spanning_trees(G):.6f}")
print(f"Optimized ln(spanning trees): {T_opt:.6f}")

# Analyze properties
analyze_graph_properties(G_opt, "Optimized Graph")
```

## Usage Examples

### 1. Single Graph Optimization

```python
from cubic_graph_optimizer import optimize_with_restarts

# Optimize with multiple random restarts
G_best, T_best, num_trials = optimize_with_restarts(
    n=20,                    # Number of vertices
    method='greedy',         # or 'first', 'sa'
    restarts=10,            # Try 10 random starting graphs
    parallel=True,          # Use multiple cores
    verbose=True
)
```

### 2. Sweep Over Multiple Sizes

```python
from cubic_graph_optimizer import run_optimization_sweep

# Optimize graphs of different sizes
results = run_optimization_sweep(
    k_values=range(5, 20),   # k values (n = 2k)
    methods=['greedy', 'first'],
    restarts=5,
    parallel=True,
    save_graphs=True
)
```

### 3. Eigenvalue Distribution Analysis

```python
from cubic_graph_optimizer.analysis.spectral import analyze_eigenvalue_distribution

# Analyze how optimization affects eigenvalue distributions
results = analyze_eigenvalue_distribution(
    k_values=[10, 15, 20],
    samples_per_k=100,
    include_optimized=True,
    plot_histograms=True    # Requires matplotlib
)
```

### 4. Loading and Saving Graphs

```python
from cubic_graph_optimizer import save_graph, load_graph

# Save optimized graph with metadata
save_graph(G_opt, "optimal_cubic_k10.pkl", T_value=T_opt)

# Load later
G_loaded = load_graph("optimal_cubic_k10.pkl")
```

## Module Structure

- **`core/`**: Core functionality
  - `spanning_trees.py`: Computing ln(spanning trees) using Kirchhoff's theorem
  - `graph_operations.py`: Whitehead flips and graph manipulations

- **`optimization/`**: Optimization algorithms
  - `methods.py`: Gradient ascent, simulated annealing, parallel optimization

- **`analysis/`**: Analysis tools
  - `properties.py`: Graph property analysis and comparison
  - `spectral.py`: Eigenvalue analysis and McKay semicircle law
  - `automorphisms.py`: Symmetry and automorphism group computation

- **`utils/`**: Utilities
  - `validation.py`: Input validation and property checking
  - `io.py`: Graph I/O operations
  - `special_graphs.py`: Known cubic cages (Petersen, Heawood, etc.)

## Command Line Usage

### Quick Start

The package includes a convenient command-line tool:

```bash
# Basic usage - optimize a cubic graph with 20 vertices
./optimize_trees 20

# Use simulated annealing
./optimize_trees 20 --method sa

# Run multiple restarts in parallel
./optimize_trees 20 --restarts 10 --parallel

# Auto-detect when to stop trying new random starts
./optimize_trees 20 --restarts auto

# Compare with random graph and show detailed analysis
./optimize_trees 20 --compare --analyze

# Save to custom directory
./optimize_trees 20 --output-dir my_results/
```

### Command Line Options

```
positional arguments:
  N                     Number of vertices (must be even)

options:
  -h, --help            Show help message
  --method {greedy,first,sa}, -m
                        Optimization method (default: greedy)
  --restarts RESTARTS, -r
                        Number of restarts or "auto" (default: 1)
  --max-iterations MAX_ITERATIONS, -i
                        Maximum iterations per optimization
  --parallel, -p        Use parallel processing
  --output-dir OUTPUT_DIR, -o
                        Directory to save results (default: optimized_graphs/)
  --no-save             Do not save the optimized graph
  --analyze, -a         Show detailed analysis
  --compare, -c         Compare with initial random graph
  --seed SEED           Random seed (default: 42)
  --quiet, -q           Minimal output
  --verbose, -v         Verbose optimization output
```

### Output Files

The tool saves optimized graphs to the `optimized_graphs/` directory with:
- Graph file: `cubic_n{N}_{method}_{timestamp}.pkl`
- Summary file: `summary_n{N}_{timestamp}.txt`

### Examples Script

You can also run the examples script directly:

```bash
python main.py
```

## Theory Background

The package maximizes the number of spanning trees in cubic graphs, which is related to:
- Matrix Tree Theorem (Kirchhoff's theorem)
- Spectral graph theory
- Ramanujan graphs
- Graph automorphisms

The optimization uses Whitehead flips, which swap edges while preserving the cubic structure:
- Type 0: (a,b), (c,d) → (a,c), (b,d)
- Type 1: (a,b), (c,d) → (a,d), (b,c)

## Performance Tips

- For graphs with k ≥ 30, the code automatically uses lightweight mode
- Use `parallel=True` for multiple restarts
- Use `restarts='auto'` for automatic stopping when no improvement is found
- Simulated annealing often finds better solutions but takes longer

## Optional Dependencies

- **pynauty**: For computing automorphism groups (recommended)
  ```bash
  pip install pynauty
  ```
  
- **matplotlib**: For plotting eigenvalue distributions
  ```bash
  pip install matplotlib
  ```

## License

MIT License

## Citation

If you use this code in your research, please cite:
```
@software{cubic_graph_optimizer,
  title={Cubic Graph Optimizer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cubic_graph_optimizer}
}
```