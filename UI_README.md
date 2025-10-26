# Enhanced Gradio UI with Alternating Optimization

## Overview

The enhanced UI provides an interactive web interface for exploring cubic graph optimization with two main modes:

1. **Single-Objective Optimization** - Optimize for one objective (spanning trees or expansion)
2. **Alternating Optimization** - Alternate between both objectives to discover limit cycles!

## Launching the UI

```bash
# Basic launch (local only)
./launch_ui_enhanced

# With public share link
./launch_ui_enhanced --share

# Custom port
./launch_ui_enhanced --port 8080

# Custom server
./launch_ui_enhanced --server-name 0.0.0.0 --port 7860
```

Then open your browser to: `http://localhost:7860`

## Features

### Tab 1: Single-Objective Optimization

Optimize cubic graphs for a single objective:

**Parameters:**
- **N (vertices)**: Graph size (must be even, 4-100)
- **Objective**:
  - `spanning_trees`: Maximize ln(spanning trees) - combinatorial structure
  - `expansion`: Minimize λ₂ - spectral gap, better connectivity
- **Method**:
  - `first`: First-improvement (fast, recommended)
  - `greedy`: Best-improvement (slow, thorough)
  - `first-multifold`: First-improvement with occasional multi-flip jumps
  - `sa`: Simulated annealing
  - `sa-multifold`: SA with multi-flip moves
- **Restarts**: Number of random restarts (or "auto")
- **Max Iterations**: Iterations per restart
- **Parallel**: Use multiprocessing for restarts
- **Seed**: Random seed for reproducibility

**Options:**
- **Detailed Analysis**: Show graph properties (diameter, clustering, etc.)
- **Compare with Random**: Show improvement over random baseline
- **Save Result**: Save optimized graph to `optimized_graphs/`

**Output:**
- Text results with objectives, improvements, special graph detection
- Graph visualization with spring layout

### Tab 2: Alternating Optimization ⭐ NEW!

The most exciting feature! Watch the optimization **oscillate** between objectives:

**What it does:**
1. Start with random graph
2. Optimize spanning trees → local optimum
3. Optimize expansion → local optimum
4. Go to step 2
5. Continue until:
   - **Converged**: Both objectives stabilize (rare!)
   - **Cycled**: Returns to previous state (common!)
   - **Max cycles**: Limit reached

**Parameters:**
- **N (vertices)**: Graph size
- **Max Cycles**: Maximum alternation cycles (1-30)
- **Max Iterations per Phase**: How long to optimize each objective
- **Random Seed**: For reproducibility
- **Save Result**: Save final graph

**Visualizations:**

1. **Trajectory Plot** (left):
   - Top: Both objectives over time
   - Blue background: Optimizing trees
   - Red background: Optimizing expansion
   - Watch them oscillate!

2. **Objective Space Plot** (right):
   - Shows path through (ln(trees), λ₂) space
   - Green dot: Start
   - Red dot: End
   - Blue segments: Trees optimization phase
   - Red segments: Expansion optimization phase
   - Gray line: Ramanujan bound (λ₂ ≤ 2√2)

3. **Final Graph Visualization** (bottom right):
   - The converged/cycled graph

**Output:**
- Detailed trajectory table showing every step
- Convergence/cycle detection
- Total improvements from random start
- Ramanujan graph check

## Example Workflows

### Find Maximum Spanning Trees for N=30

1. Go to **Single Objective** tab
2. Set N=30, objective="spanning_trees", method="first"
3. Set restarts=10 for better results
4. Click "Run Optimization"
5. Result: Likely finds the **Tutte-Coxeter graph** with ln(trees)=23.86, λ₂=2.0!

### Discover Oscillation Behavior

1. Go to **Alternating Optimization** tab
2. Set N=20, max_cycles=15
3. Click "Run Alternating Optimization"
4. Watch the trajectory plot - you'll see it **oscillate between 2 states**!
5. The objective space plot shows a beautiful back-and-forth pattern

### Find Optimal Expanders

1. Go to **Single Objective** tab
2. Set N=40, objective="expansion"
3. Set restarts=10, max_iterations=200
4. Compare with random graph to see improvement
5. Check if result is Ramanujan (λ₂ ≤ 2.828)

### Explore Fixed Points

1. Try N=10, 26, 30, 40, or 60 in **Alternating** mode
2. These sizes often find graphs where **both objectives agree**!
3. The trajectory will show immediate convergence
4. These are highly symmetric graphs with special eigenvalues

## What to Look For

### In Single-Objective Mode:

- **Special graph detection**: Petersen (N=10), Tutte-Coxeter (N=30)
- **Integer eigenvalues**: λ₂ = 1.0, √2, 2.0 indicate high symmetry
- **Ramanujan achievement**: λ₂ ≤ 2.828 is optimal
- **Improvement percentages**: See how much better than random

### In Alternating Mode:

- **Cycle length**: Does it oscillate between 2 graphs? 3? More?
- **Fixed points**: Does it converge immediately (cycle length 0)?
- **Objective trade-off**: Does one improve while the other worsens?
- **Improvements**: Total Δln(trees) and Δλ₂ from random start

### Special Behaviors by N:

| N  | Expected Behavior | Special Features |
|----|------------------|------------------|
| 10 | Immediate convergence | Petersen graph, λ₂=1.0 (minimum!) |
| 14 | 1-cycle | λ₂ = √2 ≈ 1.414 |
| 20 | 2-cycle | Oscillates between nearby optima |
| 26 | Immediate convergence | λ₂ = 2.0 |
| 30 | Immediate convergence | Tutte-Coxeter, >1440 symmetries! |
| 40 | Immediate convergence | High symmetry |
| 50 | 2-cycle | Complex oscillation |
| 60 | Immediate convergence | Fixed point |

## Tips for Best Results

1. **Start with small N** (10-30) to see results quickly
2. **Use parallel=True** with multiple restarts for better solutions
3. **Try alternating on N=30** to see the Tutte-Coxeter graph
4. **Compare objectives** - run both spanning_trees and expansion on same N
5. **Watch the animations** in alternating mode - they're mesmerizing!
6. **Save interesting results** - graphs are saved with timestamps
7. **Use higher restarts** (10-20) for publication-quality results

## Understanding the Visualizations

### Trajectory Plot (Alternating Mode)

- **Y-axes**: Blue (left) = ln(trees), Red (right) = λ₂
- **X-axis**: Optimization steps
- **Vertical bars**: Phase transitions
  - Blue bar: Switched to trees optimization
  - Red bar: Switched to expansion optimization
- **Pattern**: Watch for oscillation vs convergence!

### Objective Space Plot (Alternating Mode)

- **X-axis**: ln(spanning trees) (combinatorial)
- **Y-axis**: λ₂ second eigenvalue (spectral)
- **Path**: Shows walk through objective space
- **Colors**: Blue = trees phase, Red = expansion phase
- **Annotations**: Cycle numbers at key points
- **Gray line**: Ramanujan bound (optimal expanders below this)

### Graph Visualization

- **Layout**: Spring layout (physics-based)
- **Nodes**: Vertices (light blue circles)
- **Edges**: Graph edges (black lines)
- **Labels**: Vertex numbers

For highly symmetric graphs (like N=30), the layout may show beautiful regular patterns!

## Performance Notes

- **N < 30**: Very fast (< 1 second per optimization)
- **N = 30-50**: Fast (1-5 seconds)
- **N = 50-70**: Moderate (5-20 seconds)
- **N > 70**: Slower, but still feasible

Alternating optimization typically takes 2-5x longer than single-objective since it runs multiple optimizations in sequence.

## Troubleshooting

**"Error: N must be even"**: Cubic graphs require even vertex count

**UI not loading**: Check that port 7860 (or your chosen port) is not in use

**Slow performance**: Reduce max_iterations or use fewer restarts

**Out of memory**: Try smaller N (< 50) or disable parallel processing

## Advanced: Pareto Frontier

For command-line Pareto frontier exploration:

```bash
python3 pareto_frontier --n 30 --alphas 15 --restarts 5
```

This explores the trade-off curve between objectives using weighted combinations.

## References

See the main documentation:
- `EXPANSION_OPTIMIZATION.md` - Single-objective details
- `ALTERNATING_OPTIMIZATION.md` - Alternating optimization theory
- `README.md` - General package information

## Future Enhancements

Potential additions to the UI:
- Real-time optimization progress bars
- Comparison of multiple N values side-by-side
- Export trajectory data to CSV
- Interactive graph editor
- Symmetry group visualization
- Pareto frontier explorer tab
