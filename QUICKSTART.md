# Quick Start Guide

## 🚀 Launch the Web UI (Recommended!)

```bash
./launch_ui_enhanced
```

Open browser to: **http://localhost:7860**

### Try These First:

1. **Tab 1 - Single Objective**:
   - Set N=30, objective="spanning_trees", method="first", restarts=5
   - Click "Run Optimization"
   - See the **Tutte-Coxeter graph** with ln(trees)=23.86!

2. **Tab 2 - Alternating** ⭐:
   - Set N=20, max_cycles=10
   - Click "Run Alternating Optimization"
   - Watch it **oscillate** between two graphs!

## 📊 Command-Line Quick Tests

### Compare Both Objectives
```bash
python3 compare_objectives
```
Shows trade-off between spanning trees and expansion for N=10-40.

### Test Alternating Optimization
```bash
python3 test_alternating --n 30 --max-cycles 15
```
Generates beautiful trajectory plots!

### Analyze Symmetries
```bash
python3 analyze_symmetries --n 30
```
Shows automorphism groups for different optimization strategies.

## 🎯 Best Demonstrations

### 1. The Petersen Graph (N=10)
```bash
./optimize_trees 10 --objective expansion --method first --no-save
```
Finds λ₂ = 1.000 (the minimum possible!)

### 2. The Tutte-Coxeter Graph (N=30)
```bash
./optimize_trees 30 --objective spanning_trees --method first --restarts 5
```
Finds the highly symmetric graph with >1440 automorphisms!

### 3. Alternating Oscillation (N=20)
```bash
python3 test_alternating --n 20
```
Watch it cycle between two local optima!

### 4. Pareto Frontier (N=30)
```bash
python3 pareto_frontier --n 30 --alphas 15
```
Explores the trade-off curve (spoiler: it's flat!)

## 📖 Key Commands

| Task | Command |
|------|---------|
| **Web UI** | `./launch_ui_enhanced` |
| **Optimize (CLI)** | `./optimize_trees N --objective {spanning_trees,expansion}` |
| **Compare objectives** | `python3 compare_objectives` |
| **Alternating** | `python3 test_alternating --n N` |
| **Symmetry analysis** | `python3 analyze_symmetries --n N` |
| **Pareto frontier** | `python3 pareto_frontier --n N` |
| **Benchmarks** | `./benchmark_spanning_trees --quick` |

## 🎨 What to Expect

### Single-Objective Results

**N=10 (Petersen):**
- ln(trees) = 7.600902
- λ₂ = 1.000000 ⭐ minimum!

**N=30 (Tutte-Coxeter):**
- ln(trees) = 23.861626
- λ₂ = 2.000000
- >1440 symmetries ⭐

**N=40:**
- ln(trees) = 32.051464
- λ₂ = 2.290510

### Alternating Optimization

**Typical behaviors:**

- **N=10, 26, 30, 40, 60**: Immediate convergence (fixed points!)
- **N=20, 34, 50, 70**: 2-cycle oscillation
- **All**: Significant improvements (+10-40% in both objectives)

## 📚 Documentation

- **UI_README.md** - Detailed UI guide
- **EXPANSION_OPTIMIZATION.md** - Expansion objective details
- **ALTERNATING_OPTIMIZATION.md** - Theory and results
- **SESSION_SUMMARY.md** - Complete session overview
- **CHECKER_README.md** - Verification tools

## 🔬 Interesting Experiments

### Find Ramanujan Graphs
```bash
./optimize_trees 50 --objective expansion --restarts 10 --max-iterations 200
```
Try to minimize λ₂ below the Ramanujan bound (2.828).

### Test Cycle Length vs Graph Size
```bash
for n in 20 30 40 50 60; do
    python3 test_alternating --n $n --max-cycles 15 2>&1 | grep "Cycle length"
done
```

### Compare All Methods
```bash
./benchmark_spanning_trees --values 30 --methods first,first-multifold,sa-multifold --restarts 10
```

## 💡 Pro Tips

1. **Start with N=30** - it's fast and finds the beautiful Tutte-Coxeter graph
2. **Use the UI** - visualizations make everything clearer
3. **Try alternating on N=20** - the 2-cycle oscillation is mesmerizing
4. **Enable parallel** - much faster for multiple restarts
5. **Compare with random** - shows how much optimization helps

## ⚡ Performance Guide

| N Range | Speed | Recommended Iterations |
|---------|-------|----------------------|
| 4-20 | Very fast (< 1s) | 100 |
| 20-30 | Fast (1-3s) | 100-200 |
| 30-50 | Moderate (3-10s) | 150-200 |
| 50-70 | Slower (10-30s) | 200-300 |

## 🐛 Troubleshooting

**"ModuleNotFoundError"**: Activate conda environment:
```bash
conda activate graphs
```

**"N must be even"**: Cubic graphs require even vertex count

**UI not loading**: Check port 7860 is free, or use `--port 7861`

**Out of memory**: Try smaller N or disable `--parallel`

## 🎯 Your First Session

Try this sequence:

1. Launch UI: `./launch_ui_enhanced`
2. Go to **Alternating Optimization** tab
3. Set N=30, max_cycles=10
4. Click "Run Alternating Optimization"
5. Watch it **immediately converge** to Tutte-Coxeter!
6. Try N=20 and see the **2-cycle oscillation**
7. Explore different N values (10, 26, 40, 50)

Enjoy exploring the fascinating world of cubic graph optimization! 🔷
