# Graph Checker Utility

Standalone tool for verifying that graphs have more spanning trees than a baseline threshold.

## Files

### Standalone Checker (Recommended for Sharing)
- **`check_graphs_standalone`** - Self-contained checker script
  - Only dependencies: `networkx`, `numpy`, `scipy`
  - No project-specific imports
  - Easy to share and use independently

### Test Data
- **`test_graphs.json`** - Sample test case with 10 cubic graphs (N=40)
  - 1 optimized graph (ln(trees) ≈ 32.05)
  - 9 random cubic graphs (ln(trees) ≈ 30.9 to 31.6)

## Usage

### Basic Check
```bash
./check_graphs_standalone --baseline-T 32.0 test_graphs.json
```

Output:
```
Graph Name                         N    ln(trees) Result
-----------------------------------------------------------------
optimized                         40    32.050294 PASS ✓
random_1                          40    31.436163 FAIL ✗
random_2                          40    31.027908 FAIL ✗
...
-----------------------------------------------------------------

Summary: 1/10 graphs beat baseline
```

### Save Results to JSON
```bash
./check_graphs_standalone --baseline-T 32.0 test_graphs.json --output results.json
```

### Verbose Mode
```bash
./check_graphs_standalone --baseline-T 32.0 test_graphs.json --verbose
```

## Input Format

The test graphs JSON file should contain an array of graphs in adjacency list format:

```json
[
  {
    "name": "graph_name",
    "adjacency_list": [
      [1, 2, 3],     // Node 0 connects to nodes 1, 2, 3
      [0, 4, 5],     // Node 1 connects to nodes 0, 4, 5
      ...
    ]
  },
  ...
]
```

Or simplified format (just adjacency lists):
```json
[
  [[1, 2, 3], [0, 4, 5], ...],
  [[1, 2, 3], [0, 4, 5], ...],
  ...
]
```

## Output Format

When using `--output`, creates a JSON file:

```json
{
  "baseline": "threshold_32.0000",
  "baseline_ln_trees": 32.0,
  "test_file": "test_graphs.json",
  "results": [
    {
      "name": "optimized",
      "n": 40,
      "ln_trees": 32.050294,
      "has_more_than_baseline": true
    },
    ...
  ],
  "summary": {
    "total": 10,
    "passed": 1,
    "failed": 9
  }
}
```

## Exit Codes
- `0` - All graphs passed (beat the baseline)
- `1` - One or more graphs failed

## Example: Setting the Threshold

The baseline threshold should be set just below your optimized graph's value:

1. Check your optimized graph:
   ```bash
   # Optimized graph has ln(trees) = 32.050294
   ```

2. Set threshold slightly below:
   ```bash
   ./check_graphs_standalone --baseline-T 32.0 test_graphs.json
   ```

3. Result: Optimized graph passes, random graphs fail ✓

## Dependencies

Install with:
```bash
pip install networkx numpy scipy
```

## Sharing

To share the checker:
1. Copy `check_graphs_standalone`
2. Copy your test data JSON file (e.g., `test_graphs.json`)
3. Include this README if helpful

Recipients only need Python 3.8+ with networkx, numpy, and scipy installed.
