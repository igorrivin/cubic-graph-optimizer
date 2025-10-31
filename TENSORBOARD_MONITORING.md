# TensorBoard Monitoring for Graph Optimization

## Why TensorBoard for Graph Optimization?

TensorBoard isn't just for ML! It's excellent for monitoring any long-running optimization:
- Real-time progress visualization
- Parallel worker tracking
- Convergence plots
- Hyperparameter comparison

## Installation

```bash
pip install tensorboard
```

## Implementation

### 1. Add TensorBoard Logger to Optimization Code

Modify `cubic_graph_optimizer/planar/triangulation.py`:

```python
def optimize_triangulation_multi_restart_parallel(
    n_points: int,
    objective_func,
    maximize: bool = True,
    restarts: int = 10,
    max_iterations: int = 200,
    n_jobs: int = -1,
    verbose: bool = False,
    seed: Optional[int] = None,
    tensorboard_logdir: Optional[str] = None,  # NEW PARAMETER
) -> Tuple[nx.Graph, nx.Graph, float, dict]:
    """
    ...
    Args:
        tensorboard_logdir: If provided, log progress to TensorBoard
    """
    # Setup TensorBoard writer if requested
    writer = None
    if tensorboard_logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tensorboard_logdir)

    # ... existing code ...

    completed = 0
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_single_restart_worker, worker_args):
            restart_id, G_tri_opt, G_dual_opt, final_value, stats = result

            all_restart_stats.append(stats)
            completed += 1

            # Log to TensorBoard
            if writer:
                writer.add_scalar('value/current', final_value, completed)
                writer.add_scalar('value/best', best_value, completed)
                writer.add_scalar('progress/completed', completed, completed)
                writer.add_scalar('flips/per_restart', stats['flips_performed'], completed)
                writer.flush()  # Important for real-time updates!

            # Check if this is the best so far
            is_better = (final_value > best_value) if maximize else (final_value < best_value)
            if is_better:
                best_G_tri = G_tri_opt
                best_G_dual = G_dual_opt
                best_value = final_value

                if writer:
                    writer.add_scalar('value/improvement', final_value - old_best, completed)

                if verbose:
                    print(f"[{completed:3d}/{restarts}] Restart {restart_id:3d}: {final_value:.6f} ⭐ NEW BEST", flush=True)
            elif verbose and completed % 10 == 0:
                print(f"[{completed:3d}/{restarts}] Completed {completed} restarts, best={best_value:.6f}", flush=True)

    if writer:
        writer.close()

    return best_G_tri, best_G_dual, best_value, summary
```

### 2. Usage Example

```python
from datetime import datetime

# Run optimization with TensorBoard logging
logdir = f"runs/planar_n150_expansion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

G_tri_best, G_dual_best, best_value, stats = optimize_triangulation_multi_restart_parallel(
    n_points=77,
    objective_func=objective_expansion,
    maximize=True,
    restarts=150,
    max_iterations=200,
    n_jobs=-1,
    verbose=True,
    tensorboard_logdir=logdir  # Enable TensorBoard!
)
```

### 3. Monitor in Real-Time

In a separate terminal:
```bash
tensorboard --logdir runs --port 6006
```

Then open http://localhost:6006 in your browser to see:
- Real-time convergence curves
- Best value tracking
- Flips per restart distribution
- Progress bars

### 4. Advanced: Per-Worker Monitoring

For even more detail, modify `_single_restart_worker` to log individual iterations:

```python
def _single_restart_worker(args):
    restart_id, n_points, objective_func, maximize, max_iterations = args

    # Each worker creates its own sub-logger
    # (TensorBoard handles concurrent writes)

    # ... optimization loop ...

    for iteration in range(max_iterations):
        # ... perform flip ...

        # Log iteration-level metrics (optional, creates more data)
        # writer.add_scalar(f'worker_{restart_id}/value', current_value, iteration)
```

## Alternative: Simple JSON Progress Log

If you want something lighter than TensorBoard:

```python
import json
from pathlib import Path

def log_progress(logfile, completed, total, best_value, current_value):
    """Append progress to JSON lines file."""
    with open(logfile, 'a') as f:
        json.dump({
            'timestamp': time.time(),
            'completed': completed,
            'total': total,
            'best_value': best_value,
            'current_value': current_value,
            'progress_pct': 100 * completed / total
        }, f)
        f.write('\n')
        f.flush()
```

Then monitor with a simple Python script:
```python
# monitor.py
import json
import time

logfile = 'progress.jsonl'
while True:
    with open(logfile) as f:
        for line in f:
            data = json.loads(line)
            print(f"Progress: {data['progress_pct']:.1f}% | Best: {data['best_value']:.6f}")
    time.sleep(1)
```

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| `flush=True` + `tail -f` | Zero dependencies, simple | No visualization, just text |
| TensorBoard | Beautiful real-time plots, compare runs | Requires torch/tensorflow |
| JSON log + custom viz | Lightweight, customizable | Need to build visualization |
| Weights & Biases | Cloud-based, team sharing | Requires account, not local |

## Recommendation

For long optimizations (>5 minutes), **use TensorBoard**:
1. Minimal code changes (`tensorboard_logdir` parameter)
2. Beautiful real-time visualization
3. Can compare different runs (n=150 vs n=200, etc.)
4. Works great for parallel computations
5. Zero configuration - just install and run
