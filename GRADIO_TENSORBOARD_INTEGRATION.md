# TensorBoard + Gradio Integration

## Three Approaches (from simplest to most integrated)

### Approach 1: Side-by-Side (Simplest - 2 minutes)

Run TensorBoard separately and link to it from Gradio:

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab("Optimize"):
        # ... your existing optimization UI ...
        pass

    with gr.Tab("Monitor"):
        gr.Markdown("""
        ## Live Optimization Monitoring

        TensorBoard is running at: [http://localhost:6006](http://localhost:6006)

        Open in a new tab to see real-time progress!
        """)

demo.launch()
```

Launch script:
```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir runs --port 6006 &

# Terminal 2: Start Gradio
python launch_ui_planar
```

**Pros**: Zero code changes, works immediately
**Cons**: Requires two browser tabs

---

### Approach 2: IFrame Embedding (Recommended - 10 minutes)

Embed TensorBoard directly in a Gradio tab:

```python
import gradio as gr
import subprocess
import time

# Start TensorBoard in background
def start_tensorboard(logdir='runs', port=6006):
    """Start TensorBoard server if not already running."""
    try:
        # Check if already running
        import requests
        requests.get(f'http://localhost:{port}', timeout=1)
        print(f"TensorBoard already running on port {port}")
    except:
        # Start TensorBoard
        subprocess.Popen(
            ['tensorboard', '--logdir', logdir, '--port', str(port), '--host', '0.0.0.0'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Wait for startup
        print(f"Started TensorBoard on port {port}")

with gr.Blocks() as demo:
    with gr.Tab("Optimize"):
        # ... your existing optimization UI ...
        pass

    with gr.Tab("Live Monitoring"):
        gr.Markdown("## Real-time Optimization Progress")

        # Embed TensorBoard via iframe
        gr.HTML("""
        <iframe
            src="http://localhost:6006"
            width="100%"
            height="800px"
            frameborder="0">
        </iframe>
        """)

# Start TensorBoard before launching Gradio
start_tensorboard(logdir='runs', port=6006)

demo.launch()
```

**Pros**: Single browser tab, fully integrated
**Cons**: Requires TensorBoard running in background

---

### Approach 3: Native Gradio Plots (Most Integrated - 30 minutes)

Read TensorBoard event files and plot with Gradio's native plotting:

```python
import gradio as gr
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import plotly.graph_objects as go
import glob
import time

def read_tensorboard_logs(logdir):
    """Read TensorBoard event files and extract metrics."""
    event_files = glob.glob(f"{logdir}/**/events.out.tfevents.*", recursive=True)

    if not event_files:
        return None

    # Read most recent event file
    latest_file = max(event_files, key=lambda p: os.path.getmtime(p))

    ea = EventAccumulator(latest_file)
    ea.Reload()

    # Extract scalars
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }

    return data

def create_progress_plot(logdir):
    """Create Plotly figure from TensorBoard logs."""
    data = read_tensorboard_logs(logdir)

    if not data:
        return go.Figure().add_annotation(
            text="No data yet - optimization hasn't started",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = go.Figure()

    # Plot best value over time
    if 'value/best' in data:
        fig.add_trace(go.Scatter(
            x=data['value/best']['steps'],
            y=data['value/best']['values'],
            mode='lines+markers',
            name='Best Value',
            line=dict(color='green', width=2)
        ))

    # Plot current values
    if 'value/current' in data:
        fig.add_trace(go.Scatter(
            x=data['value/current']['steps'],
            y=data['value/current']['values'],
            mode='markers',
            name='Current Value',
            marker=dict(size=4, opacity=0.5)
        ))

    fig.update_layout(
        title="Optimization Progress",
        xaxis_title="Restart #",
        yaxis_title="Objective Value",
        height=600
    )

    return fig

def optimize_with_monitoring(n_points, objective, restarts, max_iterations):
    """Run optimization with live monitoring."""
    from datetime import datetime

    # Create logdir
    logdir = f"runs/{objective.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start optimization in background thread
    import threading

    def run_optimization():
        # Your optimization code here, with tensorboard_logdir=logdir
        pass

    thread = threading.Thread(target=run_optimization)
    thread.start()

    # Return initial plot
    return create_progress_plot(logdir), logdir

with gr.Blocks() as demo:
    with gr.Tab("Optimize"):
        n_points = gr.Slider(10, 100, value=20, step=1, label="Graph Size (n_points)")
        objective = gr.Radio(["Spanning Trees", "Expansion"], label="Objective")
        restarts = gr.Slider(10, 200, value=50, step=10, label="Restarts")
        max_iter = gr.Slider(50, 500, value=200, step=50, label="Max Iterations")

        optimize_btn = gr.Button("Start Optimization")

        plot_output = gr.Plot(label="Live Progress")
        logdir_state = gr.State()

        # Auto-refresh plot every 2 seconds
        @gr.Timer(2)
        def refresh_plot(logdir):
            if logdir:
                return create_progress_plot(logdir)
            return None

        optimize_btn.click(
            optimize_with_monitoring,
            inputs=[n_points, objective, restarts, max_iter],
            outputs=[plot_output, logdir_state]
        )

        refresh_plot(logdir_state).then(lambda x: x, logdir_state, plot_output)

demo.launch()
```

**Pros**: Fully integrated, customizable plots, no iframe
**Cons**: More code, need to parse TensorBoard format

---

## Complete Example: Enhanced Planar UI with Monitoring

Here's a complete drop-in replacement for `ui_planar.py`:

```python
"""Gradio web interface for planar cubic graph optimization with TensorBoard monitoring."""

import gradio as gr
import subprocess
import time
import matplotlib
matplotlib.use('Agg')

from cubic_graph_optimizer.planar.triangulation import (
    optimize_triangulation_multi_restart_parallel,
    triangulation_to_dual_cubic,
)
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue

def start_tensorboard(logdir='runs', port=6006):
    """Start TensorBoard server."""
    try:
        import requests
        requests.get(f'http://localhost:{port}', timeout=1)
        print(f"✓ TensorBoard already running on port {port}")
    except:
        subprocess.Popen(
            ['tensorboard', '--logdir', logdir, '--port', str(port), '--host', '0.0.0.0'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        print(f"✓ Started TensorBoard on port {port}")

def optimize_with_tensorboard(n_points, objective, restarts, max_iterations, seed):
    """Run optimization with TensorBoard logging."""
    from datetime import datetime

    # Create unique logdir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = f"runs/planar_n{n_points}_{objective.lower()}_{timestamp}"

    # Define objective function
    if objective == "Spanning Trees":
        def objective_func(G_tri):
            return count_spanning_trees(G_tri)
        maximize = True
    else:  # Expansion
        def objective_func(G_tri):
            G_dual = triangulation_to_dual_cubic(G_tri)
            return -get_second_eigenvalue(G_dual)  # Negative to maximize
        maximize = True

    # Run optimization with TensorBoard logging
    G_tri_best, G_dual_best, best_value, stats = optimize_triangulation_multi_restart_parallel(
        n_points=n_points,
        objective_func=objective_func,
        maximize=maximize,
        restarts=restarts,
        max_iterations=max_iterations,
        n_jobs=-1,
        verbose=True,
        tensorboard_logdir=logdir  # Enable TensorBoard!
    )

    # Compute final metrics
    ln_trees = count_spanning_trees(G_dual_best)
    lambda2 = get_second_eigenvalue(G_dual_best)

    results = f"""
## ✅ Optimization Complete!

**Results:**
- ln(spanning trees): {ln_trees:.6f}
- λ₂: {lambda2:.6f}
- Total flips: {stats['total_flips']}
- Restarts: {restarts}

**View detailed metrics in the Monitoring tab**

TensorBoard logs saved to: `{logdir}`
"""

    return results

# Create Gradio interface
with gr.Blocks(title="Planar Cubic Graph Optimizer") as demo:
    gr.Markdown("# Planar Cubic Graph Optimization")

    with gr.Tab("⚙️ Optimize"):
        gr.Markdown("Configure and run optimization")

        with gr.Row():
            n_points = gr.Slider(10, 100, value=20, step=1, label="Graph Size (n_points)")
            objective = gr.Radio(["Spanning Trees", "Expansion"], value="Expansion", label="Objective")

        with gr.Row():
            restarts = gr.Slider(10, 200, value=50, step=10, label="Restarts")
            max_iter = gr.Slider(50, 500, value=200, step=50, label="Max Iterations/Restart")
            seed = gr.Number(value=42, label="Random Seed")

        optimize_btn = gr.Button("🚀 Start Optimization", variant="primary")

        output_text = gr.Markdown()

        optimize_btn.click(
            optimize_with_tensorboard,
            inputs=[n_points, objective, restarts, max_iter, seed],
            outputs=output_text
        )

    with gr.Tab("📊 Live Monitoring"):
        gr.Markdown("""
        ## Real-time Optimization Progress

        Watch your optimization in real-time! The plots update automatically as restarts complete.
        """)

        # Embed TensorBoard
        gr.HTML("""
        <iframe
            src="http://localhost:6006"
            width="100%"
            height="800px"
            frameborder="0"
            style="border: 1px solid #ddd; border-radius: 4px;">
        </iframe>
        """)

        gr.Markdown("""
        **💡 Tip:** Click on different metrics in the left sidebar to see:
        - `value/best` - Best objective found so far
        - `value/current` - Each restart's result
        - `flips/per_restart` - Optimization activity
        - `progress/completed` - Number of restarts done
        """)

    with gr.Tab("📖 Help"):
        gr.Markdown("""
        ## How to Use

        1. **Configure** your optimization in the "Optimize" tab
        2. Click **Start Optimization** (runs in background)
        3. Switch to **Live Monitoring** tab to watch progress
        4. Plots update in real-time as restarts complete!

        ## Objectives

        - **Spanning Trees**: Maximize ln(number of spanning trees)
        - **Expansion**: Minimize λ₂ (second eigenvalue) of dual graph

        ## Tips

        - More restarts = better results but longer runtime
        - For quick tests: 10-20 restarts
        - For best results: 100-200 restarts
        - Use monitoring to see when you hit diminishing returns
        """)

# Start TensorBoard before launching
start_tensorboard(logdir='runs', port=6006)

if __name__ == "__main__":
    demo.launch(share=False)
```

---

## Installation

```bash
pip install tensorboard gradio plotly
```

## Usage

```bash
python cubic_graph_optimizer/ui_planar.py
```

Then open `http://localhost:7860` and enjoy integrated monitoring!

---

## Comparison

| Approach | Setup Time | Integration | Updates | Best For |
|----------|-----------|-------------|---------|----------|
| Side-by-side | 2 min | Minimal | Real-time | Quick prototype |
| IFrame | 10 min | Good | Real-time | Production use |
| Native Plots | 30 min | Excellent | Custom | Full control |

## Recommendation

**Use Approach 2 (IFrame)** for the best balance of simplicity and integration!
