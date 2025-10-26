"""Enhanced Gradio web interface with alternating optimization."""

import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from cubic_graph_optimizer.optimization.methods import optimize_with_restarts
from cubic_graph_optimizer.optimization.alternating import alternating_optimization
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue
from cubic_graph_optimizer.analysis.properties import analyze_graph_properties
from cubic_graph_optimizer.utils.special_graphs import check_known_cages
from cubic_graph_optimizer.utils.io import save_graph


def visualize_graph(G, title="Graph"):
    """Create a matplotlib figure of the graph."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42, k=1/G.number_of_nodes()**0.5, iterations=50)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=300, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    return fig


def optimize_single_objective(n, objective, method, restarts, max_iterations,
                              parallel, seed, analyze, compare, save_result):
    """Run single-objective optimization."""

    # Validate inputs
    if n % 2 != 0:
        return None, "Error: N must be even", None
    if n < 4:
        return None, "Error: N must be at least 4", None

    # Parse restarts
    if restarts == "auto":
        restarts_param = "auto"
    else:
        try:
            restarts_param = int(restarts)
            if restarts_param < 1:
                return None, "Error: Restarts must be positive or 'auto'", None
        except ValueError:
            return None, "Error: Restarts must be an integer or 'auto'", None

    try:
        # For comparison
        if compare:
            G_initial = nx.random_regular_graph(3, n, seed=seed)
            T_initial = count_spanning_trees(G_initial)
            lambda2_initial = get_second_eigenvalue(G_initial)

        # Run optimization
        G_opt, obj_value, num_trials = optimize_with_restarts(
            n=n,
            method=method,
            restarts=restarts_param,
            parallel=parallel,
            base_seed=seed,
            verbose=False,
            max_iterations=max_iterations,
            objective=objective
        )

        # Measure both objectives
        T_opt = count_spanning_trees(G_opt)
        lambda2_opt = get_second_eigenvalue(G_opt)
        ramanujan = lambda2_opt <= 2.828427

        # Build results
        result_lines = [
            "="*70,
            "OPTIMIZATION COMPLETE",
            "="*70,
            f"Objective: {objective}",
            f"Trials run: {num_trials}",
            "",
            "Final Results:",
            f"  ln(spanning trees): {T_opt:.6f}",
            f"  λ₂ (second eigenvalue): {lambda2_opt:.6f}",
            f"  Ramanujan graph: {ramanujan}",
            ""
        ]

        if compare:
            result_lines.extend([
                "Comparison with Random Graph:",
                f"  Initial ln(trees): {T_initial:.6f}",
                f"  Final ln(trees):   {T_opt:.6f}",
                f"  Improvement:       {T_opt - T_initial:+.6f} ({100*(T_opt-T_initial)/T_initial:.2f}%)",
                "",
                f"  Initial λ₂: {lambda2_initial:.6f}",
                f"  Final λ₂:   {lambda2_opt:.6f}",
                f"  Improvement: {lambda2_opt - lambda2_initial:+.6f} ({100*(lambda2_opt-lambda2_initial)/lambda2_initial:.2f}%)",
                ""
            ])

        # Check for special graphs
        special = check_known_cages(G_opt)
        if special:
            result_lines.append(f"Special graph detected: {special}")
            result_lines.append("")

        # Analysis
        if analyze:
            result_lines.append("="*70)
            result_lines.append("DETAILED ANALYSIS")
            result_lines.append("="*70)
            props = analyze_graph_properties(G_opt)
            for key, value in props.items():
                result_lines.append(f"  {key}: {value}")

        # Save
        if save_result:
            output_dir = Path("optimized_graphs")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"graph_n{n}_{objective}_{timestamp}.pkl"
            save_graph(G_opt, filename)
            result_lines.append(f"\nGraph saved to: {filename}")

        result_lines.append("="*70)
        results = "\n".join(result_lines)

        # Create visualization
        fig = visualize_graph(G_opt, f"Optimized Graph (N={n}, {objective})")

        return fig, results, G_opt

    except Exception as e:
        return None, f"Error during optimization: {str(e)}", None


def run_alternating_optimization(n, max_cycles, max_iter_per_phase, seed,
                                 save_result):
    """Run alternating optimization and return visualization."""

    # Validate
    if n % 2 != 0:
        return None, None, "Error: N must be even", None
    if n < 4:
        return None, None, "Error: N must be at least 4", None

    try:
        # Generate starting graph
        G_initial = nx.random_regular_graph(3, n, seed=seed)

        # Run alternating optimization
        result = alternating_optimization(
            G_initial,
            objectives=['spanning_trees', 'expansion'],
            max_cycles=max_cycles,
            max_iterations_per_phase=max_iter_per_phase,
            verbose=False
        )

        # Extract trajectory
        trajectory = result['trajectory']

        # Build results message
        result_lines = [
            "="*70,
            "ALTERNATING OPTIMIZATION COMPLETE",
            "="*70,
            f"Total cycles: {result['num_cycles']}",
            f"Converged: {result['converged']}",
            f"Cycled: {result['cycled']}",
            ""
        ]

        if result['cycled']:
            cycle_length = result['num_cycles'] - result['cycle_start']
            result_lines.extend([
                f"Cycle detected!",
                f"  Started at cycle: {result['cycle_start']}",
                f"  Cycle length: {cycle_length}",
                ""
            ])

        # Initial vs final
        initial = trajectory[0]
        final = trajectory[-1]

        result_lines.extend([
            "Initial State:",
            f"  ln(trees) = {initial['ln_trees']:.6f}",
            f"  λ₂        = {initial['lambda2']:.6f}",
            "",
            "Final State:",
            f"  ln(trees) = {final['ln_trees']:.6f}",
            f"  λ₂        = {final['lambda2']:.6f}",
            "",
            "Total Improvement:",
            f"  Δln(trees) = {final['ln_trees'] - initial['ln_trees']:+.6f}",
            f"  Δλ₂        = {final['lambda2'] - initial['lambda2']:+.6f}",
            ""
        ])

        # Ramanujan check
        ramanujan = final['lambda2'] <= 2.828427
        result_lines.append(f"Final graph is Ramanujan: {ramanujan}")
        result_lines.append("")

        # Trajectory table
        result_lines.extend([
            "="*70,
            "TRAJECTORY",
            "="*70,
            f"{'Step':<6} {'Cycle':<6} {'Phase':<20} {'ln(trees)':<14} {'λ₂':<12}",
            "-"*70
        ])

        for i, state in enumerate(trajectory):
            result_lines.append(
                f"{i:<6} {state['cycle']:<6} {state['phase']:<20} "
                f"{state['ln_trees']:<14.6f} {state['lambda2']:<12.6f}"
            )

        result_lines.append("="*70)

        # Save
        if save_result:
            output_dir = Path("optimized_graphs")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"graph_n{n}_alternating_{timestamp}.pkl"
            save_graph(result['final_graph'], filename)
            result_lines.append(f"\nGraph saved to: {filename}")

        results_text = "\n".join(result_lines)

        # Create visualizations
        fig1 = create_trajectory_plot(trajectory, n)
        fig2 = visualize_graph(result['final_graph'],
                              f"Final Graph (N={n}, Alternating)")

        return fig1, fig2, results_text, result['final_graph']

    except Exception as e:
        return None, None, f"Error: {str(e)}", None


def create_trajectory_plot(trajectory, n):
    """Create a 2-panel trajectory visualization."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    ln_trees_vals = [s['ln_trees'] for s in trajectory]
    lambda2_vals = [s['lambda2'] for s in trajectory]
    phases = [s['phase'] for s in trajectory]

    # Plot 1: Time series
    steps = list(range(len(trajectory)))

    ax1.plot(steps, ln_trees_vals, 'b-o', label='ln(trees)', markersize=8, linewidth=2)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('ln(spanning trees)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, lambda2_vals, 'r-s', label='λ₂', markersize=8, linewidth=2)
    ax1_twin.set_ylabel('λ₂ (second eigenvalue)', color='r', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # Mark phase transitions
    for i, phase in enumerate(phases):
        if phase == 'spanning_trees':
            ax1.axvline(i, color='blue', alpha=0.15, linewidth=3)
        elif phase == 'expansion':
            ax1.axvline(i, color='red', alpha=0.15, linewidth=3)

    ax1.set_title(f'Alternating Optimization Trajectory (N={n})',
                 fontsize=14, fontweight='bold')

    # Plot 2: Objective space trajectory
    ax2.plot(ln_trees_vals, lambda2_vals, 'k-o', markersize=8, alpha=0.6, linewidth=2)
    ax2.plot(ln_trees_vals[0], lambda2_vals[0], 'go', markersize=15,
            label='Start', zorder=10)
    ax2.plot(ln_trees_vals[-1], lambda2_vals[-1], 'ro', markersize=15,
            label='End', zorder=10)

    # Color segments by phase
    for i in range(len(trajectory) - 1):
        if phases[i] == 'spanning_trees':
            color = 'blue'
        elif phases[i] == 'expansion':
            color = 'red'
        else:
            color = 'gray'
        ax2.plot(ln_trees_vals[i:i+2], lambda2_vals[i:i+2],
                color=color, alpha=0.5, linewidth=4)

    # Ramanujan bound
    ramanujan = 2.828427
    ax2.axhline(ramanujan, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(min(ln_trees_vals), ramanujan + 0.05, 'Ramanujan bound',
            fontsize=10, alpha=0.7)

    ax2.set_xlabel('ln(spanning trees)', fontsize=12)
    ax2.set_ylabel('λ₂', fontsize=12)
    ax2.set_title('Trajectory in Objective Space', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    # Add cycle annotations
    for i in range(0, len(trajectory), max(1, len(trajectory)//5)):
        cycle = trajectory[i]['cycle']
        ax2.annotate(f'C{cycle}', (ln_trees_vals[i], lambda2_vals[i]),
                    xytext=(7, 7), textcoords='offset points',
                    fontsize=9, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    plt.tight_layout()
    return fig


def create_interface():
    """Create the enhanced Gradio interface with tabs."""

    with gr.Blocks(title="Cubic Graph Optimizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔷 Cubic Graph Optimizer

        Optimize cubic (3-regular) graphs for maximum spanning trees or minimum λ₂ (expansion).
        """)

        with gr.Tabs():
            # Tab 1: Single Objective
            with gr.Tab("Single Objective"):
                gr.Markdown("""
                ### Single-Objective Optimization
                Optimize for **spanning trees** (combinatorial) or **expansion** (spectral).
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Parameters")

                        n_single = gr.Slider(4, 100, value=20, step=2, label="N (vertices)")

                        objective_single = gr.Radio(
                            choices=["spanning_trees", "expansion"],
                            value="spanning_trees",
                            label="Objective",
                            info="spanning_trees: maximize ln(trees), expansion: minimize λ₂"
                        )

                        method_single = gr.Dropdown(
                            choices=["greedy", "first", "first-multifold", "sa", "sa-multifold"],
                            value="first",
                            label="Method"
                        )

                        restarts_single = gr.Textbox(value="5", label="Restarts")
                        max_iter_single = gr.Slider(10, 1000, value=100, step=10, label="Max Iterations")

                        with gr.Row():
                            parallel_single = gr.Checkbox(value=True, label="Parallel")
                            seed_single = gr.Number(value=42, label="Seed")

                        with gr.Row():
                            analyze_single = gr.Checkbox(value=False, label="Detailed Analysis")
                            compare_single = gr.Checkbox(value=True, label="Compare with Random")
                            save_single = gr.Checkbox(value=True, label="Save Result")

                        run_single_btn = gr.Button("🚀 Run Optimization", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        results_single = gr.Textbox(label="Results", lines=25, max_lines=40)
                        graph_single = gr.Plot(label="Optimized Graph")

                gr.Examples(
                    examples=[
                        [20, "spanning_trees", "first", "5", 100, True, 42, False, True, True],
                        [30, "expansion", "first", "10", 100, True, 42, False, True, True],
                        [20, "spanning_trees", "first-multifold", "5", 100, True, 42, False, True, True],
                    ],
                    inputs=[n_single, objective_single, method_single, restarts_single, max_iter_single,
                           parallel_single, seed_single, analyze_single, compare_single, save_single]
                )

                run_single_btn.click(
                    fn=optimize_single_objective,
                    inputs=[n_single, objective_single, method_single, restarts_single, max_iter_single,
                           parallel_single, seed_single, analyze_single, compare_single, save_single],
                    outputs=[graph_single, results_single, gr.State()]
                )

            # Tab 2: Alternating Optimization
            with gr.Tab("Alternating Optimization"):
                gr.Markdown("""
                ### Alternating Optimization
                Alternate between optimizing **spanning trees** and **expansion** until convergence or cycling!

                The algorithm repeatedly:
                1. Optimize spanning trees until local optimum
                2. Optimize expansion until local optimum
                3. Repeat until convergence or cycle detection
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Parameters")

                        n_alt = gr.Slider(4, 100, value=30, step=2, label="N (vertices)")
                        max_cycles = gr.Slider(1, 30, value=15, step=1, label="Max Cycles")
                        max_iter_phase = gr.Slider(10, 500, value=100, step=10,
                                                   label="Max Iterations per Phase")
                        seed_alt = gr.Number(value=42, label="Random Seed")
                        save_alt = gr.Checkbox(value=True, label="Save Result")

                        run_alt_btn = gr.Button("🔄 Run Alternating Optimization",
                                               variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        results_alt = gr.Textbox(label="Results", lines=25, max_lines=40)

                with gr.Row():
                    trajectory_plot = gr.Plot(label="Optimization Trajectory")
                    final_graph_plot = gr.Plot(label="Final Graph")

                gr.Examples(
                    examples=[
                        [20, 10, 100, 42, True],
                        [30, 15, 150, 42, True],
                        [40, 20, 200, 42, True],
                    ],
                    inputs=[n_alt, max_cycles, max_iter_phase, seed_alt, save_alt]
                )

                run_alt_btn.click(
                    fn=run_alternating_optimization,
                    inputs=[n_alt, max_cycles, max_iter_phase, seed_alt, save_alt],
                    outputs=[trajectory_plot, final_graph_plot, results_alt, gr.State()]
                )

        gr.Markdown("""
        ---
        ### About

        This tool explores cubic (3-regular) graphs using Whitehead flips.

        **Objectives:**
        - **Spanning Trees**: Maximize the number of spanning trees (combinatorial structure)
        - **Expansion**: Minimize λ₂ (spectral gap, random walk mixing)

        **Methods:**
        - `greedy`: Always take best flip (slow but thorough)
        - `first`: Take first improving flip (fast)
        - `first-multifold`: First improvement with occasional multi-flip jumps
        - `sa`: Simulated annealing
        - `sa-multifold`: SA with multi-flip moves

        **Ramanujan graphs**: Optimal expanders with λ₂ ≤ 2√2 ≈ 2.828
        """)

    return app


def launch_ui(share=False, server_name="127.0.0.1", server_port=7860):
    """Launch the enhanced Gradio interface."""
    app = create_interface()
    app.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_ui()
