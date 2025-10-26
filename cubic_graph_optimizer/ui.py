"""Gradio web interface for cubic graph optimizer."""

import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from pathlib import Path

from cubic_graph_optimizer.optimization.methods import optimize_with_restarts
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees
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


def optimize_graph(n, method, restarts, max_iterations, parallel, seed,
                  analyze, compare, save_result):
    """Run the optimization and return results."""

    # Validate inputs
    if n % 2 != 0:
        return None, "Error: N must be even (cubic graphs require even number of vertices)", None

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

    # Build status message
    k = n // 2
    status_lines = [
        "="*70,
        "OPTIMIZATION STARTED",
        "="*70,
        f"N={n} vertices (k={k})",
        f"Method: {method}",
        f"Restarts: {restarts_param}",
        f"Max iterations: {max_iterations}",
        f"Parallel: {parallel}",
        f"Random seed: {seed}",
        "="*70,
        "",
        "Running optimization... (this may take a moment)",
        ""
    ]
    status = "\n".join(status_lines)

    try:
        # For comparison, generate initial random graph with same seed
        if compare:
            G_initial = nx.random_regular_graph(3, n, seed=seed)
            T_initial = count_spanning_trees(G_initial)

        # Run optimization
        G_opt, T_opt, num_trials = optimize_with_restarts(
            n=n,
            method=method,
            restarts=restarts_param,
            parallel=parallel,
            base_seed=seed,
            verbose=False,
            max_iterations=max_iterations
        )

        # Build results message
        result_lines = [
            "="*70,
            "OPTIMIZATION COMPLETE",
            "="*70,
            f"Trials run: {num_trials}",
            f"Best ln(spanning trees): {T_opt:.6f}",
            f"Approx. spanning trees: {T_opt:.2e}",
            ""
        ]

        if compare:
            improvement = T_opt - T_initial
            result_lines.extend([
                "Comparison with random graph:",
                f"  Initial: {T_initial:.6f}",
                f"  Final:   {T_opt:.6f}",
                f"  Improvement: {improvement:.6f} ({100*improvement/T_initial:.2f}%)",
                ""
            ])

        # Check if it's a known special graph
        is_known, cage_name = check_known_cages(G_opt, k)
        if is_known:
            result_lines.append(f"Special graph detected: {cage_name}\n")

        # Add detailed analysis if requested
        if analyze:
            result_lines.extend([
                "="*70,
                "DETAILED ANALYSIS",
                "="*70,
                ""
            ])

            # Capture analysis output
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            lightweight = (n >= 60)
            analyze_graph_properties(
                G_opt,
                "Optimized Graph",
                compute_girth=not lightweight,
                compute_cycles=not lightweight,
                compute_automorphisms=(n <= 100)
            )

            analysis_output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            result_lines.append(analysis_output)

        # Save the graph if requested
        if save_result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("optimized_graphs")
            output_dir.mkdir(exist_ok=True)

            filename = f"cubic_n{n}_{method}_{timestamp}.pkl"
            filepath = output_dir / filename

            # Save with metadata
            G_opt.graph['optimization_method'] = method
            G_opt.graph['optimization_trials'] = num_trials
            G_opt.graph['timestamp'] = timestamp
            if compare:
                G_opt.graph['initial_T'] = T_initial
                G_opt.graph['improvement'] = T_opt - T_initial

            save_graph(G_opt, str(filepath), T_value=T_opt,
                      compute_all_properties=(n <= 60))

            result_lines.extend([
                "",
                f"Graph saved to: {filepath}"
            ])

        result_lines.append("="*70)
        results_text = "\n".join(result_lines)

        # Create visualization
        fig = visualize_graph(G_opt, f"Optimized Cubic Graph (N={n})")

        return fig, results_text, G_opt

    except Exception as e:
        error_msg = f"Error during optimization:\n{str(e)}"
        import traceback
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg, None


def create_interface():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Cubic Graph Optimizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Cubic Graph Optimizer

        Optimize spanning trees in cubic graphs using Whitehead flips.

        This tool finds cubic graphs (3-regular graphs) that maximize the number of spanning trees
        using local search with Whitehead flips (2-2 edge swaps).
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Optimization Parameters")

                n_input = gr.Number(
                    label="Number of Vertices (N)",
                    value=14,
                    precision=0,
                    info="Must be even (minimum 4)"
                )

                method_input = gr.Radio(
                    label="Optimization Method",
                    choices=["greedy", "first", "first-multifold", "sa", "sa-multifold"],
                    value="greedy",
                    info="greedy: best improvement, first: first improvement, first-multifold: first improvement with multi-flips, sa: simulated annealing, sa-multifold: multi-flip SA"
                )

                restarts_input = gr.Textbox(
                    label="Number of Restarts",
                    value="1",
                    info="Integer or 'auto' for automatic convergence detection"
                )

                max_iter_input = gr.Number(
                    label="Max Iterations per Restart",
                    value=100,
                    precision=0,
                    info="Maximum optimization iterations (default: 100 for greedy/first, 500 for sa/sa-multifold)"
                )

                seed_input = gr.Number(
                    label="Random Seed",
                    value=42,
                    precision=0,
                    info="For reproducibility"
                )

                gr.Markdown("### Options")

                parallel_input = gr.Checkbox(
                    label="Use Parallel Processing",
                    value=False,
                    info="Speed up multiple restarts using all CPU cores"
                )

                analyze_input = gr.Checkbox(
                    label="Detailed Analysis",
                    value=False,
                    info="Compute and display detailed graph properties"
                )

                compare_input = gr.Checkbox(
                    label="Compare with Random Graph",
                    value=False,
                    info="Show improvement over initial random graph"
                )

                save_input = gr.Checkbox(
                    label="Save Optimized Graph",
                    value=True,
                    info="Save graph to optimized_graphs/ directory"
                )

                optimize_btn = gr.Button("Run Optimization", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")

                results_text = gr.Textbox(
                    label="Optimization Results",
                    lines=20,
                    max_lines=30,
                    show_label=True,
                    interactive=False
                )

                graph_plot = gr.Plot(label="Optimized Graph Visualization")

        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                [14, "greedy", "1", 100, False, 42, False, False, True],
                [20, "first", "5", 100, True, 42, False, True, True],
                [20, "first-multifold", "5", 100, True, 42, False, True, True],
                [16, "sa", "auto", 500, True, 42, True, False, True],
                [20, "sa-multifold", "5", 500, True, 42, False, True, True],
                [30, "greedy", "10", 100, True, 42, False, False, False],
            ],
            inputs=[n_input, method_input, restarts_input, max_iter_input,
                   parallel_input, seed_input, analyze_input, compare_input, save_input],
            label="Click an example to load it"
        )

        # Connect the button
        optimize_btn.click(
            fn=optimize_graph,
            inputs=[n_input, method_input, restarts_input, max_iter_input,
                   parallel_input, seed_input, analyze_input, compare_input, save_input],
            outputs=[graph_plot, results_text, gr.State()]
        )

    return app


def launch_ui(share=False, server_name="127.0.0.1", server_port=7860):
    """Launch the Gradio interface."""
    app = create_interface()
    app.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_ui()
